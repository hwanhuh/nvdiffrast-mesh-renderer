import pathlib
import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import trimesh

from .config import RenderConfig
from .geometry_utils import (
    compute_face_normals,
    compute_face_normals_torch,
    compute_vertex_tangents,
    compute_vertex_tangents_torch,
    scene_bounds,
)
from .materials import GltfMaterialOverrides, MaterialExtractor, load_gltf_material_overrides, resize_scene_material_textures
from .math_utils import look_at, orbit_camera, orthographic, perspective, safe_normalize, safe_normalize_np
from .textures import TextureCache
from .types import CameraData, MeshData


@dataclass(frozen=True)
class PreloadedSceneAsset:
    path: pathlib.Path
    scene: trimesh.Scene
    overrides: dict[str, GltfMaterialOverrides]


def preload_scene_asset(path: pathlib.Path, texture_map_max_size: int = 0) -> PreloadedSceneAsset:
    overrides = load_gltf_material_overrides(path)
    scene_or_mesh = trimesh.load(path, force="scene", process=False)
    scene = scene_or_mesh if isinstance(scene_or_mesh, trimesh.Scene) else trimesh.Scene(scene_or_mesh)
    resize_scene_material_textures(scene, max(int(texture_map_max_size), 0))
    return PreloadedSceneAsset(path=path, scene=scene, overrides=overrides)


class SceneBuilder:
    def __init__(self, cache: TextureCache, device: torch.device):
        self.cache = cache
        self.device = device
        self.materials = MaterialExtractor(cache=cache, device=device)
        self._geometry_preprocess_device = "auto"
        self._geometry_cuda_threshold_faces = 100000
        self._geometry_cuda_threshold_vertices = 100000
        self._texture_map_max_size = 0

    def load_meshes(self, path: pathlib.Path) -> List[MeshData]:
        return self.load_meshes_from_preloaded(preload_scene_asset(path, self._texture_map_max_size))

    def load_meshes_from_preloaded(self, preloaded: PreloadedSceneAsset) -> List[MeshData]:
        scene = preloaded.scene
        overrides = preloaded.overrides
        meshes = []
        used_names = {}
        used_name_counts = {}
        for index, (mesh_name, mesh, transform) in enumerate(self._iter_scene_meshes(scene)):
            if not isinstance(mesh, trimesh.Trimesh) or mesh.faces is None or len(mesh.faces) == 0:
                continue
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            uv = getattr(mesh.visual, "uv", None)
            uv = np.asarray(uv, dtype=np.float32) if uv is not None and len(uv) == len(vertices) else None
            vertices, normals = self._apply_transform(vertices, normals, transform)
            material, vertex_colors = self.materials.extract(mesh, overrides.get(mesh_name))
            use_cuda_preprocess = self._should_preprocess_on_cuda(vertices, faces)
            positions = torch.as_tensor(vertices, dtype=torch.float32, device=self.device).contiguous()
            positions_h = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=-1).contiguous()
            faces_t = torch.as_tensor(faces, dtype=torch.int32, device=self.device).contiguous()
            normals_t = torch.as_tensor(normals, dtype=torch.float32, device=self.device).contiguous()
            uv_t = None if uv is None else torch.as_tensor(uv, dtype=torch.float32, device=self.device).contiguous()
            if use_cuda_preprocess:
                face_normals = compute_face_normals_torch(positions, faces_t)
                tangents = compute_vertex_tangents_torch(positions, faces_t, uv_t, normals_t) if uv_t is not None else None
            else:
                tangents_np = compute_vertex_tangents(vertices, faces, uv, normals) if uv is not None else None
                face_normals = torch.as_tensor(compute_face_normals(vertices, faces), dtype=torch.float32, device=self.device).contiguous()
                tangents = None if tangents_np is None else torch.as_tensor(tangents_np, dtype=torch.float32, device=self.device).contiguous()
            instance_name = trimesh.util.unique_name(mesh_name or f"mesh_{index}", used_names, counts=used_name_counts)
            meshes.append(
                MeshData(
                    name=instance_name,
                    positions=positions,
                    positions_h=positions_h,
                    faces=faces_t,
                    face_normals=face_normals.contiguous(),
                    normals=normals_t,
                    uv=uv_t,
                    tangents=None if tangents is None else tangents.contiguous(),
                    vertex_colors=None if vertex_colors is None else vertex_colors.contiguous(),
                    material=material,
                )
            )
        return meshes

    def build_camera(self, meshes: Sequence[MeshData], config: RenderConfig) -> CameraData:
        center, radius = scene_bounds(meshes)
        eye, target, distance = orbit_camera(
            center,
            radius,
            config.elev,
            config.azim,
            config.fov,
            config.distance_scale,
            config.distance,
            projection_type=config.camera,
        )
        forward = safe_normalize_np(target - eye)
        view = look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        near, far = self._clip_planes_from_meshes(meshes, view)
        proj = self._projection_matrix(config, near, far, distance)
        return CameraData(
            view=torch.as_tensor(view, dtype=torch.float32, device=self.device),
            proj=torch.as_tensor(proj, dtype=torch.float32, device=self.device),
            mvp=torch.as_tensor(proj @ view, dtype=torch.float32, device=self.device),
            position=torch.as_tensor(eye, dtype=torch.float32, device=self.device),
            cam_to_world=torch.as_tensor(np.linalg.inv(view), dtype=torch.float32, device=self.device),
            forward=torch.as_tensor(forward, dtype=torch.float32, device=self.device),
            projection_type=config.camera,
        )

    def build_camera_from_bounds(self, center: np.ndarray, radius: float, config: RenderConfig) -> CameraData:
        eye, target, distance = orbit_camera(
            center,
            radius,
            config.elev,
            config.azim,
            config.fov,
            config.distance_scale,
            config.distance,
            projection_type=config.camera,
        )
        forward = safe_normalize_np(target - eye)
        near, far = self._clip_planes_from_depth_range(distance - radius, distance + radius)
        view = look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        proj = self._projection_matrix(config, near, far, distance)
        return CameraData(
            view=torch.as_tensor(view, dtype=torch.float32, device=self.device),
            proj=torch.as_tensor(proj, dtype=torch.float32, device=self.device),
            mvp=torch.as_tensor(proj @ view, dtype=torch.float32, device=self.device),
            position=torch.as_tensor(eye, dtype=torch.float32, device=self.device),
            cam_to_world=torch.as_tensor(np.linalg.inv(view), dtype=torch.float32, device=self.device),
            forward=torch.as_tensor(forward, dtype=torch.float32, device=self.device),
            projection_type=config.camera,
        )

    def build_lights(self, strength: float) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        specs = [
            ([0.65, 0.85, 1.0], [2.4, 2.2, 2.0]),
            ([-0.75, 0.25, 0.55], [0.9, 0.95, 1.1]),
            ([0.15, 0.55, -1.0], [0.35, 0.38, 0.48]),
        ]
        lights = []
        for direction, color in specs:
            direction_world = safe_normalize(torch.tensor(direction, device=self.device, dtype=torch.float32))
            light_color = torch.tensor(color, device=self.device, dtype=torch.float32)
            lights.append((direction_world.view(1, 1, 1, 3), light_color.view(1, 1, 1, 3) * strength))
        return lights

    def build_view_seeded_lights(
        self,
        strength: float,
        *,
        camera_direction: np.ndarray,
        view_seed: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random(int(view_seed))
        camera_dir = safe_normalize_np(np.asarray(camera_direction, dtype=np.float32).reshape(3))
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(camera_dir, world_up))) > 0.95:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        tangent = safe_normalize_np(np.cross(world_up, camera_dir))
        bitangent = safe_normalize_np(np.cross(camera_dir, tangent))
        remaining_strength = max(float(strength), 1e-3) * 1.5
        lights = []
        for _index in range(rng.randint(1, 3)):
            light_dir = safe_normalize_np(
                (camera_dir * rng.uniform(0.35, 1.0))
                + (tangent * rng.uniform(-1.0, 1.0))
                + (bitangent * rng.uniform(-0.75, 0.75))
                + (world_up * rng.uniform(-0.15, 0.35))
            )
            camera_ratio = max(float(np.dot(camera_dir, light_dir)) * 0.5 + 0.5, 0.0)
            max_energy = remaining_strength / max(float(np.dot(camera_dir, light_dir)) * 0.45 + 0.55, 0.1)
            light_strength = math.sqrt(rng.uniform(0.01, 1.0)) * max_energy
            remaining_strength = max(remaining_strength - (camera_ratio * light_strength), max(float(strength) * 0.15, 0.05))
            color = np.array(
                [
                    rng.uniform(0.94, 1.06),
                    rng.uniform(0.94, 1.06),
                    rng.uniform(0.94, 1.06),
                ],
                dtype=np.float32,
            ) * light_strength
            direction_world = safe_normalize(torch.tensor(light_dir, device=self.device, dtype=torch.float32))
            light_color = torch.tensor(color, device=self.device, dtype=torch.float32)
            lights.append((direction_world.view(1, 1, 1, 3), light_color.view(1, 1, 1, 3)))
        return lights

    def _iter_scene_meshes(self, scene: trimesh.Scene):
        if len(scene.geometry) == 0:
            return
        nodes = list(getattr(scene.graph, "nodes_geometry", []))
        if not nodes:
            for geom_name, mesh in scene.geometry.items():
                yield geom_name, mesh, np.eye(4, dtype=np.float32)
            return
        for node_name in nodes:
            transform, geom_name = scene.graph[node_name]
            mesh = scene.geometry.get(geom_name)
            if mesh is not None:
                yield geom_name, mesh, np.asarray(transform, dtype=np.float32)

    def _apply_transform(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        transform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.allclose(transform, np.eye(4, dtype=np.float32), atol=1e-8):
            return vertices, safe_normalize_np(normals.astype(np.float32, copy=False))
        linear = np.asarray(transform[:3, :3], dtype=np.float32)
        translate = np.asarray(transform[:3, 3], dtype=np.float32)
        normal_matrix = np.linalg.inv(linear).T.astype(np.float32)
        transformed_vertices = vertices @ linear.T + translate
        transformed_normals = safe_normalize_np(normals @ normal_matrix.T)
        return transformed_vertices.astype(np.float32, copy=False), transformed_normals.astype(np.float32, copy=False)

    def _should_preprocess_on_cuda(self, vertices: np.ndarray, faces: np.ndarray) -> bool:
        if self.device.type != "cuda":
            return False
        if self._geometry_preprocess_device == "cpu":
            return False
        if self._geometry_preprocess_device == "cuda":
            return True
        return len(faces) >= self._geometry_cuda_threshold_faces or len(vertices) >= self._geometry_cuda_threshold_vertices

    def configure_geometry_preprocess(self, config: RenderConfig) -> None:
        self._geometry_preprocess_device = config.geometry_preprocess_device
        self._geometry_cuda_threshold_faces = config.geometry_cuda_threshold_faces
        self._geometry_cuda_threshold_vertices = config.geometry_cuda_threshold_vertices
        self._texture_map_max_size = config.texture_map_max_size

    def _clip_planes_from_meshes(self, meshes: Sequence[MeshData], view: np.ndarray) -> tuple[float, float]:
        view_t = torch.as_tensor(view, dtype=torch.float32, device=self.device)
        positive_depth_min = None
        positive_depth_max = None
        for mesh in meshes:
            view_pos = torch.matmul(mesh.positions_h, view_t.t())[:, 2]
            depth = -view_pos
            visible_depth = depth[depth > 0.0]
            if visible_depth.numel() == 0:
                continue
            current_min = float(visible_depth.min().item())
            current_max = float(visible_depth.max().item())
            positive_depth_min = current_min if positive_depth_min is None else min(positive_depth_min, current_min)
            positive_depth_max = current_max if positive_depth_max is None else max(positive_depth_max, current_max)
        if positive_depth_min is None or positive_depth_max is None:
            return 0.01, 1.0
        return self._clip_planes_from_depth_range(positive_depth_min, positive_depth_max)

    def _clip_planes_from_depth_range(self, depth_min: float, depth_max: float) -> tuple[float, float]:
        depth_min = max(float(depth_min), 1e-3)
        depth_max = max(float(depth_max), depth_min + 1e-3)
        near = max(0.01, depth_min * 0.95)
        far = max(near + 1.0, depth_max * 1.05)
        return near, far

    def _projection_matrix(self, config: RenderConfig, near: float, far: float, distance: float) -> np.ndarray:
        if config.camera == "orthographic":
            return orthographic(distance, 1.0, near, far)
        return perspective(config.fov, 1.0, near, far)
