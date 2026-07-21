import pathlib
import math
import random
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
import trimesh
from trimesh.visual import material as trimesh_material

from .config import RenderConfig
from .geometry_utils import scene_bounds
from .materials import GltfMaterialOverrides, MaterialExtractor, load_gltf_material_overrides, resize_scene_material_textures
from .math_utils import look_at, orbit_camera, orthographic, perspective, safe_normalize, safe_normalize_np
from .textures import TextureCache
from .types import CameraData, MeshData


@dataclass(frozen=True)
class PreloadedSceneAsset:
    path: pathlib.Path
    scene: trimesh.Scene
    overrides: dict[str, GltfMaterialOverrides]
    texture_map_max_size: int


@dataclass(frozen=True)
class PreloadedMeshEntry:
    name: str
    mesh: trimesh.Trimesh
    transform: np.ndarray
    override: GltfMaterialOverrides


@dataclass(frozen=True)
class PreloadedSceneSummary:
    mesh_count: int
    pbr_count: int
    vertex_count: int
    face_count: int


def preload_scene_asset(path: pathlib.Path, texture_map_max_size: int = 0) -> PreloadedSceneAsset:
    texture_map_max_size = max(int(texture_map_max_size), 0)
    overrides = load_gltf_material_overrides(path)
    scene_or_mesh = trimesh.load(path, force="scene", process=False)
    scene = scene_or_mesh if isinstance(scene_or_mesh, trimesh.Scene) else trimesh.Scene(scene_or_mesh)
    resize_scene_material_textures(scene, texture_map_max_size)
    return PreloadedSceneAsset(
        path=path,
        scene=scene,
        overrides=overrides,
        texture_map_max_size=texture_map_max_size,
    )


def _iter_scene_meshes_standalone(scene: trimesh.Scene):
    if len(scene.geometry) == 0:
        return
    nodes = list(getattr(scene.graph, "nodes_geometry", []))
    if not nodes:
        for geom_name, mesh in scene.geometry.items():
            yield geom_name, mesh
        return
    for node_name in nodes:
        _transform, geom_name = scene.graph[node_name]
        mesh = scene.geometry.get(geom_name)
        if mesh is not None:
            yield geom_name, mesh


def summarize_scene_asset(preloaded: PreloadedSceneAsset) -> PreloadedSceneSummary:
    """Count meshes/vertices/faces without needing a CUDA device or renderer."""
    mesh_count = 0
    pbr_count = 0
    vertex_count = 0
    face_count = 0
    for _name, mesh in _iter_scene_meshes_standalone(preloaded.scene):
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces is None or len(mesh.faces) == 0:
            continue
        mesh_count += 1
        vertex_count += int(len(mesh.vertices))
        face_count += int(len(mesh.faces))
        material = getattr(getattr(mesh, "visual", None), "material", None)
        if isinstance(material, trimesh_material.PBRMaterial):
            has_pbr_signals = any(
                [
                    material.metallicFactor is not None,
                    material.roughnessFactor is not None,
                    material.metallicRoughnessTexture is not None,
                    material.normalTexture is not None,
                    material.occlusionTexture is not None,
                    material.emissiveTexture is not None,
                    material.emissiveFactor is not None,
                ]
            )
            if has_pbr_signals:
                pbr_count += 1
    return PreloadedSceneSummary(
        mesh_count=mesh_count,
        pbr_count=pbr_count,
        vertex_count=vertex_count,
        face_count=face_count,
    )


class SceneBuilder:
    def __init__(self, cache: TextureCache, device: torch.device, *, texture_map_max_size: int = 0):
        self.cache = cache
        self.device = device
        self.materials = MaterialExtractor(cache=cache, device=device)
        self._texture_map_max_size = max(int(texture_map_max_size), 0)

    def load_meshes(self, path: pathlib.Path) -> List[MeshData]:
        return self.load_meshes_from_preloaded(preload_scene_asset(path, self._texture_map_max_size))

    def load_meshes_from_preloaded(self, preloaded: PreloadedSceneAsset) -> List[MeshData]:
        return [self.load_mesh_from_entry(entry) for entry in self.iter_preloaded_mesh_entries(preloaded)]

    def iter_preloaded_mesh_entries(self, preloaded: PreloadedSceneAsset) -> Iterator[PreloadedMeshEntry]:
        used_names = {}
        used_name_counts = {}
        for index, (mesh_name, mesh, transform) in enumerate(self._iter_scene_meshes(preloaded.scene)):
            if not isinstance(mesh, trimesh.Trimesh) or mesh.faces is None or len(mesh.faces) == 0:
                continue
            instance_name = trimesh.util.unique_name(mesh_name or f"mesh_{index}", used_names, counts=used_name_counts)
            yield PreloadedMeshEntry(
                name=instance_name,
                mesh=mesh,
                transform=np.asarray(transform, dtype=np.float32),
                override=preloaded.overrides.get(mesh_name, GltfMaterialOverrides()),
            )

    def load_mesh_from_entry(self, entry: PreloadedMeshEntry) -> MeshData:
        mesh = entry.mesh
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        uv = getattr(mesh.visual, "uv", None)
        uv = np.asarray(uv, dtype=np.float32) if uv is not None and len(uv) == len(vertices) else None
        uv = self._sanitize_uv_coordinates(uv)
        vertices, normals = self._apply_transform(vertices, normals, entry.transform)
        material, vertex_colors = self.materials.extract(mesh, entry.override)
        positions = torch.as_tensor(vertices, dtype=torch.float32, device=self.device).contiguous()
        faces_t = torch.as_tensor(faces, dtype=torch.int32, device=self.device).contiguous()
        normals_t = torch.as_tensor(normals, dtype=torch.float32, device=self.device).contiguous()
        uv_t = None if uv is None else torch.as_tensor(uv, dtype=torch.float32, device=self.device).contiguous()
        return MeshData(
            name=entry.name,
            positions=positions,
            faces=faces_t,
            normals=normals_t,
            uv=uv_t,
            tangents=None,
            vertex_colors=None if vertex_colors is None else vertex_colors.contiguous(),
            material=material,
        )

    @staticmethod
    def _sanitize_uv_coordinates(uv: np.ndarray | None) -> np.ndarray | None:
        if uv is None:
            return None
        uv = np.asarray(uv, dtype=np.float32)
        if not np.isfinite(uv).all():
            uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.max(np.abs(uv), initial=0.0)) > 1e6:
            # Values this large lose useful fractional precision in float32 and
            # can fault CUDA texture sampling. Preserve only the wrapped tile.
            uv = np.remainder(uv.astype(np.float64), 1.0).astype(np.float32)
        return np.ascontiguousarray(uv)

    def summarize_preloaded(self, preloaded: PreloadedSceneAsset) -> PreloadedSceneSummary:
        mesh_count = 0
        pbr_count = 0
        vertex_count = 0
        face_count = 0
        for entry in self.iter_preloaded_mesh_entries(preloaded):
            mesh_count += 1
            vertex_count += int(len(entry.mesh.vertices))
            face_count += int(len(entry.mesh.faces))
            if self._workflow_for_material(getattr(getattr(entry.mesh, "visual", None), "material", None)) == "pbr":
                pbr_count += 1
        return PreloadedSceneSummary(
            mesh_count=mesh_count,
            pbr_count=pbr_count,
            vertex_count=vertex_count,
            face_count=face_count,
        )

    def bounds_from_preloaded(self, preloaded: PreloadedSceneAsset) -> tuple[np.ndarray, float]:
        bounds_min = None
        bounds_max = None
        for entry in self.iter_preloaded_mesh_entries(preloaded):
            vertices = np.asarray(entry.mesh.vertices, dtype=np.float32)
            if vertices.size == 0:
                continue
            transformed = self._apply_vertex_transform(vertices, entry.transform)
            current_min = transformed.min(axis=0)
            current_max = transformed.max(axis=0)
            bounds_min = current_min if bounds_min is None else np.minimum(bounds_min, current_min)
            bounds_max = current_max if bounds_max is None else np.maximum(bounds_max, current_max)
        if bounds_min is None or bounds_max is None:
            return np.zeros(3, dtype=np.float32), 1e-3
        center = ((bounds_min + bounds_max) * 0.5).astype(np.float32)
        radius = 1e-3
        for entry in self.iter_preloaded_mesh_entries(preloaded):
            vertices = np.asarray(entry.mesh.vertices, dtype=np.float32)
            if vertices.size == 0:
                continue
            transformed = self._apply_vertex_transform(vertices, entry.transform)
            radius = max(radius, float(np.linalg.norm(transformed - center, axis=-1).max(initial=0.0)))
        return center, max(radius, 1e-3)

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
        near, far = self._clip_planes_from_meshes(meshes, view, camera_position=eye)
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
        try:
            normal_matrix = np.linalg.inv(linear).T.astype(np.float32)
        except np.linalg.LinAlgError:
            # Some GLBs contain zero-scale scene nodes. Their geometry can still
            # be transformed, but the corresponding normal matrix is singular.
            normal_matrix = np.linalg.pinv(linear).T.astype(np.float32)
        transformed_vertices = vertices @ linear.T + translate
        transformed_normals = safe_normalize_np(normals @ normal_matrix.T)
        return transformed_vertices.astype(np.float32, copy=False), transformed_normals.astype(np.float32, copy=False)

    def _apply_vertex_transform(self, vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
        if np.allclose(transform, np.eye(4, dtype=np.float32), atol=1e-8):
            return vertices.astype(np.float32, copy=False)
        linear = np.asarray(transform[:3, :3], dtype=np.float32)
        translate = np.asarray(transform[:3, 3], dtype=np.float32)
        return (vertices @ linear.T + translate).astype(np.float32, copy=False)

    def _workflow_for_material(self, material) -> str:
        if isinstance(material, trimesh_material.PBRMaterial):
            has_pbr_signals = any(
                [
                    material.metallicFactor is not None,
                    material.roughnessFactor is not None,
                    material.metallicRoughnessTexture is not None,
                    material.normalTexture is not None,
                    material.occlusionTexture is not None,
                    material.emissiveTexture is not None,
                    material.emissiveFactor is not None,
                ]
            )
            return "pbr" if has_pbr_signals else "diffuse"
        return "diffuse"

    def _clip_planes_from_meshes(
        self,
        meshes: Sequence[MeshData],
        view: np.ndarray,
        camera_position: np.ndarray | None = None,
    ) -> tuple[float, float]:
        view_t = torch.as_tensor(view, dtype=torch.float32, device=self.device)
        if camera_position is None:
            rotation = np.asarray(view[:3, :3], dtype=np.float32)
            camera_position = -(rotation.T @ np.asarray(view[:3, 3], dtype=np.float32))
        camera_position_t = torch.as_tensor(
            camera_position,
            dtype=torch.float32,
            device=self.device,
        )
        positive_depth_min = None
        positive_depth_max = None
        for mesh in meshes:
            view_pos = torch.matmul(
                mesh.positions - camera_position_t.view(1, 3),
                view_t[:3, :3].t(),
            )
            depth = -view_pos[:, 2]
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
        fallback = (0.01, 1.0)
        try:
            depth_min = float(depth_min)
            depth_max = float(depth_max)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(depth_min) or not math.isfinite(depth_max):
            return fallback

        if depth_min > depth_max:
            depth_min, depth_max = depth_max, depth_min
        depth_scale = max(abs(depth_min), abs(depth_max))
        if depth_scale == 0.0 or depth_max <= 0.0:
            return fallback

        minimum_span = depth_scale * 1e-6
        if not math.isfinite(minimum_span) or minimum_span <= 0.0:
            return fallback
        if depth_min <= 0.0:
            depth_min = minimum_span

        near = depth_min * 0.95
        far = max(depth_max * 1.05, near + minimum_span)
        if not math.isfinite(near) or not math.isfinite(far) or near <= 0.0 or far <= near:
            return fallback
        return near, far

    def _projection_matrix(self, config: RenderConfig, near: float, far: float, distance: float) -> np.ndarray:
        if config.camera == "orthographic":
            return orthographic(distance, 1.0, near, far)
        return perspective(config.fov, 1.0, near, far)
