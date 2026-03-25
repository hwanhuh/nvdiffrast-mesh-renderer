import pathlib
from typing import List, Sequence, Tuple

import numpy as np
import torch
import trimesh

from .config import RenderConfig
from .geometry_utils import compute_face_normals, compute_vertex_tangents, scene_bounds
from .materials import MaterialExtractor, load_gltf_material_overrides
from .math_utils import look_at, orbit_camera, perspective, safe_normalize
from .textures import TextureCache
from .types import CameraData, MeshData


class SceneBuilder:
    def __init__(self, cache: TextureCache, device: torch.device):
        self.cache = cache
        self.device = device
        self.materials = MaterialExtractor(cache=cache, device=device)

    def load_meshes(self, path: pathlib.Path) -> List[MeshData]:
        overrides = load_gltf_material_overrides(path)
        scene_or_mesh = trimesh.load(path, force="scene", process=False)
        scene = scene_or_mesh if isinstance(scene_or_mesh, trimesh.Scene) else trimesh.Scene(scene_or_mesh)
        meshes = []
        for index, mesh in enumerate(scene.dump(concatenate=False)):
            if not isinstance(mesh, trimesh.Trimesh) or mesh.faces is None or len(mesh.faces) == 0:
                continue
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            uv = getattr(mesh.visual, "uv", None)
            uv = np.asarray(uv, dtype=np.float32) if uv is not None and len(uv) == len(vertices) else None
            tangents = compute_vertex_tangents(vertices, faces, uv, normals) if uv is not None else None
            material, vertex_colors = self.materials.extract(mesh, overrides.get(mesh.metadata.get("name", "")))
            meshes.append(
                MeshData(
                    name=mesh.metadata.get("name", f"mesh_{index}"),
                    positions=torch.as_tensor(vertices, dtype=torch.float32, device=self.device).contiguous(),
                    faces=torch.as_tensor(faces, dtype=torch.int32, device=self.device).contiguous(),
                    face_normals=torch.as_tensor(compute_face_normals(vertices, faces), dtype=torch.float32, device=self.device).contiguous(),
                    normals=torch.as_tensor(normals, dtype=torch.float32, device=self.device).contiguous(),
                    uv=None if uv is None else torch.as_tensor(uv, dtype=torch.float32, device=self.device).contiguous(),
                    tangents=None if tangents is None else torch.as_tensor(tangents, dtype=torch.float32, device=self.device).contiguous(),
                    vertex_colors=None if vertex_colors is None else vertex_colors.contiguous(),
                    material=material,
                )
            )
        return meshes

    def build_camera(self, meshes: Sequence[MeshData], config: RenderConfig) -> CameraData:
        center, radius = scene_bounds(meshes)
        eye, target, distance = orbit_camera(center, radius, config.elev, config.azim, config.fov, config.distance_scale, config.distance)
        near = max(0.01, distance - radius * 2.5)
        far = distance + radius * 2.5
        view = look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        proj = perspective(config.fov, 1.0, near, far)
        return CameraData(
            view=torch.as_tensor(view, dtype=torch.float32, device=self.device),
            proj=torch.as_tensor(proj, dtype=torch.float32, device=self.device),
            mvp=torch.as_tensor(proj @ view, dtype=torch.float32, device=self.device),
            position=torch.as_tensor(eye, dtype=torch.float32, device=self.device),
            cam_to_world=torch.as_tensor(np.linalg.inv(view), dtype=torch.float32, device=self.device),
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
