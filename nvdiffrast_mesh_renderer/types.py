from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class GpuTexture:
    tex: torch.Tensor
    mip: Optional[Sequence[torch.Tensor]]
    can_mip: bool
    max_mip_level: Optional[int]


@dataclass
class MaterialData:
    workflow: str
    base_color_factor: torch.Tensor
    base_color_texture: Optional[GpuTexture]
    metallic_factor: float
    roughness_factor: float
    metallic_roughness_texture: Optional[GpuTexture]
    normal_texture: Optional[GpuTexture]
    normal_scale: float
    occlusion_texture: Optional[GpuTexture]
    occlusion_strength: float
    emissive_factor: torch.Tensor
    emissive_texture: Optional[GpuTexture]
    alpha_mode: str
    alpha_cutoff: float
    double_sided: bool
    has_pbr_signals: bool


@dataclass
class MeshData:
    name: str
    positions: torch.Tensor
    positions_h: torch.Tensor
    faces: torch.Tensor
    face_normals: torch.Tensor
    normals: torch.Tensor
    uv: Optional[torch.Tensor]
    tangents: Optional[torch.Tensor]
    vertex_colors: Optional[torch.Tensor]
    material: MaterialData


@dataclass
class CameraData:
    view: torch.Tensor
    proj: torch.Tensor
    mvp: torch.Tensor
    position: torch.Tensor
    cam_to_world: torch.Tensor
    forward: torch.Tensor
    projection_type: str


@dataclass
class EnvironmentData:
    texture: GpuTexture
    light_intensity: float
    background_intensity: float


@dataclass
class GeometryBuffer:
    rast: torch.Tensor
    rast_db: Optional[torch.Tensor]
    valid: torch.Tensor
    triangle_id: torch.Tensor
    uvw: torch.Tensor
    barycentric: torch.Tensor
    world_pos: torch.Tensor
    view_pos: torch.Tensor
    normal_world: torch.Tensor
    normal_view: torch.Tensor
    face_normal_world: torch.Tensor
    face_normal_view: torch.Tensor
    uv: Optional[torch.Tensor]
    uv_da: Optional[torch.Tensor]
    tangent: Optional[torch.Tensor]
    vertex_color: Optional[torch.Tensor]
    base_rgba: torch.Tensor
    emissive: torch.Tensor
    ao: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    clip_pos: torch.Tensor
    tri: torch.Tensor
    side: str


@dataclass
class RenderLayer:
    mesh: MeshData
    geometry: GeometryBuffer


@dataclass
class RenderImage:
    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor
