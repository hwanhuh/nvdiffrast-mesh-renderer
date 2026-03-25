import json
import pathlib
import struct
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from trimesh.visual import material as trimesh_material

from .math_utils import to_float_array
from .textures import TextureCache
from .types import MaterialData


def load_gltf_header(path: pathlib.Path) -> Optional[dict]:
    if path.suffix.lower() == ".gltf":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.suffix.lower() != ".glb":
        return None
    with path.open("rb") as handle:
        head = handle.read(20)
        if len(head) != 20:
            raise ValueError(f"Invalid GLB header in {path}")
        magic, version, _length, chunk_length, chunk_type = struct.unpack("<4sIIII", head)
        if magic != b"glTF" or version != 2 or chunk_type != 0x4E4F534A:
            raise ValueError(f"Unsupported GLB header in {path}")
        chunk = handle.read(chunk_length)
    return json.loads(chunk.decode("utf-8").rstrip("\x00"))


def get_gltf_texture_scalar(material_def: dict, texture_key: str, scalar_key: str, default: float) -> float:
    texture_info = material_def.get(texture_key)
    if not isinstance(texture_info, dict):
        return default
    try:
        return float(texture_info.get(scalar_key, default))
    except (TypeError, ValueError):
        return default


def load_gltf_material_overrides(path: pathlib.Path) -> Dict[str, Tuple[float, float]]:
    try:
        header = load_gltf_header(path)
    except Exception:
        return {}
    if not isinstance(header, dict):
        return {}
    materials = header.get("materials", [])
    overrides, names, name_counts = {}, {}, {}
    for mesh_def in header.get("meshes", []):
        mesh_name = mesh_def.get("name", "GLTF")
        for primitive in mesh_def.get("primitives", []):
            geometry_name = trimesh.util.unique_name(mesh_name, names, counts=name_counts)
            names[geometry_name] = True
            material_idx = primitive.get("material")
            if isinstance(material_idx, int) and 0 <= material_idx < len(materials):
                material_def = materials[material_idx]
                overrides[geometry_name] = (
                    get_gltf_texture_scalar(material_def, "normalTexture", "scale", 1.0),
                    get_gltf_texture_scalar(material_def, "occlusionTexture", "strength", 1.0),
                )
            else:
                overrides[geometry_name] = (1.0, 1.0)
    return overrides


class MaterialExtractor:
    def __init__(self, cache: TextureCache, device: torch.device):
        self.cache = cache
        self.device = device

    def _default(self) -> MaterialData:
        return MaterialData(
            workflow="diffuse",
            base_color_factor=torch.tensor([0.8, 0.8, 0.8, 1.0], dtype=torch.float32, device=self.device),
            base_color_texture=None,
            metallic_factor=0.0,
            roughness_factor=1.0,
            metallic_roughness_texture=None,
            normal_texture=None,
            normal_scale=1.0,
            occlusion_texture=None,
            occlusion_strength=1.0,
            emissive_factor=torch.zeros(3, dtype=torch.float32, device=self.device),
            emissive_texture=None,
            alpha_mode="OPAQUE",
            alpha_cutoff=0.5,
            double_sided=False,
            has_pbr_signals=False,
        )

    def extract(self, mesh, gltf_overrides: Optional[Tuple[float, float]] = None):
        visual = mesh.visual
        vertex_colors = None
        normal_scale, occlusion_strength = (1.0, 1.0) if gltf_overrides is None else map(float, gltf_overrides)
        if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None and len(visual.vertex_colors) == len(mesh.vertices):
            colors = torch.as_tensor(np.asarray(visual.vertex_colors, dtype=np.float32), dtype=torch.float32, device=self.device)
            colors = colors / 255.0 if colors.max().item() > 1.0 else colors
            vertex_colors = torch.cat([colors, torch.ones_like(colors[:, :1])], dim=-1) if colors.shape[1] == 3 else colors
        material = getattr(visual, "material", None)
        if material is None:
            return self._default(), vertex_colors
        if isinstance(material, trimesh_material.PBRMaterial):
            has_pbr_signals = any([material.metallicFactor is not None, material.roughnessFactor is not None, material.metallicRoughnessTexture is not None, material.normalTexture is not None, material.occlusionTexture is not None, material.emissiveTexture is not None, material.emissiveFactor is not None])
            return MaterialData(
                workflow="pbr" if has_pbr_signals else "diffuse",
                base_color_factor=torch.tensor(to_float_array(material.baseColorFactor, 4, [1.0, 1.0, 1.0, 1.0]), dtype=torch.float32, device=self.device),
                base_color_texture=self.cache.get_pil(material.baseColorTexture, srgb=True, mode="RGBA"),
                metallic_factor=1.0 if material.metallicFactor is None else float(material.metallicFactor),
                roughness_factor=1.0 if material.roughnessFactor is None else float(material.roughnessFactor),
                metallic_roughness_texture=self.cache.get_pil(material.metallicRoughnessTexture, srgb=False, mode="RGB"),
                normal_texture=self.cache.get_pil(material.normalTexture, srgb=False, mode="RGB"),
                normal_scale=normal_scale,
                occlusion_texture=self.cache.get_pil(material.occlusionTexture, srgb=False, mode="L"),
                occlusion_strength=occlusion_strength,
                emissive_factor=torch.tensor(to_float_array(material.emissiveFactor, 3, [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device),
                emissive_texture=self.cache.get_pil(material.emissiveTexture, srgb=True, mode="RGB"),
                alpha_mode=(material.alphaMode or "OPAQUE").upper(),
                alpha_cutoff=0.5 if material.alphaCutoff is None else float(material.alphaCutoff),
                double_sided=bool(material.doubleSided),
                has_pbr_signals=has_pbr_signals,
            ), vertex_colors
        if isinstance(material, trimesh_material.SimpleMaterial):
            diffuse = to_float_array(material.diffuse, 4, [1.0, 1.0, 1.0, 1.0])
            return MaterialData(
                workflow="diffuse",
                base_color_factor=torch.tensor(diffuse, dtype=torch.float32, device=self.device),
                base_color_texture=self.cache.get_pil(material.image, srgb=True, mode="RGBA"),
                metallic_factor=0.0,
                roughness_factor=(2.0 / (float(material.glossiness) + 2.0)) ** 0.25 if material.glossiness is not None else 1.0,
                metallic_roughness_texture=None,
                normal_texture=None,
                normal_scale=1.0,
                occlusion_texture=None,
                occlusion_strength=1.0,
                emissive_factor=torch.zeros(3, dtype=torch.float32, device=self.device),
                emissive_texture=None,
                alpha_mode="BLEND" if diffuse[3] < 0.999 else "OPAQUE",
                alpha_cutoff=0.5,
                double_sided=False,
                has_pbr_signals=False,
            ), vertex_colors
        return self._default(), vertex_colors
