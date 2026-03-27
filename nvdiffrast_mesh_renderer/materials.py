import json
import pathlib
import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from trimesh.visual import material as trimesh_material
from PIL import Image

from .math_utils import to_float_array
from .textures import TextureCache
from .types import MaterialData


@dataclass(frozen=True)
class GltfMaterialOverrides:
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    packed_orm: bool = False


_DOWNSCALE_RESAMPLE = getattr(Image, "Resampling", Image).BOX


def resize_texture_image(image: Image.Image | None, max_size: int) -> Image.Image | None:
    if image is None or max_size <= 0:
        return image
    if not isinstance(image, Image.Image):
        return image
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_size:
        return image
    scale = float(max_size) / float(longest_side)
    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        resample=_DOWNSCALE_RESAMPLE,
    )
    resized.format = image.format
    return resized


def resize_scene_material_textures(scene: trimesh.Scene, max_size: int) -> int:
    if max_size <= 0:
        return 0
    resized_count = 0
    resized_images: dict[int, Image.Image] = {}
    for mesh in scene.geometry.values():
        material = getattr(getattr(mesh, "visual", None), "material", None)
        if material is None:
            continue
        if isinstance(material, trimesh_material.PBRMaterial):
            texture_fields = (
                "baseColorTexture",
                "metallicRoughnessTexture",
                "normalTexture",
                "occlusionTexture",
                "emissiveTexture",
            )
        elif isinstance(material, trimesh_material.SimpleMaterial):
            texture_fields = ("image",)
        else:
            continue
        for field_name in texture_fields:
            image = getattr(material, field_name, None)
            if not isinstance(image, Image.Image):
                continue
            cache_key = id(image)
            resized = resized_images.get(cache_key)
            if resized is None:
                resized = resize_texture_image(image, max_size)
                resized_images[cache_key] = resized
                if resized is not image:
                    resized_count += 1
            setattr(material, field_name, resized)
    return resized_count


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


def _get_gltf_texture_identity(textures: list[object], texture_info: object) -> Optional[Tuple[str, int]]:
    if not isinstance(texture_info, dict):
        return None
    texture_index = texture_info.get("index")
    if not isinstance(texture_index, int) or not (0 <= texture_index < len(textures)):
        return None
    texture_def = textures[texture_index]
    if isinstance(texture_def, dict):
        source_index = texture_def.get("source")
        if isinstance(source_index, int):
            return ("source", source_index)
    return ("texture", texture_index)


def load_gltf_material_overrides(path: pathlib.Path) -> Dict[str, GltfMaterialOverrides]:
    try:
        header = load_gltf_header(path)
    except Exception:
        return {}
    if not isinstance(header, dict):
        return {}
    materials = header.get("materials", [])
    textures = header.get("textures", [])
    overrides, names, name_counts = {}, {}, {}
    for mesh_def in header.get("meshes", []):
        mesh_name = mesh_def.get("name", "GLTF")
        for primitive in mesh_def.get("primitives", []):
            geometry_name = trimesh.util.unique_name(mesh_name, names, counts=name_counts)
            names[geometry_name] = True
            material_idx = primitive.get("material")
            if isinstance(material_idx, int) and 0 <= material_idx < len(materials):
                material_def = materials[material_idx]
                pbr_def = material_def.get("pbrMetallicRoughness", {})
                mr_identity = _get_gltf_texture_identity(
                    textures,
                    pbr_def.get("metallicRoughnessTexture") if isinstance(pbr_def, dict) else None,
                )
                occlusion_identity = _get_gltf_texture_identity(textures, material_def.get("occlusionTexture"))
                overrides[geometry_name] = GltfMaterialOverrides(
                    normal_scale=get_gltf_texture_scalar(material_def, "normalTexture", "scale", 1.0),
                    occlusion_strength=get_gltf_texture_scalar(material_def, "occlusionTexture", "strength", 1.0),
                    packed_orm=mr_identity is not None and mr_identity == occlusion_identity,
                )
            else:
                overrides[geometry_name] = GltfMaterialOverrides()
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

    def extract(self, mesh, gltf_overrides: Optional[Tuple[float, float] | GltfMaterialOverrides] = None):
        visual = mesh.visual
        vertex_colors = None
        if gltf_overrides is None:
            overrides = GltfMaterialOverrides()
        elif isinstance(gltf_overrides, GltfMaterialOverrides):
            overrides = gltf_overrides
        else:
            normal_scale, occlusion_strength = map(float, gltf_overrides)
            overrides = GltfMaterialOverrides(normal_scale=normal_scale, occlusion_strength=occlusion_strength)
        if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None and len(visual.vertex_colors) == len(mesh.vertices):
            colors = torch.as_tensor(np.asarray(visual.vertex_colors, dtype=np.float32), dtype=torch.float32, device=self.device)
            colors = colors / 255.0 if colors.max().item() > 1.0 else colors
            vertex_colors = torch.cat([colors, torch.ones_like(colors[:, :1])], dim=-1) if colors.shape[1] == 3 else colors
        material = getattr(visual, "material", None)
        if material is None:
            return self._default(), vertex_colors
        if isinstance(material, trimesh_material.PBRMaterial):
            alpha_mode = (material.alphaMode or "OPAQUE").upper()
            base_color_factor = to_float_array(material.baseColorFactor, 4, [1.0, 1.0, 1.0, 1.0])
            base_color_mode = "RGB" if alpha_mode == "OPAQUE" and base_color_factor[3] >= 0.999 else "RGBA"
            has_pbr_signals = any([material.metallicFactor is not None, material.roughnessFactor is not None, material.metallicRoughnessTexture is not None, material.normalTexture is not None, material.occlusionTexture is not None, material.emissiveTexture is not None, material.emissiveFactor is not None])
            metallic_roughness_texture = self.cache.get_pil(material.metallicRoughnessTexture, srgb=False, mode="RGB")
            shared_packed_orm = (
                material.occlusionTexture is not None
                and material.metallicRoughnessTexture is not None
                and (overrides.packed_orm or material.occlusionTexture is material.metallicRoughnessTexture)
            )
            occlusion_texture = metallic_roughness_texture if shared_packed_orm else self.cache.get_pil(material.occlusionTexture, srgb=False, mode="L")
            return MaterialData(
                workflow="pbr" if has_pbr_signals else "diffuse",
                base_color_factor=torch.tensor(base_color_factor, dtype=torch.float32, device=self.device),
                base_color_texture=self.cache.get_pil(material.baseColorTexture, srgb=True, mode=base_color_mode),
                metallic_factor=1.0 if material.metallicFactor is None else float(material.metallicFactor),
                roughness_factor=1.0 if material.roughnessFactor is None else float(material.roughnessFactor),
                metallic_roughness_texture=metallic_roughness_texture,
                normal_texture=self.cache.get_pil(material.normalTexture, srgb=False, mode="RGB"),
                normal_scale=overrides.normal_scale,
                occlusion_texture=occlusion_texture,
                occlusion_strength=overrides.occlusion_strength,
                emissive_factor=torch.tensor(to_float_array(material.emissiveFactor, 3, [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device),
                emissive_texture=self.cache.get_pil(material.emissiveTexture, srgb=True, mode="RGB"),
                alpha_mode=alpha_mode,
                alpha_cutoff=0.5 if material.alphaCutoff is None else float(material.alphaCutoff),
                double_sided=bool(material.doubleSided),
                has_pbr_signals=has_pbr_signals,
            ), vertex_colors
        if isinstance(material, trimesh_material.SimpleMaterial):
            diffuse = to_float_array(material.diffuse, 4, [1.0, 1.0, 1.0, 1.0])
            alpha_mode = "BLEND" if diffuse[3] < 0.999 else "OPAQUE"
            return MaterialData(
                workflow="diffuse",
                base_color_factor=torch.tensor(diffuse, dtype=torch.float32, device=self.device),
                base_color_texture=self.cache.get_pil(material.image, srgb=True, mode="RGB" if alpha_mode == "OPAQUE" else "RGBA"),
                metallic_factor=0.0,
                roughness_factor=(2.0 / (float(material.glossiness) + 2.0)) ** 0.25 if material.glossiness is not None else 1.0,
                metallic_roughness_texture=None,
                normal_texture=None,
                normal_scale=1.0,
                occlusion_texture=None,
                occlusion_strength=1.0,
                emissive_factor=torch.zeros(3, dtype=torch.float32, device=self.device),
                emissive_texture=None,
                alpha_mode=alpha_mode,
                alpha_cutoff=0.5,
                double_sided=False,
                has_pbr_signals=False,
            ), vertex_colors
        return self._default(), vertex_colors
