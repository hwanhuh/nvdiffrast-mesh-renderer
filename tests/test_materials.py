import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image
from trimesh.visual import material as trimesh_material

from nvdiffrast_mesh_renderer.geometry_pass import GeometryPassRenderer
from nvdiffrast_mesh_renderer.materials import GltfMaterialOverrides, MaterialExtractor, load_gltf_material_overrides


class _FakeTextureCache:
    def __init__(self):
        self.calls = []
        self._textures = {}

    def get_pil(self, image, srgb, mode):
        if image is None:
            return None
        self.calls.append((image, srgb, mode))
        key = (id(image), srgb, mode)
        if key not in self._textures:
            self._textures[key] = object()
        return self._textures[key]


class MaterialOverrideTests(unittest.TestCase):
    def test_load_gltf_material_overrides_detects_packed_orm(self):
        material_doc = {
            "asset": {"version": "2.0"},
            "textures": [{"source": 0}, {"source": 0}, {"source": 1}],
            "materials": [
                {
                    "normalTexture": {"index": 2, "scale": 0.25},
                    "occlusionTexture": {"index": 1, "strength": 0.6},
                    "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}},
                }
            ],
            "meshes": [{"name": "PackedMesh", "primitives": [{"material": 0}]}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            gltf_path = Path(temp_dir) / "packed.gltf"
            gltf_path.write_text(json.dumps(material_doc), encoding="utf-8")

            overrides = load_gltf_material_overrides(gltf_path)

        override = overrides["PackedMesh"]
        self.assertAlmostEqual(override.normal_scale, 0.25)
        self.assertAlmostEqual(override.occlusion_strength, 0.6)
        self.assertTrue(override.packed_orm)


class MaterialExtractorTests(unittest.TestCase):
    def test_extract_reuses_single_texture_for_packed_orm(self):
        image = Image.new("RGB", (2, 2), color=(128, 96, 64))
        mesh = SimpleNamespace(
            vertices=np.zeros((3, 3), dtype=np.float32),
            visual=SimpleNamespace(
                material=trimesh_material.PBRMaterial(
                    metallicRoughnessTexture=image,
                    occlusionTexture=image,
                )
            ),
        )
        cache = _FakeTextureCache()
        extractor = MaterialExtractor(cache=cache, device=torch.device("cpu"))

        material, _ = extractor.extract(mesh, GltfMaterialOverrides(packed_orm=True))

        self.assertIs(material.occlusion_texture, material.metallic_roughness_texture)
        self.assertEqual(
            [(srgb, mode) for _image, srgb, mode in cache.calls],
            [(False, "RGB")],
        )


class PackedOrmSamplingTests(unittest.TestCase):
    def test_sample_material_channels_fetches_packed_orm_once(self):
        renderer = GeometryPassRenderer.__new__(GeometryPassRenderer)
        shared_texture = object()
        mesh = SimpleNamespace(
            material=SimpleNamespace(
                base_color_factor=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
                base_color_texture=None,
                emissive_factor=torch.zeros(3, dtype=torch.float32),
                emissive_texture=None,
                occlusion_texture=shared_texture,
                occlusion_strength=0.5,
                metallic_factor=0.5,
                roughness_factor=0.25,
                metallic_roughness_texture=shared_texture,
            )
        )
        rast = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
        sample_calls = []

        def _fake_sample(texture, uv, uv_da=None, boundary_mode="wrap", mip_level_bias=None):
            del uv, uv_da, boundary_mode, mip_level_bias
            if texture is None:
                return None
            sample_calls.append(texture)
            return torch.tensor([[[[0.8, 0.4, 0.6]]]], dtype=torch.float32)

        with mock.patch("nvdiffrast_mesh_renderer.geometry_pass.sample_texture", side_effect=_fake_sample):
            _base_rgba, _emissive, ao, roughness, metallic = renderer._sample_material_channels(mesh, rast, None, None, None)

        self.assertEqual(sample_calls, [shared_texture])
        self.assertTrue(torch.allclose(ao, torch.tensor([[[[0.9]]]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(roughness, torch.tensor([[[[0.1]]]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(metallic, torch.tensor([[[[0.3]]]], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
