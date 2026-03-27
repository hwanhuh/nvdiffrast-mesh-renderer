import unittest
from types import SimpleNamespace

import torch

from nvdiffrast_mesh_renderer.beauty import RenderModeRenderer


class _ZeroIbl:
    def diffuse(self, normal):
        return torch.zeros_like(normal)

    def specular(self, normal, view_dir, roughness):
        del view_dir, roughness
        return torch.zeros_like(normal)


class _OneIbl:
    def diffuse(self, normal):
        return torch.ones_like(normal)

    def specular(self, normal, view_dir, roughness):
        del view_dir, roughness
        return torch.ones_like(normal)


class BeautyShadingTests(unittest.TestCase):
    def _make_renderer(self, *, ibl):
        renderer = RenderModeRenderer.__new__(RenderModeRenderer)
        renderer.config = SimpleNamespace(env_usage="light")
        renderer.lights = []
        renderer.ibl = ibl
        renderer.geometry = None
        renderer.compositor = None
        return renderer

    def _make_common_inputs(self):
        shape_rgb = (1, 1, 1, 3)
        shape_scalar = (1, 1, 1, 1)
        return {
            "normal": torch.tensor([[[[0.0, 0.0, 1.0]]]], dtype=torch.float32),
            "face_normal": torch.tensor([[[[0.0, 0.0, 1.0]]]], dtype=torch.float32),
            "view_dir": torch.tensor([[[[0.0, 0.0, 1.0]]]], dtype=torch.float32),
            "emissive": torch.zeros(shape_rgb, dtype=torch.float32),
            "roughness": torch.full(shape_scalar, 0.5, dtype=torch.float32),
            "metallic": torch.zeros(shape_scalar, dtype=torch.float32),
            "gbuf": SimpleNamespace(valid=torch.ones(shape_scalar, dtype=torch.bool)),
        }

    def test_pbr_keeps_small_ambient_floor_only_without_ibl(self):
        inputs = self._make_common_inputs()
        base_rgb = torch.full((1, 1, 1, 3), 0.5, dtype=torch.float32)
        ao = torch.ones((1, 1, 1, 1), dtype=torch.float32)

        without_ibl = self._make_renderer(ibl=None)._shade("pbr", base_rgb=base_rgb, ao=ao, **inputs)
        with_zero_ibl = self._make_renderer(ibl=_ZeroIbl())._shade("pbr", base_rgb=base_rgb, ao=ao, **inputs)

        self.assertGreater(float(without_ibl.max().item()), 0.0)
        self.assertTrue(torch.allclose(with_zero_ibl, torch.zeros_like(with_zero_ibl)))

    def test_indirect_specular_is_occluded_by_ao(self):
        inputs = self._make_common_inputs()
        base_rgb = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        low_ao = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float32)
        high_ao = torch.ones((1, 1, 1, 1), dtype=torch.float32)

        renderer = self._make_renderer(ibl=_OneIbl())
        low_result = renderer._shade("pbr", base_rgb=base_rgb, ao=low_ao, **inputs)
        high_result = renderer._shade("pbr", base_rgb=base_rgb, ao=high_ao, **inputs)

        self.assertGreater(float(high_result.max().item()), float(low_result.max().item()))


if __name__ == "__main__":
    unittest.main()
