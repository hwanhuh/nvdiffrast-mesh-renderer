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


class OglRenderModeTests(unittest.TestCase):
    def _make_renderer(self):
        renderer = RenderModeRenderer.__new__(RenderModeRenderer)
        renderer.config = SimpleNamespace(antialias=False)
        renderer.lights = []
        renderer.ibl = None
        renderer.geometry = None
        renderer.compositor = None
        renderer._render_buffer = lambda rgb, alpha, gbuf, **kwargs: SimpleNamespace(rgb=rgb, alpha=alpha, kwargs=kwargs)
        renderer._render_scalar_buffer = lambda value, alpha, gbuf, **kwargs: SimpleNamespace(value=value, alpha=alpha, kwargs=kwargs)
        return renderer

    def _make_layer(self, *, normal_world=None, world_pos=None, rast=None):
        shape_rgb = (1, 1, 1, 3)
        shape_scalar = (1, 1, 1, 1)
        normal_world = torch.tensor([[[[0.0, 0.0, 1.0]]]], dtype=torch.float32) if normal_world is None else normal_world
        world_pos = torch.tensor([[[[0.0, 0.0, 0.0]]]], dtype=torch.float32) if world_pos is None else world_pos
        rast = torch.tensor([[[[0.0, 0.0, 0.0, 1.0]]]], dtype=torch.float32) if rast is None else rast
        gbuf = SimpleNamespace(
            normal_world=normal_world,
            world_pos=world_pos,
            rast=rast,
            valid=torch.ones(shape_scalar, dtype=torch.bool),
            base_rgba=torch.ones((1, 1, 1, 4), dtype=torch.float32),
            tri=torch.zeros((1, 3), dtype=torch.int32),
        )
        material = SimpleNamespace(alpha_mode="OPAQUE", alpha_cutoff=0.5)
        return SimpleNamespace(mesh=SimpleNamespace(material=material), geometry=gbuf)

    def test_normal_ogl_matches_axis_swizzle_from_ogl_renderer(self):
        renderer = self._make_renderer()
        layer = self._make_layer()
        camera = SimpleNamespace(position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))

        result = renderer.render_mode("normal_ogl", layer, camera)

        expected = torch.tensor([[[[0.5, 0.0, 0.5]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result.rgb, expected))
        self.assertTrue(torch.allclose(result.alpha, torch.ones_like(result.alpha)))

    def test_depth_ogl_maps_ndc_depth_to_opengl_depth_buffer_range(self):
        renderer = self._make_renderer()
        layer = self._make_layer(rast=torch.tensor([[[[0.0, 0.0, 0.25, 1.0]]]], dtype=torch.float32))
        camera = SimpleNamespace(position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))

        result = renderer.render_mode("depth_ogl", layer, camera)

        expected = torch.tensor([[[[0.625]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result.value, expected))

    def test_position_ogl_applies_axis_swizzle_and_contrast_boost(self):
        renderer = self._make_renderer()
        world_pos = torch.tensor([[[[0.5, 0.25, -0.5]]]], dtype=torch.float32)
        layer = self._make_layer(world_pos=world_pos)
        camera = SimpleNamespace(position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))

        result = renderer.render_mode("position_ogl", layer, camera)

        expected = torch.tensor([[[[0.875, 0.875, 0.6875]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result.rgb, expected))

    def test_confidence_ogl_uses_normal_camera_dot_product(self):
        renderer = self._make_renderer()
        layer = self._make_layer()
        camera = SimpleNamespace(position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))

        result = renderer.render_mode("confidence_ogl", layer, camera)

        expected = torch.tensor([[[[1.0]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result.value, expected))

    def test_confidence_ogl_uses_constant_view_direction_for_orthographic_camera(self):
        renderer = self._make_renderer()
        normal_world = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
        layer = self._make_layer(normal_world=normal_world, world_pos=torch.tensor([[[[0.0, 0.0, 5.0]]]], dtype=torch.float32))
        camera = SimpleNamespace(
            projection_type="orthographic",
            forward=torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32),
            position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        )

        result = renderer.render_mode("confidence_ogl", layer, camera)

        expected = torch.tensor([[[[1.0]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result.value, expected))


if __name__ == "__main__":
    unittest.main()
