import unittest

import torch

from nvdiffrast_mesh_renderer.compositor import LayerCompositor
from nvdiffrast_mesh_renderer.types import RenderImage


def make_image(rgb, alpha, depth, valid=None) -> RenderImage:
    width = len(depth)
    rgb_tensor = torch.tensor(rgb, dtype=torch.float32).view(1, 1, width, 3)
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32).view(1, 1, width, 1)
    depth_tensor = torch.tensor(depth, dtype=torch.float32).view(1, 1, width, 1)
    if valid is None:
        valid_tensor = alpha_tensor > 1e-5
    else:
        valid_tensor = torch.tensor(valid, dtype=torch.bool).view(1, 1, width, 1)
    return RenderImage(
        rgb=rgb_tensor,
        alpha=alpha_tensor,
        depth=depth_tensor,
        valid=valid_tensor,
    )


class LayerCompositorTests(unittest.TestCase):
    def test_merge_double_sided_opaque_picks_nearest_side_per_pixel(self):
        compositor = LayerCompositor()
        front = make_image(
            rgb=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            alpha=[1.0, 1.0],
            depth=[3.0, 1.0],
        )
        back = make_image(
            rgb=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            alpha=[1.0, 1.0],
            depth=[1.0, 4.0],
        )

        merged = compositor.merge_double_sided(front, back, "OPAQUE")

        expected_rgb = torch.tensor(
            [[[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        expected_depth = torch.tensor([[[[1.0], [1.0]]]], dtype=torch.float32)
        self.assertTrue(torch.equal(merged.valid, torch.ones_like(merged.valid, dtype=torch.bool)))
        self.assertTrue(torch.allclose(merged.rgb, expected_rgb))
        self.assertTrue(torch.allclose(merged.alpha, torch.ones_like(merged.alpha)))
        self.assertTrue(torch.allclose(merged.depth, expected_depth))

    def test_merge_double_sided_mask_picks_nearest_visible_side(self):
        compositor = LayerCompositor()
        front = make_image(
            rgb=[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            alpha=[1.0, 0.0],
            depth=[2.5, 0.5],
        )
        back = make_image(
            rgb=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            alpha=[1.0, 1.0],
            depth=[1.0, 3.0],
        )

        merged = compositor.merge_double_sided(front, back, "MASK")

        expected_rgb = torch.tensor(
            [[[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]],
            dtype=torch.float32,
        )
        expected_alpha = torch.tensor([[[[1.0], [1.0]]]], dtype=torch.float32)
        expected_depth = torch.tensor([[[[1.0], [3.0]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(merged.rgb, expected_rgb))
        self.assertTrue(torch.allclose(merged.alpha, expected_alpha))
        self.assertTrue(torch.allclose(merged.depth, expected_depth))
        self.assertTrue(torch.equal(merged.valid, torch.ones_like(merged.valid, dtype=torch.bool)))

    def test_merge_double_sided_blend_uses_nearest_contributor_depth(self):
        compositor = LayerCompositor()
        front = make_image(
            rgb=[[0.6, 0.0, 0.0]],
            alpha=[0.6],
            depth=[4.0],
        )
        back = make_image(
            rgb=[[0.0, 0.0, 0.4]],
            alpha=[0.4],
            depth=[1.5],
        )

        merged = compositor.merge_double_sided(front, back, "BLEND")

        expected_rgb = torch.tensor([[[[0.36, 0.0, 0.4]]]], dtype=torch.float32)
        expected_alpha = torch.tensor([[[[0.76]]]], dtype=torch.float32)
        expected_depth = torch.tensor([[[[1.5]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(merged.rgb, expected_rgb))
        self.assertTrue(torch.allclose(merged.alpha, expected_alpha))
        self.assertTrue(torch.allclose(merged.depth, expected_depth))
        self.assertTrue(torch.equal(merged.valid, torch.ones_like(merged.valid, dtype=torch.bool)))


if __name__ == "__main__":
    unittest.main()
