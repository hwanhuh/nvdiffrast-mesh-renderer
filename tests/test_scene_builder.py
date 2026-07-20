import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from nvdiffrast_mesh_renderer.scene_builder import SceneBuilder
from nvdiffrast_mesh_renderer.textures import TextureCache


class SceneBuilderClipPlaneTests(unittest.TestCase):
    def setUp(self):
        self.builder = SceneBuilder(TextureCache(torch.device("cpu")), torch.device("cpu"))

    def test_clip_planes_from_depth_range_are_scale_covariant(self):
        base_depth_min = 1.216
        base_depth_max = 3.021
        for name, scale in (("normal", 1.0), ("tiny", 1e-3)):
            with self.subTest(name=name):
                depth_min = base_depth_min * scale
                depth_max = base_depth_max * scale
                near, far = self.builder._clip_planes_from_depth_range(depth_min, depth_max)

                self.assertTrue(math.isfinite(near))
                self.assertTrue(math.isfinite(far))
                self.assertGreater(near, 0.0)
                self.assertLess(near, depth_min)
                self.assertGreater(far, depth_max)
                self.assertAlmostEqual((depth_min - near) / depth_min, 0.05)
                self.assertAlmostEqual((far - depth_max) / depth_max, 0.05)
                self.assertAlmostEqual(near / scale, base_depth_min * 0.95)
                self.assertAlmostEqual(far / scale, base_depth_max * 1.05)

    def test_clip_planes_from_depth_range_guard_invalid_ranges(self):
        expected = self.builder._clip_planes_from_depth_range(0.001216, 0.003021)
        reversed_range = self.builder._clip_planes_from_depth_range(0.003021, 0.001216)
        self.assertEqual(reversed_range, expected)

        for depth_min, depth_max in ((math.nan, 1.0), (1.0, math.inf), (0.0, 0.0)):
            with self.subTest(depth_min=depth_min, depth_max=depth_max):
                near, far = self.builder._clip_planes_from_depth_range(depth_min, depth_max)
                self.assertTrue(math.isfinite(near))
                self.assertTrue(math.isfinite(far))
                self.assertGreater(near, 0.0)
                self.assertGreater(far, near)

    def test_clip_planes_from_meshes_ignore_points_behind_camera(self):
        positions_h = torch.tensor(
            [
                [0.0, 0.0, -68.0, 1.0],
                [0.0, 0.0, -179.0, 1.0],
                [0.0, 0.0, 10.0, 1.0],
            ],
            dtype=torch.float32,
        )
        mesh = SimpleNamespace(positions_h=positions_h)
        near, far = self.builder._clip_planes_from_meshes([mesh], np.eye(4, dtype=np.float32))

        self.assertAlmostEqual(near, 64.6)
        self.assertAlmostEqual(far, 187.95)

    def test_orthographic_projection_ignores_fov(self):
        center = np.zeros(3, dtype=np.float32)
        radius = 2.0
        config_narrow = SimpleNamespace(camera="orthographic", elev=0.0, azim=0.0, fov=15.0, distance=None, distance_scale=1.5)
        config_wide = SimpleNamespace(camera="orthographic", elev=0.0, azim=0.0, fov=80.0, distance=None, distance_scale=1.5)

        camera_narrow = self.builder.build_camera_from_bounds(center, radius, config_narrow)
        camera_wide = self.builder.build_camera_from_bounds(center, radius, config_wide)

        self.assertTrue(torch.allclose(camera_narrow.proj, camera_wide.proj))

    def test_orthographic_projection_scale_tracks_distance_not_fov(self):
        center = np.zeros(3, dtype=np.float32)
        radius = 2.0
        config_near = SimpleNamespace(camera="orthographic", elev=0.0, azim=0.0, fov=15.0, distance=3.0, distance_scale=1.0)
        config_far = SimpleNamespace(camera="orthographic", elev=0.0, azim=0.0, fov=80.0, distance=6.0, distance_scale=1.0)

        camera_near = self.builder.build_camera_from_bounds(center, radius, config_near)
        camera_far = self.builder.build_camera_from_bounds(center, radius, config_far)

        self.assertAlmostEqual(float(camera_near.proj[0, 0].item()), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(camera_far.proj[0, 0].item()), 1.0 / 6.0, places=6)

    def test_view_seeded_lights_are_deterministic(self):
        lights_a = self.builder.build_view_seeded_lights(1.1, camera_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32), view_seed=1234)
        lights_b = self.builder.build_view_seeded_lights(1.1, camera_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32), view_seed=1234)

        self.assertEqual(len(lights_a), len(lights_b))
        for (dir_a, color_a), (dir_b, color_b) in zip(lights_a, lights_b):
            self.assertTrue(torch.allclose(dir_a, dir_b))
            self.assertTrue(torch.allclose(color_a, color_b))

    def test_view_seeded_lights_vary_with_seed(self):
        lights_a = self.builder.build_view_seeded_lights(1.1, camera_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32), view_seed=1234)
        lights_b = self.builder.build_view_seeded_lights(1.1, camera_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32), view_seed=5678)

        serialized_a = [torch.cat([direction.reshape(-1), color.reshape(-1)]).tolist() for direction, color in lights_a]
        serialized_b = [torch.cat([direction.reshape(-1), color.reshape(-1)]).tolist() for direction, color in lights_b]
        self.assertNotEqual(serialized_a, serialized_b)


if __name__ == "__main__":
    unittest.main()
