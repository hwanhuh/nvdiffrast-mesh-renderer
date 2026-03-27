import unittest
from types import SimpleNamespace

import numpy as np
import torch

from nvdiffrast_mesh_renderer.scene_builder import SceneBuilder
from nvdiffrast_mesh_renderer.textures import TextureCache


class SceneBuilderClipPlaneTests(unittest.TestCase):
    def setUp(self):
        self.builder = SceneBuilder(TextureCache(torch.device("cpu")), torch.device("cpu"))

    def test_clip_planes_from_depth_range_track_visible_span(self):
        near, far = self.builder._clip_planes_from_depth_range(68.0, 179.0)

        self.assertAlmostEqual(near, 64.6)
        self.assertAlmostEqual(far, 187.95)

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


if __name__ == "__main__":
    unittest.main()
