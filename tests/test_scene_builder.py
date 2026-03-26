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


if __name__ == "__main__":
    unittest.main()
