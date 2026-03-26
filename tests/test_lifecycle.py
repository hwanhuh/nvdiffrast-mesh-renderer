import unittest
from unittest import mock

import torch

from nvdiffrast_mesh_renderer.config import build_argparser, config_from_args
from nvdiffrast_mesh_renderer.lifecycle import RendererCache


class RendererCacheTests(unittest.TestCase):
    def test_get_with_status_reports_miss_then_hit(self):
        cache = RendererCache.__new__(RendererCache)
        cache.device = torch.device("cpu")
        cache.logger = None
        cache._environment_service = object()
        cache._active_key = None
        cache._active_renderer = None
        config = config_from_args(build_argparser().parse_args(["example.glb"]))

        first_renderer = object()
        with mock.patch("nvdiffrast_mesh_renderer.lifecycle.SceneRenderer", return_value=first_renderer) as scene_renderer:
            renderer, created = cache.get_with_status(config)
            self.assertIs(renderer, first_renderer)
            self.assertTrue(created)

            renderer, created = cache.get_with_status(config)
            self.assertIs(renderer, first_renderer)
            self.assertFalse(created)
            self.assertEqual(scene_renderer.call_count, 1)


if __name__ == "__main__":
    unittest.main()
