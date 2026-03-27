import unittest
from types import SimpleNamespace

from nvdiffrast_mesh_renderer.cli import _render_all_mode_batch_sizes
from nvdiffrast_mesh_renderer.config import build_argparser, config_from_args
from nvdiffrast_mesh_renderer.renderer import SceneRenderer


class _FakeGeometry:
    def __init__(self, layers):
        self.layers = list(layers)
        self.calls = []

    def render_geometry_pass(self, mesh, camera):
        self.calls.append((mesh, camera))
        return list(self.layers)


class _FakeRegistry:
    def __init__(self):
        self.calls = []

    def render(self, mode, layer, camera):
        self.calls.append((mode, layer, camera))
        return f"{mode}:{layer}"


class _FakeCompositor:
    def __init__(self):
        self.merge_calls = []
        self.composite_calls = []

    def merge_double_sided(self, front, back, alpha_mode):
        self.merge_calls.append((front, back, alpha_mode))
        return f"merged({front},{back},{alpha_mode})"

    def composite_mesh_layers(self, bg_rgb, bg_alpha, layers):
        self.composite_calls.append((bg_rgb, bg_alpha, tuple(layers)))
        joined = "|".join(layers)
        return f"rgb[{joined}]", f"alpha[{joined}]"


class _FakePostprocessor:
    def __init__(self):
        self.calls = []

    def postprocess(self, rgb, alpha, render_mode=None):
        self.calls.append((rgb, alpha, render_mode))
        return {"mode": render_mode, "rgb": rgb, "alpha": alpha}


class RenderAllBatchingTests(unittest.TestCase):
    def test_render_prepared_modes_reuses_geometry_across_modes(self):
        renderer = SceneRenderer.__new__(SceneRenderer)
        renderer.config = SimpleNamespace(render_mode="beauty", double_sided_depth_peels=1)
        renderer.geometry = _FakeGeometry(["front", "back"])
        renderer.compositor = _FakeCompositor()
        renderer.postprocessor = _FakePostprocessor()
        registry = _FakeRegistry()
        renderer._build_render_registry = lambda prepared: registry

        mesh = SimpleNamespace(material=SimpleNamespace(alpha_mode="OPAQUE"))
        prepared = SimpleNamespace(
            meshes=[mesh],
            camera="camera",
            bg_rgb="bg_rgb",
            bg_alpha="bg_alpha",
        )

        images = renderer.render_prepared_modes(prepared, ("beauty", "mask"))

        self.assertEqual(renderer.geometry.calls, [(mesh, "camera")])
        self.assertEqual(
            registry.calls,
            [
                ("beauty", "front", "camera"),
                ("mask", "front", "camera"),
                ("beauty", "back", "camera"),
                ("mask", "back", "camera"),
            ],
        )
        self.assertEqual(
            renderer.compositor.merge_calls,
            [
                ("beauty:front", "beauty:back", "OPAQUE"),
                ("mask:front", "mask:back", "OPAQUE"),
            ],
        )
        self.assertEqual(
            renderer.compositor.composite_calls,
            [
                ("bg_rgb", "bg_alpha", ("merged(beauty:front,beauty:back,OPAQUE)",)),
                ("bg_rgb", "bg_alpha", ("merged(mask:front,mask:back,OPAQUE)",)),
            ],
        )
        self.assertEqual(images["beauty"]["mode"], "beauty")
        self.assertEqual(images["mask"]["mode"], "mask")


class RenderAllBatchConfigTests(unittest.TestCase):
    def test_render_all_mode_batch_sizes_disable_batching_for_benchmarks(self):
        config = SimpleNamespace(benchmark_requested=False, render_all_batch_size=4)
        self.assertEqual(_render_all_mode_batch_sizes(config, 6), (4, 3, 2, 1))
        benchmark_config = SimpleNamespace(benchmark_requested=True, render_all_batch_size=4)
        self.assertEqual(_render_all_mode_batch_sizes(benchmark_config, 6), (1,))

    def test_config_parses_render_all_batch_size(self):
        parser = build_argparser()
        args = parser.parse_args(["example.glb", "--render-all-batch-size", "0"])
        config = config_from_args(args)

        self.assertEqual(config.render_all_batch_size, 1)

    def test_config_parses_png_compression(self):
        parser = build_argparser()
        args = parser.parse_args(["example.glb", "--png-compression", "2"])
        config = config_from_args(args)

        self.assertEqual(config.png_compression, 2)

    def test_config_parses_texture_map_max_size(self):
        parser = build_argparser()
        args = parser.parse_args(["example.glb", "--texture-map-max-size", "2048"])
        config = config_from_args(args)

        self.assertEqual(config.texture_map_max_size, 2048)


if __name__ == "__main__":
    unittest.main()
