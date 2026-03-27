import pathlib
import unittest
from types import SimpleNamespace

from nvdiffrast_mesh_renderer.cli import (
    _is_multi_view,
    _multi_view_output_path,
    _multi_view_specs,
    _multiview_chunk_sizes,
    _multiview_render_modes,
    _render_all_mode_batch_sizes,
)
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

    def test_config_accepts_ogl_render_modes(self):
        parser = build_argparser()

        for mode in ("depth_ogl", "normal_ogl", "position_ogl", "confidence_ogl"):
            config = config_from_args(parser.parse_args(["example.glb", "--render-mode", mode]))
            self.assertEqual(config.render_mode, mode)

    def test_config_parses_orthographic_camera(self):
        parser = build_argparser()
        config = config_from_args(parser.parse_args(["example.glb", "--camera", "orthographic"]))

        self.assertEqual(config.camera, "orthographic")

    def test_config_parses_canonical_mv_conditions(self):
        parser = build_argparser()
        config = config_from_args(parser.parse_args(["example.glb", "--canonical-mv-conditions"]))

        self.assertTrue(config.canonical_mv_conditions)

    def test_config_parses_texture_map_max_size(self):
        parser = build_argparser()
        args = parser.parse_args(["example.glb", "--texture-map-max-size", "2048"])
        config = config_from_args(args)

        self.assertEqual(config.texture_map_max_size, 2048)


class MultiViewConfigTests(unittest.TestCase):
    def test_canonical_mv_conditions_trigger_multiview(self):
        config = SimpleNamespace(
            canonical_six_views=False,
            canonical_mv_conditions=True,
            azim_start=None,
            azim_end=None,
            azim_step=None,
            elev_start=None,
            elev_end=None,
            elev_step=None,
        )

        self.assertTrue(_is_multi_view(config))

    def test_canonical_mv_conditions_use_twelve_specs(self):
        config = SimpleNamespace(
            canonical_six_views=False,
            canonical_mv_conditions=True,
            elev=0.0,
            azim=0.0,
            elev_start=None,
            elev_end=None,
            elev_step=None,
            azim_start=None,
            azim_end=None,
            azim_step=None,
        )

        specs = _multi_view_specs(config)

        self.assertEqual(len(specs), 12)
        self.assertEqual(specs[0][1], "front")
        self.assertEqual(specs[6][1], "front2")
        self.assertEqual(specs[-1][1], "bottom2")

    def test_canonical_mv_conditions_use_normal_and_position_modes(self):
        config = SimpleNamespace(canonical_mv_conditions=True, render_mode="beauty")

        self.assertEqual(_multiview_render_modes(config), ("normal_ogl", "position_ogl"))

    def test_canonical_mv_conditions_use_chunk_size_eight_fallbacks(self):
        config = SimpleNamespace(canonical_mv_conditions=True, canonical_six_views=False, multi_view_chunk_size=24)

        self.assertEqual(_multiview_chunk_sizes(config), (8, 4, 2, 1))

    def test_multiview_output_path_includes_mode_suffix_when_needed(self):
        path = _multi_view_output_path(
            output_dir=pathlib.Path("outputs"),
            suffix=".png",
            index=0,
            elev=0.0,
            azim=0.0,
            label="front",
            mode="normal_ogl",
        )

        self.assertEqual(path, pathlib.Path("outputs/0000_front_normal_ogl.png"))


if __name__ == "__main__":
    unittest.main()
