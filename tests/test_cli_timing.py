import pathlib
import unittest

from nvdiffrast_mesh_renderer.cli import (
    TimingBreakdown,
    _format_multiview_report,
    _format_render_all_report,
    _timing_summary_console_message,
    _timing_summary_report_lines,
)


class CliTimingReportTests(unittest.TestCase):
    def test_timing_summary_report_lines_include_expected_fields(self):
        timing = TimingBreakdown(
            total_ms=3210.0,
            session_init_ms=100.0,
            data_loading_ms=200.0,
            scene_prepare_ms=300.0,
            render_ms=2400.0,
            save_ms=210.0,
        )

        lines = _timing_summary_report_lines(timing)

        self.assertEqual(
            lines,
            [
                "total_elapsed: 3.2s",
                "session_init: 100ms",
                "data_loading: 200ms",
                "scene_prepare: 300ms",
                "render: 2.4s",
                "save: 210ms",
            ],
        )

    def test_timing_summary_console_message_omits_zero_scene_prepare(self):
        timing = TimingBreakdown(
            total_ms=1500.0,
            session_init_ms=50.0,
            data_loading_ms=100.0,
            render_ms=1200.0,
            save_ms=150.0,
        )

        message = _timing_summary_console_message("Render-All Timing", timing)

        self.assertIn("[Info] Render-All Timing:", message)
        self.assertIn("total_elapsed=1.5s", message)
        self.assertNotIn("scene_prepare", message)

    def test_render_all_report_includes_timing_summary(self):
        timing = TimingBreakdown(
            total_ms=2500.0,
            session_init_ms=100.0,
            data_loading_ms=200.0,
            scene_prepare_ms=300.0,
            render_ms=1700.0,
            save_ms=200.0,
        )

        report = _format_render_all_report(
            type("Config", (), {
                "input": "mesh.glb",
                "resolution": 512,
                "benchmark_requested": False,
                "render_all_batch_size": 4,
                "benchmark_warmup_runs": 0,
                "benchmark_runs": 0,
            })(),
            [("beauty", pathlib.Path("outputs/beauty.png"), None)],
            timing,
        )

        self.assertIn("Timing Summary", report)
        self.assertIn("total_elapsed: 2.5s", report)
        self.assertIn("scene_prepare: 300ms", report)

    def test_multiview_report_includes_timing_summary(self):
        timing = TimingBreakdown(
            total_ms=4000.0,
            session_init_ms=100.0,
            data_loading_ms=300.0,
            scene_prepare_ms=600.0,
            render_ms=2600.0,
            save_ms=400.0,
        )

        report = _format_multiview_report(
            type("Config", (), {
                "input": "mesh.glb",
                "resolution": 512,
                "render_mode": "beauty",
                "canonical_mv_conditions": False,
            })(),
            [(0, "front", 0.0, 0.0, "beauty", pathlib.Path("outputs/0000_front.png"))],
            "4",
            timing,
        )

        self.assertIn("Timing Summary", report)
        self.assertIn("data_loading: 300ms", report)
        self.assertIn("save: 400ms", report)
        self.assertIn("render_modes: beauty", report)
        self.assertIn("output_count: 1", report)


if __name__ == "__main__":
    unittest.main()
