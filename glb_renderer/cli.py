import pathlib
import time
from dataclasses import replace

import torch

from .beauty import RenderModeRenderer
from .config import build_argparser as _build_argparser
from .config import config_from_args
from .renderer import SceneRenderer


def build_argparser():
    return _build_argparser()


def _render_all_output_dir(output: str) -> pathlib.Path:
    path = pathlib.Path(output)
    return path if not path.suffix else path.with_suffix("")


def _render_all_suffix(output: str) -> str:
    suffix = pathlib.Path(output).suffix.lower()
    return suffix or ".png"


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_benchmark_report(config, mode_timings: list[tuple[str, list[float], pathlib.Path]]) -> str:
    all_samples = [sample for _mode, timings, _path in mode_timings for sample in timings]
    overall_avg = sum(all_samples) / len(all_samples)
    lines = [
        "Render-All Benchmark Report",
        f"input: {config.input}",
        f"resolution: {config.resolution}",
        f"warmup_runs_per_mode: {config.benchmark_warmup_runs}",
        f"timed_runs_per_mode: {config.benchmark_runs}",
        f"mode_count: {len(mode_timings)}",
        f"overall_avg_ms: {overall_avg:.3f}",
        "",
        "mode | avg_ms | min_ms | max_ms | output",
    ]
    for mode, timings, output_path in mode_timings:
        avg_ms = sum(timings) / len(timings)
        lines.append(f"{mode} | {avg_ms:.3f} | {min(timings):.3f} | {max(timings):.3f} | {output_path}")
    return "\n".join(lines) + "\n"


def _write_benchmark_report(output_dir: pathlib.Path, report: str) -> pathlib.Path:
    report_path = output_dir / "render_all_report.txt"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _render_all_from_config(config) -> tuple[list[pathlib.Path], pathlib.Path]:
    output_dir = _render_all_output_dir(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _render_all_suffix(config.output)
    renderer = SceneRenderer(replace(config, display=False, render_all=False))
    prepared = renderer.prepare_scene(pathlib.Path(config.input))
    outputs = []
    mode_timings = []
    for mode in RenderModeRenderer.SUPPORTED_MODES:
        for _ in range(config.benchmark_warmup_runs):
            renderer.render(pathlib.Path(config.input), render_mode=mode, prepared=prepared)
        image = None
        timings = []
        for _ in range(config.benchmark_runs):
            _cuda_sync()
            start = time.perf_counter()
            image = renderer.render(pathlib.Path(config.input), render_mode=mode, prepared=prepared)
            _cuda_sync()
            timings.append((time.perf_counter() - start) * 1000.0)
        mode_output = output_dir / f"{mode}{suffix}"
        renderer.save_image(image, mode_output)
        outputs.append(mode_output)
        mode_timings.append((mode, timings, mode_output))
    report = _format_benchmark_report(config, mode_timings)
    report_path = _write_benchmark_report(output_dir, report)
    print(report, end="")
    print(f"Saved {len(outputs)} render modes under {output_dir}")
    print(f"Saved benchmark report to {report_path}")
    return outputs, report_path


@torch.inference_mode()
def render_all_modes() -> tuple[list[pathlib.Path], pathlib.Path]:
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this renderer")
    config = config_from_args(args)
    return _render_all_from_config(config)


@torch.inference_mode()
def main() -> None:
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this renderer")
    config = config_from_args(args)
    if config.render_all:
        _render_all_from_config(config)
        return
    SceneRenderer(config).render_to_file()
