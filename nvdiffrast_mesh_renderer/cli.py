import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import torch

from .beauty import RenderModeRenderer
from .config import build_argparser as _build_argparser
from .config import config_from_args
from .renderer import SceneRenderer

CANONICAL_SIX_VIEW_SPECS = (
    ("front", 0.0, 0.0),
    ("back", 0.0, 180.0),
    ("left", 0.0, 270.0),
    ("right", 0.0, 90.0),
    ("top", 90.0, 0.0),
    ("bottom", -90.0, 0.0),
)


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


def _is_multi_view(config) -> bool:
    return config.canonical_six_views or any(
        getattr(config, name) is not None
        for name in ("azim_start", "azim_end", "azim_step", "elev_start", "elev_end", "elev_step")
    )


def _axis_values(start: float | None, end: float | None, step: float | None, single: float) -> list[float]:
    if start is None:
        return [single]
    assert end is not None and step is not None
    signed_step = abs(step) if end >= start else -abs(step)
    values = []
    current = start
    eps = max(abs(signed_step) * 1e-6, 1e-8)
    if signed_step > 0.0:
        while current <= end + eps:
            values.append(float(round(current, 6)))
            current += signed_step
    else:
        while current >= end - eps:
            values.append(float(round(current, 6)))
            current += signed_step
    return values


def _multi_view_pairs(config) -> list[tuple[float, float]]:
    elevs = _axis_values(config.elev_start, config.elev_end, config.elev_step, config.elev)
    azims = _axis_values(config.azim_start, config.azim_end, config.azim_step, config.azim)
    return [(elev, azim) for elev in elevs for azim in azims]


def _multi_view_specs(config) -> list[tuple[int, str | None, float, float]]:
    if config.canonical_six_views:
        return [(index, label, elev, azim) for index, (label, elev, azim) in enumerate(CANONICAL_SIX_VIEW_SPECS)]
    return [(index, None, elev, azim) for index, (elev, azim) in enumerate(_multi_view_pairs(config))]


def _format_angle(value: float) -> str:
    return f"{value:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "_")


def _multi_view_output_path(output_dir: pathlib.Path, suffix: str, index: int, elev: float, azim: float, label: str | None = None) -> pathlib.Path:
    if label is not None:
        return output_dir / f"{index:04d}_{label}{suffix}"
    return output_dir / f"{index:04d}_elev_{_format_angle(elev)}_azim_{_format_angle(azim)}{suffix}"


def _format_multiview_report(config, rows: list[tuple[int, str | None, float, float, pathlib.Path]], chunk_size_info: str) -> str:
    lines = [
        "Multi-View Render Report",
        f"input: {config.input}",
        f"resolution: {config.resolution}",
        f"render_mode: {config.render_mode}",
        f"chunk_size: {chunk_size_info}",
        f"view_count: {len(rows)}",
        "",
        "index | name | elev | azim | output",
    ]
    for index, label, elev, azim, output_path in rows:
        lines.append(f"{index:04d} | {label or '-'} | {elev:.3f} | {azim:.3f} | {output_path}")
    return "\n".join(lines) + "\n"


def _write_multiview_report(output_dir: pathlib.Path, report: str) -> pathlib.Path:
    report_path = output_dir / "multiview_report.txt"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _is_cuda_oom(exc: BaseException) -> bool:
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        if "out of memory" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


@torch.inference_mode()
def _render_prepared_view(config, prepared, output_path: pathlib.Path, cuda_device_index: int | None = None) -> pathlib.Path:
    if cuda_device_index is not None:
        torch.cuda.set_device(cuda_device_index)
    renderer = SceneRenderer(config)
    image = renderer.render_prepared(prepared)
    return renderer.save_image(image, output_path)


def _render_all_from_config(config) -> tuple[list[pathlib.Path], pathlib.Path | None]:
    output_dir = _render_all_output_dir(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _render_all_suffix(config.output)
    renderer = SceneRenderer(replace(config, display=False, render_all=False))
    prepared = renderer.prepare_scene(pathlib.Path(config.input))
    outputs = []
    mode_timings = [] if config.benchmark_requested else None
    for mode in RenderModeRenderer.SUPPORTED_MODES:
        image = None
        mode_output = output_dir / f"{mode}{suffix}"
        if config.benchmark_requested:
            for _ in range(config.benchmark_warmup_runs):
                renderer.render(pathlib.Path(config.input), render_mode=mode, prepared=prepared)
            timings = []
            for _ in range(config.benchmark_runs):
                _cuda_sync()
                start = time.perf_counter()
                image = renderer.render(pathlib.Path(config.input), render_mode=mode, prepared=prepared)
                _cuda_sync()
                timings.append((time.perf_counter() - start) * 1000.0)
            assert mode_timings is not None
            mode_timings.append((mode, timings, mode_output))
        else:
            image = renderer.render(pathlib.Path(config.input), render_mode=mode, prepared=prepared)
        renderer.save_image(image, mode_output)
        outputs.append(mode_output)
    report_path = None
    if config.benchmark_requested:
        assert mode_timings is not None
        report = _format_benchmark_report(config, mode_timings)
        report_path = _write_benchmark_report(output_dir, report)
        print(report, end="")
    print(f"Saved {len(outputs)} render modes under {output_dir}")
    if report_path is not None:
        print(f"Saved benchmark report to {report_path}")
    return outputs, report_path


def _render_multiview_chunk(jobs: list[tuple[int, float, float, object, object, pathlib.Path, int | None]]) -> list[pathlib.Path]:
    if len(jobs) == 1:
        _index, _elev, _azim, view_config, prepared, output_path, cuda_device_index = jobs[0]
        return [_render_prepared_view(view_config, prepared, output_path, cuda_device_index=cuda_device_index)]
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [
            executor.submit(_render_prepared_view, view_config, prepared, output_path, cuda_device_index)
            for _index, _elev, _azim, view_config, prepared, output_path, cuda_device_index in jobs
        ]
        return [future.result() for future in futures]


def _render_multiview_pass(
    base_renderer,
    assets,
    base_config,
    output_dir: pathlib.Path,
    suffix: str,
    view_specs: list[tuple[int, str | None, float, float]],
    cuda_device_index: int,
    chunk_size: int,
    stop_on_cuda_oom: bool,
) -> tuple[list[pathlib.Path], list[tuple[int, str | None, float, float, pathlib.Path]]]:
    rows = []
    outputs = []
    chunk_count = (len(view_specs) + chunk_size - 1) // chunk_size
    for chunk_index, chunk_start in enumerate(range(0, len(view_specs), chunk_size), start=1):
        chunk_specs = view_specs[chunk_start: chunk_start + chunk_size]
        print(f"Rendering multi-view chunk {chunk_index}/{chunk_count} ({len(chunk_specs)} view(s))")
        jobs = []
        for view_index, label, elev, azim in chunk_specs:
            view_config = replace(base_config, elev=elev, azim=azim)
            output_path = _multi_view_output_path(output_dir, suffix, view_index, elev, azim, label=label)
            prepared = base_renderer.prepare_view(assets, config=view_config)
            jobs.append((view_index, elev, azim, view_config, prepared, output_path, cuda_device_index))
            rows.append((view_index, label, elev, azim, output_path))
        try:
            outputs.extend(_render_multiview_chunk(jobs))
        except Exception as exc:
            if stop_on_cuda_oom and _is_cuda_oom(exc):
                raise
            print(f"Parallel multi-view chunk failed ({exc!r}); retrying sequentially.")
            for _view_index, _elev, _azim, view_config, prepared, output_path, cuda_device_index in jobs:
                outputs.append(_render_prepared_view(view_config, prepared, output_path, cuda_device_index=cuda_device_index))
    return outputs, rows


def _render_multiview_from_config(config) -> tuple[list[pathlib.Path], pathlib.Path]:
    if config.render_all:
        raise ValueError("--render-all is not supported together with multi-view rendering")
    if config.display:
        print("Ignoring --display during multi-view rendering.")
    view_specs = _multi_view_specs(config)
    output_dir = _render_all_output_dir(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _render_all_suffix(config.output)
    base_config = replace(config, display=False)
    cuda_device_index = torch.cuda.current_device()
    base_renderer = SceneRenderer(base_config)
    assets = base_renderer.prepare_assets(pathlib.Path(config.input))
    if config.canonical_six_views:
        print("Using canonical six-view set with chunk size 6 and CUDA OOM fallback to chunk size 2.")
        try:
            outputs, rows = _render_multiview_pass(
                base_renderer,
                assets,
                base_config,
                output_dir,
                suffix,
                view_specs,
                cuda_device_index,
                chunk_size=6,
                stop_on_cuda_oom=True,
            )
            chunk_size_info = "6"
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            print(f"Canonical six-view render hit CUDA OOM ({exc!r}); retrying with chunk size 2.")
            torch.cuda.empty_cache()
            outputs, rows = _render_multiview_pass(
                base_renderer,
                assets,
                base_config,
                output_dir,
                suffix,
                view_specs,
                cuda_device_index,
                chunk_size=2,
                stop_on_cuda_oom=False,
            )
            chunk_size_info = "2 (fallback from 6)"
    else:
        chunk_size = max(base_config.multi_view_chunk_size, 1)
        outputs, rows = _render_multiview_pass(
            base_renderer,
            assets,
            base_config,
            output_dir,
            suffix,
            view_specs,
            cuda_device_index,
            chunk_size=chunk_size,
            stop_on_cuda_oom=False,
        )
        chunk_size_info = str(chunk_size)
    report = _format_multiview_report(config, rows, chunk_size_info)
    report_path = _write_multiview_report(output_dir, report)
    print(report, end="")
    print(f"Saved {len(outputs)} multi-view renders under {output_dir}")
    print(f"Saved multi-view report to {report_path}")
    return outputs, report_path


@torch.inference_mode()
def render_all_modes() -> tuple[list[pathlib.Path], pathlib.Path | None]:
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this renderer")
    config = config_from_args(args)
    return _render_all_from_config(config)


@torch.inference_mode()
def render_multi_view() -> tuple[list[pathlib.Path], pathlib.Path]:
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this renderer")
    config = config_from_args(args)
    if not _is_multi_view(config):
        raise ValueError("Specify --canonical-six-views or provide at least one full multi-view range triplet")
    return _render_multiview_from_config(config)


@torch.inference_mode()
def main() -> None:
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this renderer")
    config = config_from_args(args)
    if _is_multi_view(config):
        _render_multiview_from_config(config)
        return
    if config.render_all:
        _render_all_from_config(config)
        return
    SceneRenderer(config).render_to_file()
