import pathlib
import time
from dataclasses import replace

import torch

from .beauty import RenderModeRenderer
from .config import build_argparser as _build_argparser
from .config import config_from_args
from .lifecycle import RendererCache, is_cuda_failure, is_cuda_oom
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


def _render_all_from_config(config) -> tuple[list[pathlib.Path], pathlib.Path | None]:
    output_dir = _render_all_output_dir(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _render_all_suffix(config.output)
    render_config = replace(config, display=False, render_all=False)
    renderer_cache = RendererCache(device=torch.device(f"cuda:{torch.cuda.current_device()}"))
    try:
        renderer = renderer_cache.get(render_config)
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
    except Exception as exc:
        if is_cuda_failure(exc):
            renderer_cache.reset_after_cuda_failure(render_config)
        raise
    finally:
        renderer_cache.close()


def _multiview_chunk_sizes(config) -> tuple[int, ...]:
    if config.canonical_six_views:
        return (6, 2, 1)
    chunk_size = max(config.multi_view_chunk_size, 1)
    return tuple(range(chunk_size, 0, -1))


def _format_chunk_size_info(chunk_sizes_attempted: list[int], chunk_size_used: int) -> str:
    if not chunk_sizes_attempted or chunk_size_used == chunk_sizes_attempted[0]:
        return str(chunk_size_used)
    attempted = ",".join(str(size) for size in chunk_sizes_attempted[:-1])
    return f"{chunk_size_used} (fallback from {attempted})"


def _render_multiview_chunk(
    renderer: SceneRenderer,
    assets,
    base_config,
    output_dir: pathlib.Path,
    suffix: str,
    chunk_specs: list[tuple[int, str | None, float, float]],
) -> tuple[list[pathlib.Path], list[tuple[int, str | None, float, float, pathlib.Path]]]:
    prepared_rows = []
    rows = []
    for view_index, label, elev, azim in chunk_specs:
        view_config = replace(base_config, elev=elev, azim=azim)
        output_path = _multi_view_output_path(output_dir, suffix, view_index, elev, azim, label=label)
        prepared = renderer.prepare_view(assets, config=view_config)
        prepared_rows.append((prepared, output_path))
        rows.append((view_index, label, elev, azim, output_path))
    images = [renderer.render_prepared(prepared) for prepared, _output_path in prepared_rows]
    outputs = []
    for (_prepared, output_path), image in zip(prepared_rows, images):
        outputs.append(renderer.save_image(image, output_path))
    return outputs, rows


def _render_multiview_pass(
    renderer_cache: RendererCache,
    assets,
    base_config,
    output_dir: pathlib.Path,
    suffix: str,
    view_specs: list[tuple[int, str | None, float, float]],
    chunk_sizes: tuple[int, ...],
) -> tuple[list[pathlib.Path], list[tuple[int, str | None, float, float, pathlib.Path]], list[int], int]:
    rows = []
    outputs = []
    attempted_chunk_sizes = [chunk_sizes[0]]
    chunk_size_index = 0
    chunk_start = 0
    while chunk_start < len(view_specs):
        chunk_size = chunk_sizes[chunk_size_index]
        chunk_specs = view_specs[chunk_start: chunk_start + chunk_size]
        chunk_index = (chunk_start // chunk_size) + 1
        print(f"Rendering multi-view chunk {chunk_index} ({len(chunk_specs)} view(s), chunk size {chunk_size})")
        renderer = renderer_cache.get(base_config)
        try:
            chunk_outputs, chunk_rows = _render_multiview_chunk(renderer, assets, base_config, output_dir, suffix, chunk_specs)
        except Exception as exc:
            if is_cuda_oom(exc) and chunk_size_index + 1 < len(chunk_sizes):
                next_chunk_size = chunk_sizes[chunk_size_index + 1]
                renderer = None
                print(
                    f"Multi-view chunk starting at index {chunk_start:04d} hit CUDA OOM ({exc!r}); "
                    f"recreating renderer and retrying with chunk size {next_chunk_size}."
                )
                renderer_cache.reset_after_cuda_failure(base_config)
                chunk_size_index += 1
                attempted_chunk_sizes.append(next_chunk_size)
                continue
            if is_cuda_failure(exc):
                renderer = None
                renderer_cache.reset_after_cuda_failure(base_config)
            raise
        outputs.extend(chunk_outputs)
        rows.extend(chunk_rows)
        chunk_start += len(chunk_specs)
    return outputs, rows, attempted_chunk_sizes, chunk_sizes[chunk_size_index]


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
    chunk_sizes = _multiview_chunk_sizes(base_config)
    renderer_cache = RendererCache(device=torch.device(f"cuda:{torch.cuda.current_device()}"))
    try:
        renderer = renderer_cache.get(base_config)
        assets = renderer.prepare_assets(pathlib.Path(config.input))
        if config.canonical_six_views:
            print("Using canonical six-view set with sequential chunk rendering and CUDA OOM fallback through chunk sizes 6, 2, 1.")
        else:
            print(
                f"Using sequential multi-view chunk rendering with initial chunk size {chunk_sizes[0]} "
                "and renderer recreation on CUDA OOM."
            )
        outputs, rows, attempted_chunk_sizes, chunk_size_used = _render_multiview_pass(
            renderer_cache,
            assets,
            base_config,
            output_dir,
            suffix,
            view_specs,
            chunk_sizes=chunk_sizes,
        )
        chunk_size_info = _format_chunk_size_info(attempted_chunk_sizes, chunk_size_used)
        report = _format_multiview_report(config, rows, chunk_size_info)
        report_path = _write_multiview_report(output_dir, report)
        print(report, end="")
        print(f"Saved {len(outputs)} multi-view renders under {output_dir}")
        print(f"Saved multi-view report to {report_path}")
        return outputs, report_path
    except Exception as exc:
        if is_cuda_failure(exc):
            renderer_cache.reset_after_cuda_failure(base_config)
        raise
    finally:
        renderer_cache.close()


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
    renderer_cache = RendererCache(device=torch.device(f"cuda:{torch.cuda.current_device()}"))
    try:
        renderer_cache.get(config).render_to_file()
    except Exception as exc:
        if is_cuda_failure(exc):
            renderer_cache.reset_after_cuda_failure(config)
        raise
    finally:
        renderer_cache.close()
