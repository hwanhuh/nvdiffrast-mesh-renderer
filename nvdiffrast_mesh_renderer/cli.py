import pathlib
import time
from dataclasses import replace

import torch

from .beauty import RenderModeRenderer
from .config import build_argparser as _build_argparser
from .config import config_from_args
from .lifecycle import RendererCache, is_cuda_failure, is_cuda_oom
from .logging_utils import RunLogger, estimate_remaining_ms, format_duration_ms, format_path_notice
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


def _single_render_log_path(output: str) -> pathlib.Path:
    path = pathlib.Path(output)
    return path.with_suffix(".log") if path.suffix else path / "render.log"


def _render_all_log_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir / "render_all.log"


def _multi_view_log_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir / "multiview.log"


def _progress_line(label: str, completed: int, total: int, *, item_label: str, last_ms: float | None, eta_ms: float | None) -> str:
    return (
        f"[Progress] [{completed}/{total}] {label}: {item_label} "
        f"(last: {format_duration_ms(last_ms)} / ETA: {format_duration_ms(eta_ms)})"
    )


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_render_all_report(config, mode_rows: list[tuple[str, pathlib.Path, list[float] | None]]) -> str:
    lines = [
        "Render-All Report",
        f"input: {config.input}",
        f"resolution: {config.resolution}",
        f"benchmark_requested: {str(bool(config.benchmark_requested)).lower()}",
        f"warmup_runs_per_mode: {config.benchmark_warmup_runs if config.benchmark_requested else 0}",
        f"timed_runs_per_mode: {config.benchmark_runs if config.benchmark_requested else 0}",
        f"mode_count: {len(mode_rows)}",
        "",
    ]
    if config.benchmark_requested:
        all_samples = [sample for _mode, _path, timings in mode_rows for sample in (timings or [])]
        overall_avg = sum(all_samples) / len(all_samples)
        lines.extend(
            [
                f"overall_avg_ms: {overall_avg:.3f}",
                "",
                "mode | output | avg_ms | min_ms | max_ms",
            ]
        )
        for mode, output_path, timings in mode_rows:
            assert timings is not None
            avg_ms = sum(timings) / len(timings)
            lines.append(f"{mode} | {output_path} | {avg_ms:.3f} | {min(timings):.3f} | {max(timings):.3f}")
    else:
        lines.append("mode | output")
        for mode, output_path, _timings in mode_rows:
            lines.append(f"{mode} | {output_path}")
    return "\n".join(lines) + "\n"


def _write_render_all_report(output_dir: pathlib.Path, report: str) -> pathlib.Path:
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


# Render-all is a sequential multi-render entrypoint that reuses one renderer/context.
def _render_all_from_config(config) -> tuple[list[pathlib.Path], pathlib.Path]:
    output_dir = _render_all_output_dir(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(_render_all_log_path(output_dir), echo=config.print_progress)
    logger.reset()
    suffix = _render_all_suffix(config.output)
    render_config = replace(config, display=False, render_all=False)
    renderer_cache = RendererCache(device=torch.device(f"cuda:{torch.cuda.current_device()}"), logger=logger)
    total_modes = len(RenderModeRenderer.SUPPORTED_MODES)
    progress_start = time.perf_counter()
    logger.log(
        f"Starting render-all run for {total_modes} mode(s) -> {output_dir}",
        console="always",
        console_message="[Info] ===Render-All Start: Initializing===",
    )
    logger.log(
        f"Render-all total modes: {total_modes}",
        console="always",
        console_message=f"[Info] Render-All Items: total={total_modes}",
    )
    try:
        renderer = renderer_cache.get(render_config)
        prepared = renderer.prepare_scene(pathlib.Path(config.input))
        outputs = []
        mode_rows: list[tuple[str, pathlib.Path, list[float] | None]] = []
        for mode in RenderModeRenderer.SUPPORTED_MODES:
            mode_started = time.perf_counter()
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
                mode_rows.append((mode, mode_output, timings))
            else:
                image = renderer.render(pathlib.Path(config.input), render_mode=mode, prepared=prepared)
                mode_rows.append((mode, mode_output, None))
            renderer.save_image(image, mode_output)
            outputs.append(mode_output)
            elapsed_ms = (time.perf_counter() - progress_start) * 1000.0
            mode_duration_ms = (time.perf_counter() - mode_started) * 1000.0
            eta_ms = estimate_remaining_ms(len(outputs), total_modes, elapsed_ms)
            logger.log(
                f"Completed render-all mode {mode} ({len(outputs)}/{total_modes})",
                console="always",
                console_message=_progress_line(
                    "render-all",
                    len(outputs),
                    total_modes,
                    item_label=mode,
                    last_ms=mode_duration_ms,
                    eta_ms=eta_ms,
                ),
            )
        report = _format_render_all_report(config, mode_rows)
        report_path = _write_render_all_report(output_dir, report)
        logger.log(report.rstrip())
        logger.log(f"Saved {len(outputs)} render modes under {output_dir}")
        logger.log(f"Saved render-all report to {report_path}")
        logger.log(
            f"Finished render-all run -> {report_path}",
            console="always",
            console_message=format_path_notice("Info", "Done. Log file:", logger.path),
        )
        return outputs, report_path
    except Exception as exc:
        logger.log(f"Render-all run failed: {type(exc).__name__}: {exc}")
        logger.log(
            f"Render-all run failed: {type(exc).__name__}",
            console="always",
            console_message=format_path_notice("Error", f"Render-All failed: {type(exc).__name__}. Log file:", logger.path),
        )
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
    logger: RunLogger,
) -> tuple[list[pathlib.Path], list[tuple[int, str | None, float, float, pathlib.Path]], list[int], int]:
    rows = []
    outputs = []
    attempted_chunk_sizes = [chunk_sizes[0]]
    chunk_size_index = 0
    chunk_start = 0
    progress_start = time.perf_counter()
    while chunk_start < len(view_specs):
        chunk_size = chunk_sizes[chunk_size_index]
        chunk_specs = view_specs[chunk_start: chunk_start + chunk_size]
        chunk_index = (chunk_start // chunk_size) + 1
        chunk_started = time.perf_counter()
        logger.log(f"Rendering multi-view chunk {chunk_index} ({len(chunk_specs)} view(s), chunk size {chunk_size})")
        renderer = renderer_cache.get(base_config)
        try:
            chunk_outputs, chunk_rows = _render_multiview_chunk(renderer, assets, base_config, output_dir, suffix, chunk_specs)
        except Exception as exc:
            if is_cuda_oom(exc) and chunk_size_index + 1 < len(chunk_sizes):
                next_chunk_size = chunk_sizes[chunk_size_index + 1]
                renderer = None
                logger.log(
                    f"Multi-view chunk starting at index {chunk_start:04d} hit CUDA OOM ({exc!r}); "
                    f"recreating renderer and retrying with chunk size {next_chunk_size}."
                )
                renderer_cache.reset_after_cuda_failure(base_config)
                chunk_size_index += 1
                attempted_chunk_sizes.append(next_chunk_size)
                logger.log(
                    f"Multi-view fallback to chunk size {next_chunk_size} after CUDA OOM at chunk start {chunk_start:04d}",
                    console="always",
                    console_message=f"[Info] Multi-View retry: chunk -> {next_chunk_size}",
                )
                continue
            if is_cuda_failure(exc):
                renderer = None
                renderer_cache.reset_after_cuda_failure(base_config)
            raise
        outputs.extend(chunk_outputs)
        rows.extend(chunk_rows)
        elapsed_ms = (time.perf_counter() - progress_start) * 1000.0
        chunk_duration_ms = (time.perf_counter() - chunk_started) * 1000.0
        eta_ms = estimate_remaining_ms(len(rows), len(view_specs), elapsed_ms)
        logger.log(
            f"Completed multi-view chunk {chunk_index}; saved {len(rows)}/{len(view_specs)} view(s)",
            console="always",
            console_message=_progress_line(
                "multi-view",
                len(rows),
                len(view_specs),
                item_label=f"chunk {chunk_index}",
                last_ms=chunk_duration_ms,
                eta_ms=eta_ms,
            ),
        )
        chunk_start += len(chunk_specs)
    return outputs, rows, attempted_chunk_sizes, chunk_sizes[chunk_size_index]


# Standalone multi-view is also sequential: one renderer/context per invocation, no raster thread pool.
def _render_multiview_from_config(config) -> tuple[list[pathlib.Path], pathlib.Path]:
    if config.render_all:
        raise ValueError("--render-all is not supported together with multi-view rendering")
    view_specs = _multi_view_specs(config)
    output_dir = _render_all_output_dir(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(_multi_view_log_path(output_dir), echo=config.print_progress)
    logger.reset()
    total_views = len(view_specs)
    logger.log(
        f"Starting multi-view run for {total_views} view(s) -> {output_dir}",
        console="always",
        console_message="[Info] ===Multi-View Start: Initializing===",
    )
    logger.log(
        f"Multi-view total views: {total_views}",
        console="always",
        console_message=f"[Info] Multi-View Items: total={total_views}",
    )
    if config.display:
        logger.log("Ignoring --display during multi-view rendering.")
    suffix = _render_all_suffix(config.output)
    base_config = replace(config, display=False)
    chunk_sizes = _multiview_chunk_sizes(base_config)
    renderer_cache = RendererCache(device=torch.device(f"cuda:{torch.cuda.current_device()}"), logger=logger)
    try:
        renderer = renderer_cache.get(base_config)
        assets = renderer.prepare_assets(pathlib.Path(config.input))
        if config.canonical_six_views:
            logger.log("Using canonical six-view set with sequential chunk rendering and CUDA OOM fallback through chunk sizes 6, 2, 1.")
        else:
            logger.log(
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
            logger=logger,
        )
        chunk_size_info = _format_chunk_size_info(attempted_chunk_sizes, chunk_size_used)
        report = _format_multiview_report(config, rows, chunk_size_info)
        report_path = _write_multiview_report(output_dir, report)
        logger.log(report.rstrip())
        logger.log(f"Saved {len(outputs)} multi-view renders under {output_dir}")
        logger.log(f"Saved multi-view report to {report_path}")
        logger.log(
            f"Finished multi-view run -> {report_path}",
            console="always",
            console_message=format_path_notice("Info", "Done. Log file:", logger.path),
        )
        return outputs, report_path
    except Exception as exc:
        logger.log(f"Multi-view run failed: {type(exc).__name__}: {exc}")
        logger.log(
            f"Multi-view run failed: {type(exc).__name__}",
            console="always",
            console_message=format_path_notice("Error", f"Multi-View failed: {type(exc).__name__}. Log file:", logger.path),
        )
        if is_cuda_failure(exc):
            renderer_cache.reset_after_cuda_failure(base_config)
        raise
    finally:
        renderer_cache.close()


@torch.inference_mode()
def render_all_modes() -> tuple[list[pathlib.Path], pathlib.Path]:
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
    # Basic CLI mode is a single-render entrypoint with one renderer/context per invocation.
    logger = RunLogger(_single_render_log_path(config.output), echo=config.print_progress)
    logger.reset()
    renderer_cache = RendererCache(device=torch.device(f"cuda:{torch.cuda.current_device()}"), logger=logger)
    render_started = time.perf_counter()
    logger.log("Starting single render run.", console="always", console_message="[Info] ===Render Start: Initializing===")
    try:
        renderer_cache.get(config).render_to_file()
        duration_ms = (time.perf_counter() - render_started) * 1000.0
        logger.log(
            "Finished single render run.",
            console="always",
            console_message=_progress_line("render", 1, 1, item_label="completed", last_ms=duration_ms, eta_ms=0.0),
        )
        logger.log(
            "Single render completed successfully.",
            console="always",
            console_message=format_path_notice("Info", "Done. Log file:", logger.path),
        )
    except Exception as exc:
        logger.log(f"Single render failed: {type(exc).__name__}: {exc}")
        logger.log(
            f"Single render failed: {type(exc).__name__}",
            console="always",
            console_message=format_path_notice("Error", f"Render failed: {type(exc).__name__}. Log file:", logger.path),
        )
        if is_cuda_failure(exc):
            renderer_cache.reset_after_cuda_failure(config)
        raise
    finally:
        renderer_cache.close()
