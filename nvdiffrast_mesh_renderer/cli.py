import pathlib
import time
from dataclasses import dataclass, replace

import torch

from .beauty import RenderModeRenderer
from .config import build_argparser as _build_argparser
from .config import config_from_args
from .image_io import AsyncImageSaver
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
CANONICAL_OFFSET45_VIEW_SPECS = (
    ("front2", 0.0, 135.0),
    ("right2", 0.0, 45.0),
    ("back2", 0.0, -45.0),
    ("left2", 0.0, -135.0),
    ("top2", 65.0, 135.0),
    ("bottom2", -65.0, 135.0),
)
CANONICAL_MV_CONDITION_MODES = ("normal_ogl", "position_ogl")


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


@dataclass(frozen=True)
class TimingBreakdown:
    total_ms: float
    session_init_ms: float = 0.0
    data_loading_ms: float = 0.0
    scene_prepare_ms: float = 0.0
    render_ms: float = 0.0
    save_ms: float = 0.0


def _timing_summary_fields(timing: TimingBreakdown) -> list[tuple[str, float]]:
    fields = [
        ("total_elapsed", timing.total_ms),
        ("session_init", timing.session_init_ms),
        ("data_loading", timing.data_loading_ms),
    ]
    if timing.scene_prepare_ms > 0.0:
        fields.append(("scene_prepare", timing.scene_prepare_ms))
    fields.extend(
        [
            ("render", timing.render_ms),
            ("save", timing.save_ms),
        ]
    )
    return fields


def _timing_summary_report_lines(timing: TimingBreakdown) -> list[str]:
    return [f"{label}: {format_duration_ms(value)}" for label, value in _timing_summary_fields(timing)]


def _timing_summary_console_message(label: str, timing: TimingBreakdown) -> str:
    parts = ", ".join(f"{name}={format_duration_ms(value)}" for name, value in _timing_summary_fields(timing))
    return f"[Info] {label}: {parts}"


def _format_render_all_report(
    config,
    mode_rows: list[tuple[str, pathlib.Path, list[float] | None]],
    timing: TimingBreakdown,
) -> str:
    lines = [
        "Render-All Report",
        f"input: {config.input}",
        f"resolution: {config.resolution}",
        f"benchmark_requested: {str(bool(config.benchmark_requested)).lower()}",
        f"mode_batching_enabled: {str((not config.benchmark_requested) and config.render_all_batch_size > 1).lower()}",
        f"mode_batch_size: {1 if config.benchmark_requested else config.render_all_batch_size}",
        f"warmup_runs_per_mode: {config.benchmark_warmup_runs if config.benchmark_requested else 0}",
        f"timed_runs_per_mode: {config.benchmark_runs if config.benchmark_requested else 0}",
        f"mode_count: {len(mode_rows)}",
        "",
        "Timing Summary",
        *_timing_summary_report_lines(timing),
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


def _render_all_mode_batch_sizes(config, total_modes: int) -> tuple[int, ...]:
    if config.benchmark_requested:
        return (1,)
    batch_size = min(max(config.render_all_batch_size, 1), total_modes)
    return tuple(range(batch_size, 0, -1))


def _is_multi_view(config) -> bool:
    return config.canonical_six_views or getattr(config, "canonical_mv_conditions", False) or any(
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
    if getattr(config, "canonical_mv_conditions", False):
        combined_specs = CANONICAL_SIX_VIEW_SPECS + CANONICAL_OFFSET45_VIEW_SPECS
        return [(index, label, elev, azim) for index, (label, elev, azim) in enumerate(combined_specs)]
    if config.canonical_six_views:
        return [(index, label, elev, azim) for index, (label, elev, azim) in enumerate(CANONICAL_SIX_VIEW_SPECS)]
    return [(index, None, elev, azim) for index, (elev, azim) in enumerate(_multi_view_pairs(config))]


def _format_angle(value: float) -> str:
    return f"{value:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "_")


def _multi_view_output_path(
    output_dir: pathlib.Path,
    suffix: str,
    index: int,
    elev: float,
    azim: float,
    label: str | None = None,
    mode: str | None = None,
) -> pathlib.Path:
    stem = f"{index:04d}_{label}" if label is not None else f"{index:04d}_elev_{_format_angle(elev)}_azim_{_format_angle(azim)}"
    if mode is not None:
        stem = f"{stem}_{mode}"
    return output_dir / f"{stem}{suffix}"


def _multiview_render_modes(config) -> tuple[str, ...]:
    if getattr(config, "canonical_mv_conditions", False):
        return CANONICAL_MV_CONDITION_MODES
    return (config.render_mode,)


def _format_multiview_report(
    config,
    rows: list[tuple[int, str | None, float, float, str, pathlib.Path]],
    chunk_size_info: str,
    timing: TimingBreakdown,
) -> str:
    render_modes = sorted({mode for _index, _label, _elev, _azim, mode, _output_path in rows}) if rows else _multiview_render_modes(config)
    view_count = len({index for index, _label, _elev, _azim, _mode, _output_path in rows})
    lines = [
        "Multi-View Render Report",
        f"input: {config.input}",
        f"resolution: {config.resolution}",
        f"render_modes: {', '.join(render_modes)}",
        f"chunk_size: {chunk_size_info}",
        f"view_count: {view_count}",
        f"output_count: {len(rows)}",
        "",
        "Timing Summary",
        *_timing_summary_report_lines(timing),
        "",
        "index | name | elev | azim | mode | output",
    ]
    for index, label, elev, azim, mode, output_path in rows:
        lines.append(f"{index:04d} | {label or '-'} | {elev:.3f} | {azim:.3f} | {mode} | {output_path}")
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
    input_path = pathlib.Path(config.input)
    renderer_cache = RendererCache(device=torch.device(f"cuda:{torch.cuda.current_device()}"), logger=logger)
    total_modes = len(RenderModeRenderer.SUPPORTED_MODES)
    mode_batch_sizes = _render_all_mode_batch_sizes(config, total_modes)
    progress_start = time.perf_counter()
    session_init_ms = 0.0
    data_loading_ms = 0.0
    scene_prepare_ms = 0.0
    render_ms_total = 0.0
    save_ms_total = 0.0
    saver: AsyncImageSaver | None = None
    logger.log(
        f"Starting render-all run for {total_modes} mode(s) -> {output_dir}",
        console="always",
        console_message="[Info] Render-All Start: Initializing",
    )
    logger.log(
        f"Render-all total modes: {total_modes}",
        console="always",
        console_message=f"[Info] Render-All Items: total={total_modes}",
    )
    try:
        init_start = time.perf_counter()
        renderer, _created = renderer_cache.get_with_status(render_config)
        if _created:
            session_init_ms += (time.perf_counter() - init_start) * 1000.0
        assets_start = time.perf_counter()
        assets = renderer.prepare_assets(input_path)
        data_loading_ms += (time.perf_counter() - assets_start) * 1000.0
        prepare_start = time.perf_counter()
        prepared = renderer.prepare_view(assets)
        scene_prepare_ms += (time.perf_counter() - prepare_start) * 1000.0
        outputs = []
        saved_output_count = 0
        mode_rows: list[tuple[str, pathlib.Path, list[float] | None]] = []

        def _record_completed_render_all_saves(completed: list[tuple[pathlib.Path, float]]) -> None:
            nonlocal save_ms_total, saved_output_count
            for saved_path, duration_ms in completed:
                save_ms_total += duration_ms
                outputs.append(saved_path)
                saved_output_count += 1
                elapsed_ms = (time.perf_counter() - progress_start) * 1000.0
                eta_ms = estimate_remaining_ms(saved_output_count, total_modes, elapsed_ms)
                logger.log(
                    f"Completed render-all mode {saved_path.stem} ({saved_output_count}/{total_modes})",
                    console="always",
                    console_message=_progress_line(
                        "render-all",
                        saved_output_count,
                        total_modes,
                        item_label=saved_path.stem,
                        last_ms=duration_ms,
                        eta_ms=eta_ms,
                    ),
                )

        if config.benchmark_requested:
            for mode in RenderModeRenderer.SUPPORTED_MODES:
                mode_started = time.perf_counter()
                image = None
                mode_output = output_dir / f"{mode}{suffix}"
                for _ in range(config.benchmark_warmup_runs):
                    render_start = time.perf_counter()
                    renderer.render(input_path, render_mode=mode, prepared=prepared)
                    render_ms_total += (time.perf_counter() - render_start) * 1000.0
                timings = []
                for _ in range(config.benchmark_runs):
                    _cuda_sync()
                    start = time.perf_counter()
                    image = renderer.render(input_path, render_mode=mode, prepared=prepared)
                    _cuda_sync()
                    render_duration_ms = (time.perf_counter() - start) * 1000.0
                    render_ms_total += render_duration_ms
                    timings.append(render_duration_ms)
                mode_rows.append((mode, mode_output, timings))
                save_start = time.perf_counter()
                renderer.save_image(image, mode_output)
                save_ms_total += (time.perf_counter() - save_start) * 1000.0
                outputs.append(mode_output)
                saved_output_count += 1
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
        else:
            saver = AsyncImageSaver(
                max_pending=max(mode_batch_sizes[0], 2),
                png_compression=config.png_compression,
            )
            mode_index = 0
            batch_size_index = 0
            attempted_batch_sizes = [mode_batch_sizes[0]]
            all_modes = list(RenderModeRenderer.SUPPORTED_MODES)
            if mode_batch_sizes[0] > 1:
                logger.log(
                    f"Render-all mode batching enabled with initial batch size {mode_batch_sizes[0]}.",
                )
            while mode_index < total_modes:
                batch_size = mode_batch_sizes[batch_size_index]
                batch_modes = all_modes[mode_index: mode_index + batch_size]
                batch_started = time.perf_counter()
                try:
                    render_start = time.perf_counter()
                    batch_images = renderer.render_prepared_modes(prepared, batch_modes)
                    render_ms_total += (time.perf_counter() - render_start) * 1000.0
                except Exception as exc:
                    if is_cuda_oom(exc) and batch_size_index + 1 < len(mode_batch_sizes):
                        next_batch_size = mode_batch_sizes[batch_size_index + 1]
                        logger.log(
                            f"Render-all batch starting at mode index {mode_index + 1} hit CUDA OOM ({exc!r}); "
                            f"recreating renderer and retrying with batch size {next_batch_size}."
                        )
                        renderer_cache.reset_after_cuda_failure(render_config)
                        batch_size_index += 1
                        attempted_batch_sizes.append(next_batch_size)
                        init_start = time.perf_counter()
                        renderer, _created = renderer_cache.get_with_status(render_config)
                        if _created:
                            session_init_ms += (time.perf_counter() - init_start) * 1000.0
                        assets_start = time.perf_counter()
                        assets = renderer.prepare_assets(input_path)
                        data_loading_ms += (time.perf_counter() - assets_start) * 1000.0
                        prepare_start = time.perf_counter()
                        prepared = renderer.prepare_view(assets)
                        scene_prepare_ms += (time.perf_counter() - prepare_start) * 1000.0
                        continue
                    raise
                for mode in batch_modes:
                    mode_output = output_dir / f"{mode}{suffix}"
                    mode_rows.append((mode, mode_output, None))
                    _record_completed_render_all_saves(saver.submit(batch_images[mode], mode_output))
                mode_index += len(batch_modes)
            _record_completed_render_all_saves(saver.finish())
            if attempted_batch_sizes[-1] != attempted_batch_sizes[0]:
                attempted = ",".join(str(size) for size in attempted_batch_sizes[:-1])
                logger.log(
                    f"Render-all completed with reduced mode batch size {attempted_batch_sizes[-1]} "
                    f"(fallback from {attempted})."
                )
        timing = TimingBreakdown(
            total_ms=(time.perf_counter() - progress_start) * 1000.0,
            session_init_ms=session_init_ms,
            data_loading_ms=data_loading_ms,
            scene_prepare_ms=scene_prepare_ms,
            render_ms=render_ms_total,
            save_ms=save_ms_total,
        )
        report = _format_render_all_report(config, mode_rows, timing)
        report_path = _write_render_all_report(output_dir, report)
        logger.log(report.rstrip())
        logger.log(f"Saved {len(outputs)} render modes under {output_dir}")
        logger.log(f"Saved render-all report to {report_path}")
        logger.log(
            f"Render-all timing summary: total_ms={timing.total_ms:.3f}",
            console="always",
            console_message=_timing_summary_console_message("Render-All Timing", timing),
        )
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
        if saver is not None:
            try:
                saver.close()
            except Exception:
                pass
        renderer_cache.close()


def _multiview_chunk_sizes(config) -> tuple[int, ...]:
    if getattr(config, "canonical_mv_conditions", False):
        return (8, 4, 2, 1)
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
    render_modes: tuple[str, ...],
) -> tuple[list[tuple[pathlib.Path, object]], list[tuple[int, str | None, float, float, str, pathlib.Path]], float, float]:
    prepared_rows = []
    rows = []
    prepare_views_ms = 0.0
    render_ms = 0.0
    for view_index, label, elev, azim in chunk_specs:
        view_config = replace(base_config, elev=elev, azim=azim)
        prepare_start = time.perf_counter()
        prepared = renderer.prepare_view(assets, config=view_config)
        prepare_views_ms += (time.perf_counter() - prepare_start) * 1000.0
        prepared_rows.append((prepared, view_index, label, elev, azim))
    render_start = time.perf_counter()
    images = [renderer.render_prepared_modes(prepared, render_modes) for prepared, _view_index, _label, _elev, _azim in prepared_rows]
    render_ms += (time.perf_counter() - render_start) * 1000.0
    outputs: list[tuple[pathlib.Path, object]] = []
    for (prepared, view_index, label, elev, azim), image_map in zip(prepared_rows, images):
        del prepared
        for mode in render_modes:
            output_path = _multi_view_output_path(
                output_dir,
                suffix,
                view_index,
                elev,
                azim,
                label=label,
                mode=mode if len(render_modes) > 1 else None,
            )
            outputs.append((output_path, image_map[mode]))
            rows.append((view_index, label, elev, azim, mode, output_path))
    return outputs, rows, prepare_views_ms, render_ms


def _render_multiview_pass(
    renderer_cache: RendererCache,
    current_renderer: SceneRenderer,
    assets,
    base_config,
    output_dir: pathlib.Path,
    suffix: str,
    view_specs: list[tuple[int, str | None, float, float]],
    render_modes: tuple[str, ...],
    chunk_sizes: tuple[int, ...],
    logger: RunLogger,
) -> tuple[list[pathlib.Path], list[tuple[int, str | None, float, float, str, pathlib.Path]], list[int], int, float, float, float, float]:
    rows = []
    outputs = []
    attempted_chunk_sizes = [chunk_sizes[0]]
    chunk_size_index = 0
    chunk_start = 0
    progress_start = time.perf_counter()
    total_outputs = len(view_specs) * len(render_modes)
    session_init_ms = 0.0
    prepare_views_ms_total = 0.0
    render_ms_total = 0.0
    save_ms_total = 0.0
    saver = AsyncImageSaver(
        max_pending=max(chunk_sizes[0] * len(render_modes), 2),
        png_compression=base_config.png_compression,
    )
    try:
        while chunk_start < len(view_specs):
            chunk_size = chunk_sizes[chunk_size_index]
            chunk_specs = view_specs[chunk_start: chunk_start + chunk_size]
            chunk_index = (chunk_start // chunk_size) + 1
            chunk_started = time.perf_counter()
            logger.log(f"Rendering multi-view chunk {chunk_index} ({len(chunk_specs)} view(s), chunk size {chunk_size})")
            init_start = time.perf_counter()
            renderer, _created = renderer_cache.get_with_status(base_config)
            if _created or renderer is not current_renderer:
                session_init_ms += (time.perf_counter() - init_start) * 1000.0
            current_renderer = renderer
            try:
                chunk_outputs, chunk_rows, chunk_prepare_views_ms, chunk_render_ms = _render_multiview_chunk(
                    renderer,
                    assets,
                    base_config,
                    output_dir,
                    suffix,
                    chunk_specs,
                    render_modes,
                )
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
                    current_renderer = None
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
            for output_path, image in chunk_outputs:
                outputs.append(output_path)
                for _saved_path, duration_ms in saver.submit(image, output_path):
                    save_ms_total += duration_ms
            rows.extend(chunk_rows)
            prepare_views_ms_total += chunk_prepare_views_ms
            render_ms_total += chunk_render_ms
            elapsed_ms = (time.perf_counter() - progress_start) * 1000.0
            chunk_duration_ms = (time.perf_counter() - chunk_started) * 1000.0
            eta_ms = estimate_remaining_ms(len(rows), total_outputs, elapsed_ms)
            logger.log(
                f"Completed multi-view chunk {chunk_index}; saved {len(rows)}/{total_outputs} output(s)",
                console="always",
                console_message=_progress_line(
                    "multi-view",
                    len(rows),
                    total_outputs,
                    item_label=f"chunk {chunk_index}",
                    last_ms=chunk_duration_ms,
                    eta_ms=eta_ms,
                ),
            )
            chunk_start += len(chunk_specs)
        for _saved_path, duration_ms in saver.finish():
            save_ms_total += duration_ms
        return (
            outputs,
            rows,
            attempted_chunk_sizes,
            chunk_sizes[chunk_size_index],
            session_init_ms,
            prepare_views_ms_total,
            render_ms_total,
            save_ms_total,
        )
    finally:
        try:
            saver.close()
        except Exception:
            pass


# Standalone multi-view is also sequential: one renderer/context per invocation, no raster thread pool.
def _render_multiview_from_config(config) -> tuple[list[pathlib.Path], pathlib.Path]:
    if config.render_all:
        raise ValueError("--render-all is not supported together with multi-view rendering")
    view_specs = _multi_view_specs(config)
    render_modes = _multiview_render_modes(config)
    output_dir = _render_all_output_dir(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(_multi_view_log_path(output_dir), echo=config.print_progress)
    logger.reset()
    total_outputs = len(view_specs) * len(render_modes)
    logger.log(
        f"Starting multi-view run for {len(view_specs)} view(s) / {total_outputs} output(s) -> {output_dir}",
        console="always",
        console_message="[Info] Multi-View Start: Initializing",
    )
    logger.log(
        f"Multi-view total views: {len(view_specs)}; render modes per view: {', '.join(render_modes)}",
        console="always",
        console_message=f"[Info] Multi-View Items: total={total_outputs}",
    )
    if config.display:
        logger.log("Ignoring --display during multi-view rendering.")
    suffix = _render_all_suffix(config.output)
    base_config = replace(config, display=False)
    chunk_sizes = _multiview_chunk_sizes(base_config)
    renderer_cache = RendererCache(device=torch.device(f"cuda:{torch.cuda.current_device()}"), logger=logger)
    run_started = time.perf_counter()
    session_init_ms = 0.0
    data_loading_ms = 0.0
    try:
        init_start = time.perf_counter()
        renderer, _created = renderer_cache.get_with_status(base_config)
        if _created:
            session_init_ms += (time.perf_counter() - init_start) * 1000.0
        assets_start = time.perf_counter()
        assets = renderer.prepare_assets(pathlib.Path(config.input))
        data_loading_ms += (time.perf_counter() - assets_start) * 1000.0
        if getattr(config, "canonical_mv_conditions", False):
            logger.log(
                "Using canonical twelve-view condition set with sequential chunk rendering and modes normal_ogl, position_ogl; initial chunk size 8."
            )
        elif config.canonical_six_views:
            logger.log("Using canonical six-view set with sequential chunk rendering and CUDA OOM fallback through chunk sizes 6, 2, 1.")
        else:
            logger.log(
                f"Using sequential multi-view chunk rendering with initial chunk size {chunk_sizes[0]} "
                "and renderer recreation on CUDA OOM."
            )
        (
            outputs,
            rows,
            attempted_chunk_sizes,
            chunk_size_used,
            extra_session_init_ms,
            prepare_views_ms_total,
            render_ms_total,
            save_ms_total,
        ) = _render_multiview_pass(
            renderer_cache,
            renderer,
            assets,
            base_config,
            output_dir,
            suffix,
            view_specs,
            render_modes,
            chunk_sizes=chunk_sizes,
            logger=logger,
        )
        timing = TimingBreakdown(
            total_ms=(time.perf_counter() - run_started) * 1000.0,
            session_init_ms=session_init_ms + extra_session_init_ms,
            data_loading_ms=data_loading_ms,
            scene_prepare_ms=prepare_views_ms_total,
            render_ms=render_ms_total,
            save_ms=save_ms_total,
        )
        chunk_size_info = _format_chunk_size_info(attempted_chunk_sizes, chunk_size_used)
        report = _format_multiview_report(config, rows, chunk_size_info, timing)
        report_path = _write_multiview_report(output_dir, report)
        logger.log(report.rstrip())
        logger.log(f"Saved {len(outputs)} multi-view renders under {output_dir}")
        logger.log(f"Saved multi-view report to {report_path}")
        logger.log(
            f"Multi-view timing summary: total_ms={timing.total_ms:.3f}",
            console="always",
            console_message=_timing_summary_console_message("Multi-View Timing", timing),
        )
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
        raise ValueError("Specify --canonical-six-views, --canonical-mv-conditions, or provide at least one full multi-view range triplet")
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
    logger.log("Starting single render run.", console="always", console_message="[Info] Render Start: Initializing")
    try:
        output_path = renderer_cache.get(config).render_to_file()
        duration_ms = (time.perf_counter() - render_started) * 1000.0
        logger.log(
            "Finished single render run.",
            console="always",
            console_message=_progress_line("render", 1, 1, item_label="completed", last_ms=duration_ms, eta_ms=0.0),
        )
        logger.log(
            "Single render completed successfully.",
            console="always",
            console_message=format_path_notice("Info", "Done. Output:", output_path),
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
