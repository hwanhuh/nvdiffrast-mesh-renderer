import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import pathlib
import queue as queue_module
import shutil
import sqlite3
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import RenderConfig, add_render_arguments, config_from_args, config_with_overrides
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
PRESET_VIEW_SETS = {
    "canonical_six": CANONICAL_SIX_VIEW_SPECS,
    "canonical-six": CANONICAL_SIX_VIEW_SPECS,
}


@dataclass(frozen=True)
class CandidateEntry:
    row_index: int
    raw_row: dict[str, Any]
    mesh_id: str | None
    selected_index: int
    error_message: str | None = None


@dataclass(frozen=True)
class ManifestRow:
    row_index: int
    selected_index: int
    mesh_id: str
    input_path: pathlib.Path
    output_rel: str | None
    views_json: Any | None
    views_file: pathlib.Path | None
    overrides: dict[str, Any]
    tags: dict[str, Any] | None
    source_row: dict[str, Any]


@dataclass(frozen=True)
class ViewSpec:
    index: int
    name: str
    elev: float
    azim: float
    distance: float | None = None
    distance_scale: float | None = None
    fov: float | None = None


@dataclass(frozen=True)
class JobSpec:
    row_index: int
    selected_index: int
    mesh_id: str
    input_path: str
    output_dir: str
    temp_output_dir: str
    render_config: RenderConfig
    views: tuple[ViewSpec, ...]
    image_format: str
    jpg_quality: int
    keep_partial_on_failure: bool
    overwrite: str
    view_chunk_sizes: tuple[int, ...]
    source_row: dict[str, Any]
    tags: dict[str, Any] | None


@dataclass
class WorkerState:
    slot: int
    gpu_index: int
    job_queue: Any
    process: mp.Process
    current_job: JobSpec | None = None
    current_started_at: str | None = None
    current_started_mono: float | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round(value: float) -> float:
    return round(float(value), 3)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(payload), ensure_ascii=True) + "\n")


def _remove_path(path: pathlib.Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _stable_hash(mesh_id: str) -> int:
    digest = hashlib.sha1(mesh_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _require_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return dict(value)


def _parse_json_field(value: Any, *, field_name: str, allow_none: bool = True) -> Any:
    if value is None:
        return None if allow_none else {}
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None if allow_none else {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} is not valid JSON: {exc.msg}") from exc
    return value


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_mesh_id(value: Any) -> str:
    mesh_id = _normalize_optional_string(value)
    if mesh_id is None:
        raise ValueError("mesh_id is required")
    if mesh_id in {".", ".."} or "/" in mesh_id or "\\" in mesh_id:
        raise ValueError("mesh_id must not contain path separators")
    return mesh_id


def _resolve_relative_path(base_dir: pathlib.Path, value: Any, *, field_name: str) -> pathlib.Path:
    text = _normalize_optional_string(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _validate_output_rel(output_root: pathlib.Path, value: Any) -> str | None:
    output_rel = _normalize_optional_string(value)
    if output_rel is None:
        return None
    path = pathlib.Path(output_rel)
    if path.is_absolute():
        raise ValueError("output_rel must be relative to output_root")
    resolved = (output_root / path).resolve()
    output_root_resolved = output_root.resolve()
    if resolved != output_root_resolved and output_root_resolved not in resolved.parents:
        raise ValueError("output_rel must stay within output_root")
    return path.as_posix()


def _load_manifest(manifest_path: pathlib.Path, manifest_format: str) -> list[dict[str, Any]]:
    if manifest_format == "auto":
        suffix = manifest_path.suffix.lower()
        if suffix == ".jsonl":
            manifest_format = "jsonl"
        elif suffix == ".csv":
            manifest_format = "csv"
        else:
            raise ValueError("Could not infer manifest format; use --manifest-format jsonl or csv")
    if manifest_format == "jsonl":
        rows = []
        for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL at line {line_number}: expected object")
            rows.append(dict(row))
        return rows
    if manifest_format == "csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    raise ValueError(f"Unsupported manifest format: {manifest_format}")


def _select_candidate_entries(
    raw_rows: list[dict[str, Any]],
    shard_index: int | None,
    num_shards: int | None,
    limit: int | None,
) -> list[CandidateEntry]:
    entries: list[CandidateEntry] = []
    for row_index, raw_row in enumerate(raw_rows):
        try:
            enabled = _parse_bool(raw_row.get("enabled", True), field_name="enabled")
        except ValueError as exc:
            entries.append(CandidateEntry(row_index=row_index, raw_row=raw_row, mesh_id=None, selected_index=-1, error_message=str(exc)))
            continue
        if not enabled:
            continue
        try:
            mesh_id = _validate_mesh_id(raw_row.get("mesh_id"))
        except ValueError as exc:
            entries.append(CandidateEntry(row_index=row_index, raw_row=raw_row, mesh_id=None, selected_index=-1, error_message=str(exc)))
            continue
        if num_shards is not None and shard_index is not None and (_stable_hash(mesh_id) % num_shards) != shard_index:
            continue
        entries.append(CandidateEntry(row_index=row_index, raw_row=raw_row, mesh_id=mesh_id, selected_index=-1))
    if limit is not None:
        entries = entries[:limit]
    return [
        CandidateEntry(
            row_index=entry.row_index,
            raw_row=entry.raw_row,
            mesh_id=entry.mesh_id,
            selected_index=index,
            error_message=entry.error_message,
        )
        for index, entry in enumerate(entries)
    ]


def _normalize_manifest_row(entry: CandidateEntry, manifest_dir: pathlib.Path, output_root: pathlib.Path) -> ManifestRow:
    raw_row = dict(entry.raw_row)
    assert entry.mesh_id is not None
    input_path = _resolve_relative_path(manifest_dir, raw_row.get("input"), field_name="input")
    views_json = _parse_json_field(raw_row.get("views_json"), field_name="views_json")
    views_file_raw = _normalize_optional_string(raw_row.get("views_file"))
    views_file = None if views_file_raw is None else _resolve_relative_path(manifest_dir, views_file_raw, field_name="views_file")
    overrides_json = _parse_json_field(raw_row.get("overrides_json"), field_name="overrides_json", allow_none=False)
    overrides = {} if overrides_json is None else _require_dict(overrides_json, field_name="overrides_json")
    tags_json = _parse_json_field(raw_row.get("tags_json"), field_name="tags_json")
    tags = None if tags_json is None else _require_dict(tags_json, field_name="tags_json")
    output_rel = _validate_output_rel(output_root, raw_row.get("output_rel"))
    return ManifestRow(
        row_index=entry.row_index,
        selected_index=entry.selected_index,
        mesh_id=entry.mesh_id,
        input_path=input_path,
        output_rel=output_rel,
        views_json=views_json,
        views_file=views_file,
        overrides=overrides,
        tags=tags,
        source_row=raw_row,
    )


def _ensure_unique_mesh_ids(entries: list[CandidateEntry]) -> None:
    seen: dict[str, int] = {}
    for entry in entries:
        if entry.error_message is not None or entry.mesh_id is None:
            continue
        if entry.mesh_id in seen:
            first_row = seen[entry.mesh_id] + 1
            second_row = entry.row_index + 1
            raise ValueError(f"Duplicate mesh_id '{entry.mesh_id}' in selected rows (lines {first_row} and {second_row})")
        seen[entry.mesh_id] = entry.row_index


def _parse_gpu_list(value: str | None) -> list[int]:
    if value is None or value.strip() == "":
        return [torch.cuda.current_device()]
    gpu_list = []
    for item in value.split(","):
        stripped = item.strip()
        if stripped == "":
            continue
        gpu_list.append(int(stripped))
    if not gpu_list:
        raise ValueError("--gpus must specify at least one device index")
    device_count = torch.cuda.device_count()
    for gpu_index in gpu_list:
        if gpu_index < 0 or gpu_index >= device_count:
            raise ValueError(f"GPU index {gpu_index} is out of range for {device_count} CUDA device(s)")
    return gpu_list


def _parse_view_chunk_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--view-chunk-sizes must be a comma-separated list of integers") from exc
    if not sizes:
        raise ValueError("--view-chunk-sizes must not be empty")
    if any(size <= 0 for size in sizes):
        raise ValueError("--view-chunk-sizes must contain positive integers")
    if any(left <= right for left, right in zip(sizes, sizes[1:])):
        raise ValueError("--view-chunk-sizes must be strictly descending")
    return sizes


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


def _format_angle(value: float) -> str:
    return f"{value:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "_")


def _legacy_view_name(elev: float, azim: float) -> str:
    return f"elev_{_format_angle(elev)}_azim_{_format_angle(azim)}"


def _sanitize_view_name(name: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in name.strip())
    sanitized = sanitized.strip("._")
    if not sanitized:
        raise ValueError("view name is empty after sanitization")
    return sanitized


def _parse_view_object(value: Any, *, name_override: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("view entry must be an object")
    if "elev" not in value or "azim" not in value:
        raise ValueError("view entry must define both elev and azim")
    name = name_override if name_override is not None else value.get("name")
    return {
        "name": None if name is None else str(name),
        "elev": float(value["elev"]),
        "azim": float(value["azim"]),
        "distance": None if value.get("distance") is None else float(value["distance"]),
        "distance_scale": None if value.get("distance_scale") is None else float(value["distance_scale"]),
        "fov": None if value.get("fov") is None else float(value["fov"]),
    }


def _parse_grid_axis(value: Any, *, axis_name: str) -> tuple[float, float, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{axis_name} grid spec must be an object")
    if {"start", "end", "step"} - set(value):
        raise ValueError(f"{axis_name} grid spec requires start, end, and step")
    start = float(value["start"])
    end = float(value["end"])
    step = float(value["step"])
    if step == 0.0:
        raise ValueError(f"{axis_name}.step must be non-zero")
    return start, end, step


def _parse_preset_views(name: str) -> tuple[list[dict[str, Any]], str | None]:
    preset_key = name.strip().lower()
    if preset_key not in PRESET_VIEW_SETS:
        supported = ", ".join(sorted(PRESET_VIEW_SETS))
        raise ValueError(f"Unsupported view preset '{name}'. Supported presets: {supported}")
    return [
        {"name": label, "elev": elev, "azim": azim, "distance": None, "distance_scale": None, "fov": None}
        for label, elev, azim in PRESET_VIEW_SETS[preset_key]
    ], None


def _parse_view_source_value(value: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(value, list):
        return [_parse_view_object(item) for item in value], None
    if isinstance(value, dict):
        if value.get("type") == "grid":
            elev_start, elev_end, elev_step = _parse_grid_axis(value.get("elev"), axis_name="elev")
            azim_start, azim_end, azim_step = _parse_grid_axis(value.get("azim"), axis_name="azim")
            elevs = _axis_values(elev_start, elev_end, elev_step, 0.0)
            azims = _axis_values(azim_start, azim_end, azim_step, 0.0)
            order = str(value.get("order", "elev_major"))
            if order not in {"elev_major", "azim_major"}:
                raise ValueError("grid order must be 'elev_major' or 'azim_major'")
            if order == "elev_major":
                views = [{"name": None, "elev": elev, "azim": azim, "distance": None, "distance_scale": None, "fov": None} for elev in elevs for azim in azims]
            else:
                views = [{"name": None, "elev": elev, "azim": azim, "distance": None, "distance_scale": None, "fov": None} for azim in azims for elev in elevs]
            return views, None if value.get("name_template") is None else str(value["name_template"])
        if "preset" in value:
            return _parse_preset_views(str(value["preset"]))
        return [_parse_view_object(item, name_override=name) for name, item in value.items()], None
    raise ValueError("views spec must be a list or object")


@dataclass(frozen=True)
class GlobalViewSource:
    kind: str
    value: Any


def _parse_cli_views_json(value: str) -> Any:
    if value.startswith("@"):
        path = pathlib.Path(value[1:]).expanduser().resolve()
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"--views-json file does not exist: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"--views-json file contains invalid JSON: {exc.msg}") from exc
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--views-json is not valid JSON: {exc.msg}") from exc


def _build_global_view_source(args: argparse.Namespace) -> GlobalViewSource | None:
    sources: list[GlobalViewSource] = []
    if getattr(args, "views_json", None):
        sources.append(GlobalViewSource(kind="views_json", value=_parse_cli_views_json(args.views_json)))
    if getattr(args, "views_file", None):
        view_path = pathlib.Path(args.views_file).expanduser().resolve()
        if not view_path.exists():
            raise ValueError(f"--views-file does not exist: {view_path}")
        sources.append(GlobalViewSource(kind="views_file", value=view_path))
    if getattr(args, "view_preset", None):
        sources.append(GlobalViewSource(kind="view_preset", value=str(args.view_preset)))
    has_range_source = any(
        getattr(args, name, None) is not None
        for name in ("elev_start", "elev_end", "elev_step", "azim_start", "azim_end", "azim_step")
    )
    if has_range_source:
        sources.append(GlobalViewSource(kind="range", value=None))
    if getattr(args, "canonical_six_views", False):
        sources.append(GlobalViewSource(kind="canonical_six", value="canonical_six"))
    if len(sources) > 1:
        raise ValueError("Specify exactly one global view source among --views-json, --views-file, --view-preset, range flags, or --canonical-six-views")
    return sources[0] if sources else None


def _finalize_view_specs(raw_views: list[dict[str, Any]], *, default_name_template: str | None, source_name_template: str | None) -> tuple[ViewSpec, ...]:
    finalized: list[ViewSpec] = []
    seen_names: set[str] = set()
    active_template = source_name_template if source_name_template is not None else default_name_template
    for index, raw_view in enumerate(raw_views):
        explicit_name = raw_view.get("name")
        if explicit_name is not None:
            final_name = _sanitize_view_name(str(explicit_name))
        elif active_template:
            try:
                formatted = active_template.format(index=index, elev=float(raw_view["elev"]), azim=float(raw_view["azim"]))
            except Exception as exc:
                raise ValueError(f"invalid view name template: {exc}") from exc
            final_name = _sanitize_view_name(formatted)
        else:
            final_name = _legacy_view_name(float(raw_view["elev"]), float(raw_view["azim"]))
        if final_name in seen_names:
            raise ValueError(f"duplicate view name after sanitization: {final_name}")
        seen_names.add(final_name)
        finalized.append(
            ViewSpec(
                index=index,
                name=final_name,
                elev=float(raw_view["elev"]),
                azim=float(raw_view["azim"]),
                distance=None if raw_view.get("distance") is None else float(raw_view["distance"]),
                distance_scale=None if raw_view.get("distance_scale") is None else float(raw_view["distance_scale"]),
                fov=None if raw_view.get("fov") is None else float(raw_view["fov"]),
            )
        )
    return tuple(finalized)


def _resolve_job_views(row: ManifestRow, global_source: GlobalViewSource | None, config: RenderConfig, default_name_template: str | None) -> tuple[ViewSpec, ...]:
    if row.views_json is not None and row.views_file is not None:
        raise ValueError("manifest row cannot define both views_json and views_file")
    if row.views_json is not None:
        raw_views, source_template = _parse_view_source_value(row.views_json)
        return _finalize_view_specs(raw_views, default_name_template=default_name_template, source_name_template=source_template)
    if row.views_file is not None:
        try:
            file_value = json.loads(row.views_file.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"views_file does not exist: {row.views_file}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"views_file contains invalid JSON: {exc.msg}") from exc
        raw_views, source_template = _parse_view_source_value(file_value)
        return _finalize_view_specs(raw_views, default_name_template=default_name_template, source_name_template=source_template)
    if global_source is None:
        raise ValueError("no view source supplied for row")
    if global_source.kind == "views_json":
        raw_views, source_template = _parse_view_source_value(global_source.value)
        return _finalize_view_specs(raw_views, default_name_template=default_name_template, source_name_template=source_template)
    if global_source.kind == "views_file":
        try:
            file_value = json.loads(pathlib.Path(global_source.value).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"--views-file contains invalid JSON: {exc.msg}") from exc
        raw_views, source_template = _parse_view_source_value(file_value)
        return _finalize_view_specs(raw_views, default_name_template=default_name_template, source_name_template=source_template)
    if global_source.kind == "view_preset":
        raw_views, source_template = _parse_preset_views(str(global_source.value))
        return _finalize_view_specs(raw_views, default_name_template=default_name_template, source_name_template=source_template)
    if global_source.kind == "canonical_six":
        raw_views, source_template = _parse_preset_views("canonical_six")
        return _finalize_view_specs(raw_views, default_name_template=default_name_template, source_name_template=source_template)
    elevs = _axis_values(config.elev_start, config.elev_end, config.elev_step, config.elev)
    azims = _axis_values(config.azim_start, config.azim_end, config.azim_step, config.azim)
    raw_views = [{"name": None, "elev": elev, "azim": azim, "distance": None, "distance_scale": None, "fov": None} for elev in elevs for azim in azims]
    return _finalize_view_specs(raw_views, default_name_template=default_name_template, source_name_template=None)


def _build_output_dir(output_root: pathlib.Path, row: ManifestRow, meshes_per_output_shard: int) -> pathlib.Path:
    if row.output_rel is not None:
        return (output_root / row.output_rel).resolve()
    shard_index = row.selected_index // meshes_per_output_shard
    return output_root / f"shard-{shard_index:06d}" / row.mesh_id


def _validate_output_root(output_root: pathlib.Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    probe_path = output_root / ".batch_write_probe"
    probe_path.write_text("ok", encoding="utf-8")
    probe_path.unlink()


def _batch_log_path(output_root: pathlib.Path) -> pathlib.Path:
    return output_root / "batch.log"


def _init_status_db(path: pathlib.Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batch_status (
            mesh_id TEXT PRIMARY KEY,
            input_path TEXT,
            status TEXT,
            started_at TEXT,
            finished_at TEXT,
            attempt_count INTEGER,
            output_dir TEXT,
            error_type TEXT,
            error_message TEXT,
            worker_id TEXT,
            gpu_index INTEGER
        )
        """
    )
    conn.commit()
    return conn


def _status_row(conn: sqlite3.Connection, mesh_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM batch_status WHERE mesh_id = ?", (mesh_id,)).fetchone()


def _upsert_status(
    conn: sqlite3.Connection,
    *,
    mesh_id: str,
    input_path: str,
    status: str,
    output_dir: str | None,
    started_at: str | None,
    finished_at: str | None,
    error_type: str | None,
    error_message: str | None,
    worker_id: str | None,
    gpu_index: int | None,
    increment_attempt: bool,
) -> int:
    previous = _status_row(conn, mesh_id)
    attempt_count = (0 if previous is None or previous["attempt_count"] is None else int(previous["attempt_count"])) + (1 if increment_attempt else 0)
    if previous is None:
        conn.execute(
            """
            INSERT INTO batch_status (
                mesh_id, input_path, status, started_at, finished_at, attempt_count,
                output_dir, error_type, error_message, worker_id, gpu_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (mesh_id, input_path, status, started_at, finished_at, attempt_count, output_dir, error_type, error_message, worker_id, gpu_index),
        )
    else:
        conn.execute(
            """
            UPDATE batch_status
            SET input_path = ?, status = ?, started_at = ?, finished_at = ?, attempt_count = ?,
                output_dir = ?, error_type = ?, error_message = ?, worker_id = ?, gpu_index = ?
            WHERE mesh_id = ?
            """,
            (input_path, status, started_at, finished_at, attempt_count, output_dir, error_type, error_message, worker_id, gpu_index, mesh_id),
        )
    conn.commit()
    return attempt_count


def _existing_success_output(output_dir: pathlib.Path) -> bool:
    if not output_dir.is_dir():
        return False
    report_path = output_dir / "job_report.json"
    cameras_path = output_dir / "cameras.json"
    if not report_path.is_file() or not cameras_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return report.get("status") == "success"


def _skip_reason(conn: sqlite3.Connection, job: JobSpec) -> str | None:
    output_dir = pathlib.Path(job.output_dir)
    status = _status_row(conn, job.mesh_id)
    has_existing_success = _existing_success_output(output_dir)
    if job.overwrite == "all":
        return None
    if has_existing_success:
        return "existing successful output"
    if status is not None and status["status"] == "success" and output_dir.exists():
        return "status DB marks job successful"
    if job.overwrite == "failed":
        return None
    if status is not None and status["status"] == "failed":
        return "previous failed status with overwrite=never"
    if output_dir.exists():
        return "output directory already exists"
    return None


def _view_config_for_job(job: JobSpec, view: ViewSpec) -> RenderConfig:
    return replace(
        job.render_config,
        elev=view.elev,
        azim=view.azim,
        distance=job.render_config.distance if view.distance is None else view.distance,
        distance_scale=job.render_config.distance_scale if view.distance_scale is None else view.distance_scale,
        fov=job.render_config.fov if view.fov is None else view.fov,
    )


def _save_image_array(image: np.ndarray, output_path: pathlib.Path, image_format: str, jpg_quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "jpg":
        rgb = (image[..., :3] * 255.0).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(output_path, format="JPEG", quality=jpg_quality)
        return
    rgba = (image * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(output_path, format="PNG")


def _tensor_to_list(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().tolist()


def _build_camera_entry(
    *,
    job: JobSpec,
    view: ViewSpec,
    view_config: RenderConfig,
    prepared: Any,
    scene_center: np.ndarray,
    image_name: str,
) -> dict[str, Any]:
    position = prepared.camera.position.detach().cpu().numpy()
    distance = float(np.linalg.norm(position - scene_center))
    return {
        "index": view.index,
        "name": view.name,
        "elev_deg": _round(view.elev),
        "azim_deg": _round(view.azim),
        "distance": _round(distance),
        "distance_scale": _round(view_config.distance_scale),
        "fov_deg": _round(view_config.fov),
        "image_relpath": image_name,
        "view_matrix": _tensor_to_list(prepared.camera.view),
        "projection_matrix": _tensor_to_list(prepared.camera.proj),
        "mvp_matrix": _tensor_to_list(prepared.camera.mvp),
        "camera_position": _tensor_to_list(prepared.camera.position),
        "camera_to_world": _tensor_to_list(prepared.camera.cam_to_world),
    }


def _render_chunk_arrays(renderer: SceneRenderer, prepared_rows: list[tuple[Any, ...]]) -> list[np.ndarray]:
    # One nvdiffrast CUDA context is owned by one worker and used sequentially.
    return [renderer.render_prepared(prepared) for _view, _view_config, prepared, _image_name in prepared_rows]


def _run_render_attempt(job: JobSpec, renderer: SceneRenderer, assets: Any, chunk_size: int) -> tuple[list[dict[str, Any]], float, float, float]:
    temp_output_dir = pathlib.Path(job.temp_output_dir)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    prepare_views_ms = 0.0
    render_ms = 0.0
    save_ms = 0.0
    cameras: list[dict[str, Any]] = []
    image_suffix = ".jpg" if job.image_format == "jpg" else ".png"
    for chunk_start in range(0, len(job.views), chunk_size):
        chunk_views = job.views[chunk_start: chunk_start + chunk_size]
        prepared_rows = []
        for view in chunk_views:
            view_config = _view_config_for_job(job, view)
            start = time.perf_counter()
            prepared = renderer.prepare_view(assets, config=view_config)
            prepare_views_ms += (time.perf_counter() - start) * 1000.0
            image_name = f"{view.index:04d}_{view.name}{image_suffix}"
            prepared_rows.append((view, view_config, prepared, image_name))
        start = time.perf_counter()
        images = _render_chunk_arrays(renderer, prepared_rows)
        render_ms += (time.perf_counter() - start) * 1000.0
        for (view, view_config, prepared, image_name), image in zip(prepared_rows, images):
            start = time.perf_counter()
            _save_image_array(image, temp_output_dir / image_name, job.image_format, job.jpg_quality)
            save_ms += (time.perf_counter() - start) * 1000.0
            cameras.append(
                _build_camera_entry(
                    job=job,
                    view=view,
                    view_config=view_config,
                    prepared=prepared,
                    scene_center=assets.center,
                    image_name=image_name,
                )
            )
    return cameras, prepare_views_ms, render_ms, save_ms


def _build_cameras_payload(job: JobSpec, cameras: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mesh_id": job.mesh_id,
        "input_path": job.input_path,
        "image_format": job.image_format,
        "resolution": job.render_config.resolution,
        "render_mode": job.render_config.render_mode,
        "view_count": len(cameras),
        "views": cameras,
    }


def _build_job_report(
    job: JobSpec,
    *,
    status: str,
    started_at: str,
    finished_at: str,
    duration_ms: float,
    session_init_ms: float,
    prepare_assets_ms: float,
    prepare_views_ms_total: float,
    render_ms_total: float,
    save_ms_total: float,
    view_chunk_sizes_attempted: list[int],
    view_chunk_size_used: int | None,
    oom_retries: int,
    output_dir: str,
    error_type: str | None = None,
    error_message: str | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    payload = {
        "mesh_id": job.mesh_id,
        "input_path": job.input_path,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": _round(duration_ms),
        "session_init_ms": _round(session_init_ms),
        "prepare_assets_ms": _round(prepare_assets_ms),
        "prepare_views_ms_total": _round(prepare_views_ms_total),
        "render_ms_total": _round(render_ms_total),
        "save_ms_total": _round(save_ms_total),
        "view_count": len(job.views),
        "view_chunk_sizes_attempted": view_chunk_sizes_attempted,
        "view_chunk_size_used": view_chunk_size_used,
        "oom_retries": oom_retries,
        "output_dir": output_dir,
        "source_row": job.source_row,
    }
    if error_type is not None:
        payload["error_type"] = error_type
    if error_message is not None:
        payload["error_message"] = error_message
    if traceback_text is not None:
        payload["traceback"] = traceback_text
    if job.tags is not None:
        payload["tags"] = job.tags
    return payload


def _write_failed_job_report_if_needed(
    job: JobSpec,
    *,
    started_at: str,
    finished_at: str,
    duration_ms: float,
    session_init_ms: float,
    prepare_assets_ms: float,
    prepare_views_ms_total: float,
    render_ms_total: float,
    save_ms_total: float,
    view_chunk_sizes_attempted: list[int],
    view_chunk_size_used: int | None,
    oom_retries: int,
    error_type: str,
    error_message: str,
    traceback_text: str | None,
) -> None:
    temp_output_dir = pathlib.Path(job.temp_output_dir)
    if not job.keep_partial_on_failure:
        if temp_output_dir.exists():
            _remove_path(temp_output_dir)
        return
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    report = _build_job_report(
        job,
        status="failed",
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        session_init_ms=session_init_ms,
        prepare_assets_ms=prepare_assets_ms,
        prepare_views_ms_total=prepare_views_ms_total,
        render_ms_total=render_ms_total,
        save_ms_total=save_ms_total,
        view_chunk_sizes_attempted=view_chunk_sizes_attempted,
        view_chunk_size_used=view_chunk_size_used,
        oom_retries=oom_retries,
        output_dir=job.output_dir,
        error_type=error_type,
        error_message=error_message,
        traceback_text=traceback_text,
    )
    _write_json(temp_output_dir / "job_report.json", report)


def _execute_job(job: JobSpec, gpu_index: int, renderer_cache: RendererCache, worker_slot: int, logger: RunLogger) -> dict[str, Any]:
    torch.cuda.set_device(gpu_index)
    worker_id = f"worker-{worker_slot}-gpu{gpu_index}"
    output_dir = pathlib.Path(job.output_dir)
    temp_output_dir = pathlib.Path(job.temp_output_dir)
    started_at = _utc_now()
    overall_start = time.perf_counter()
    session_init_ms = 0.0
    prepare_assets_ms = 0.0
    prepare_views_ms_total = 0.0
    render_ms_total = 0.0
    save_ms_total = 0.0
    attempted_chunk_sizes: list[int] = []
    oom_retries = 0
    view_chunk_size_used: int | None = None
    renderer: SceneRenderer | None = None
    assets: Any | None = None
    try:
        logger.log(f"Started job {job.mesh_id} with {len(job.views)} view(s) -> {job.output_dir}")
        if temp_output_dir.exists():
            _remove_path(temp_output_dir)
        if output_dir.exists() and job.overwrite in {"all", "failed"}:
            _remove_path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        renderer_cache.clear_texture_caches()
        init_start = time.perf_counter()
        renderer, created = renderer_cache.get_with_status(job.render_config)
        if created:
            session_init_ms += (time.perf_counter() - init_start) * 1000.0
        prepare_start = time.perf_counter()
        assets = renderer.prepare_assets(pathlib.Path(job.input_path))
        prepare_assets_ms = (time.perf_counter() - prepare_start) * 1000.0
        last_exc: BaseException | None = None
        cameras: list[dict[str, Any]] | None = None
        for chunk_size in job.view_chunk_sizes:
            attempted_chunk_sizes.append(chunk_size)
            if temp_output_dir.exists():
                _remove_path(temp_output_dir)
            temp_output_dir.mkdir(parents=True, exist_ok=True)
            try:
                attempt_cameras, attempt_prepare_views_ms, attempt_render_ms, attempt_save_ms = _run_render_attempt(job, renderer, assets, chunk_size)
                cameras = attempt_cameras
                prepare_views_ms_total = attempt_prepare_views_ms
                render_ms_total = attempt_render_ms
                save_ms_total = attempt_save_ms
                view_chunk_size_used = chunk_size
                break
            except Exception as exc:
                last_exc = exc
                if is_cuda_oom(exc) and chunk_size != job.view_chunk_sizes[-1]:
                    oom_retries += 1
                    next_chunk_size = job.view_chunk_sizes[job.view_chunk_sizes.index(chunk_size) + 1]
                    logger.log(
                        f"Job {job.mesh_id} hit CUDA OOM at chunk size {chunk_size}; "
                        f"recreating renderer and retrying with chunk size {next_chunk_size}."
                    )
                    renderer = None
                    renderer_cache.reset_after_cuda_failure(job.render_config)
                    init_start = time.perf_counter()
                    renderer, created = renderer_cache.get_with_status(job.render_config)
                    if created:
                        session_init_ms += (time.perf_counter() - init_start) * 1000.0
                    continue
                if is_cuda_failure(exc):
                    renderer = None
                    renderer_cache.reset_after_cuda_failure(job.render_config)
                raise
        if cameras is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("render attempt produced no output")
        cameras_payload = _build_cameras_payload(job, cameras)
        finished_at = _utc_now()
        duration_ms = (time.perf_counter() - overall_start) * 1000.0
        _write_json(temp_output_dir / "cameras.json", cameras_payload)
        _write_json(
            temp_output_dir / "job_report.json",
            _build_job_report(
                job,
                status="success",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                session_init_ms=session_init_ms,
                prepare_assets_ms=prepare_assets_ms,
                prepare_views_ms_total=prepare_views_ms_total,
                render_ms_total=render_ms_total,
                save_ms_total=save_ms_total,
                view_chunk_sizes_attempted=attempted_chunk_sizes,
                view_chunk_size_used=view_chunk_size_used,
                oom_retries=oom_retries,
                output_dir=job.output_dir,
            ),
        )
        temp_output_dir.rename(output_dir)
        logger.log(
            f"Completed job {job.mesh_id}: {len(job.views)} view(s), "
            f"chunk_size={view_chunk_size_used}, oom_retries={oom_retries}, output={job.output_dir}"
        )
        return {
            "worker_slot": worker_slot,
            "worker_id": worker_id,
            "gpu_index": gpu_index,
            "mesh_id": job.mesh_id,
            "input_path": job.input_path,
            "output_dir": job.output_dir,
            "status": "success",
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "session_init_ms": session_init_ms,
            "prepare_assets_ms": prepare_assets_ms,
            "prepare_views_ms_total": prepare_views_ms_total,
            "render_ms_total": render_ms_total,
            "save_ms_total": save_ms_total,
            "view_count": len(job.views),
            "view_chunk_sizes_attempted": attempted_chunk_sizes,
            "view_chunk_size_used": view_chunk_size_used,
            "oom_retries": oom_retries,
            "error_type": None,
            "error_message": None,
            "traceback": None,
        }
    except Exception as exc:
        logger.log(f"Failed job {job.mesh_id}: {type(exc).__name__}: {exc}")
        if is_cuda_failure(exc):
            renderer_cache.drop(job.render_config)
        finished_at = _utc_now()
        duration_ms = (time.perf_counter() - overall_start) * 1000.0
        error_type = type(exc).__name__
        error_message = str(exc)
        traceback_text = traceback.format_exc()
        _write_failed_job_report_if_needed(
            job,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            session_init_ms=session_init_ms,
            prepare_assets_ms=prepare_assets_ms,
            prepare_views_ms_total=prepare_views_ms_total,
            render_ms_total=render_ms_total,
            save_ms_total=save_ms_total,
            view_chunk_sizes_attempted=attempted_chunk_sizes,
            view_chunk_size_used=view_chunk_size_used,
            oom_retries=oom_retries,
            error_type=error_type,
            error_message=error_message,
            traceback_text=traceback_text,
        )
        return {
            "worker_slot": worker_slot,
            "worker_id": worker_id,
            "gpu_index": gpu_index,
            "mesh_id": job.mesh_id,
            "input_path": job.input_path,
            "output_dir": job.output_dir,
            "status": "failed",
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "session_init_ms": session_init_ms,
            "prepare_assets_ms": prepare_assets_ms,
            "prepare_views_ms_total": prepare_views_ms_total,
            "render_ms_total": render_ms_total,
            "save_ms_total": save_ms_total,
            "view_count": len(job.views),
            "view_chunk_sizes_attempted": attempted_chunk_sizes,
            "view_chunk_size_used": view_chunk_size_used,
            "oom_retries": oom_retries,
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback_text,
        }
    finally:
        renderer = None
        assets = None
        renderer_cache.clear_texture_caches()
        renderer_cache.release_cuda_memory()


def _worker_main(worker_slot: int, gpu_index: int, job_queue: Any, result_queue: Any, log_path: str, echo: bool) -> None:
    torch.cuda.set_device(gpu_index)
    # Batch mode isolates concurrency at the process level; each worker uses one renderer/context sequentially.
    logger = RunLogger(pathlib.Path(log_path), echo=echo, prefix=f"worker-{worker_slot}-gpu{gpu_index}")
    renderer_cache = RendererCache(device=torch.device(f"cuda:{gpu_index}"), logger=logger)
    while True:
        job = job_queue.get()
        if job is None:
            renderer_cache.close()
            return
        result_queue.put(_execute_job(job, gpu_index, renderer_cache, worker_slot, logger))


def _start_worker(ctx: mp.context.BaseContext, slot: int, gpu_index: int, result_queue: Any, log_path: str, echo: bool) -> WorkerState:
    job_queue = ctx.Queue()
    process = ctx.Process(target=_worker_main, args=(slot, gpu_index, job_queue, result_queue, log_path, echo), daemon=True)
    process.start()
    return WorkerState(slot=slot, gpu_index=gpu_index, job_queue=job_queue, process=process)


def _restart_worker(ctx: mp.context.BaseContext, worker: WorkerState, result_queue: Any, log_path: str, echo: bool) -> WorkerState:
    if worker.process.is_alive():
        worker.process.terminate()
    worker.process.join(timeout=5)
    return _start_worker(ctx, worker.slot, worker.gpu_index, result_queue, log_path, echo)


def _stop_worker(worker: WorkerState) -> None:
    if worker.process.is_alive():
        try:
            worker.job_queue.put(None)
        except Exception:
            pass
        worker.process.join(timeout=5)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout=5)


def _record_event(events_path: pathlib.Path, event_type: str, **payload: Any) -> None:
    _append_jsonl(events_path, {"type": event_type, "timestamp": _utc_now(), **payload})


def _finalize_job_result(
    conn: sqlite3.Connection,
    events_path: pathlib.Path,
    result: dict[str, Any],
) -> None:
    _upsert_status(
        conn,
        mesh_id=result["mesh_id"],
        input_path=result["input_path"],
        status=result["status"],
        output_dir=result["output_dir"],
        started_at=result["started_at"],
        finished_at=result["finished_at"],
        error_type=result["error_type"],
        error_message=result["error_message"],
        worker_id=result["worker_id"],
        gpu_index=result["gpu_index"],
        increment_attempt=False,
    )
    event_type = "job_succeeded" if result["status"] == "success" else "job_failed"
    _record_event(
        events_path,
        event_type,
        mesh_id=result["mesh_id"],
        gpu_index=result["gpu_index"],
        worker_id=result["worker_id"],
        duration_ms=_round(result["duration_ms"]),
        view_count=result["view_count"],
        status=result["status"],
        error_type=result["error_type"],
        error_message=result["error_message"],
    )


def _external_failure_result(job: JobSpec, worker: WorkerState, *, error_type: str, error_message: str, duration_ms: float) -> dict[str, Any]:
    finished_at = _utc_now()
    started_at = worker.current_started_at or finished_at
    _write_failed_job_report_if_needed(
        job,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        session_init_ms=0.0,
        prepare_assets_ms=0.0,
        prepare_views_ms_total=0.0,
        render_ms_total=0.0,
        save_ms_total=0.0,
        view_chunk_sizes_attempted=[],
        view_chunk_size_used=None,
        oom_retries=0,
        error_type=error_type,
        error_message=error_message,
        traceback_text=None,
    )
    return {
        "worker_slot": worker.slot,
        "worker_id": f"worker-{worker.slot}-gpu{worker.gpu_index}",
        "gpu_index": worker.gpu_index,
        "mesh_id": job.mesh_id,
        "input_path": job.input_path,
        "output_dir": job.output_dir,
        "status": "failed",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "session_init_ms": 0.0,
        "prepare_assets_ms": 0.0,
        "prepare_views_ms_total": 0.0,
        "render_ms_total": 0.0,
        "save_ms_total": 0.0,
        "view_count": len(job.views),
        "view_chunk_sizes_attempted": [],
        "view_chunk_size_used": None,
        "oom_retries": 0,
        "error_type": error_type,
        "error_message": error_message,
        "traceback": None,
    }


def _format_batch_timing_console(duration_ms: float, success_metrics: list[dict[str, Any]]) -> str:
    if not success_metrics:
        return f"[Info] Batch Timing: total_elapsed={format_duration_ms(duration_ms)}"
    count = len(success_metrics)
    avg_session_init_ms = sum(item["session_init_ms"] for item in success_metrics) / count
    avg_prepare_assets_ms = sum(item["prepare_assets_ms"] for item in success_metrics) / count
    avg_prepare_views_ms = sum(item["prepare_views_ms_total"] for item in success_metrics) / count
    avg_render_ms = sum(item["render_ms_total"] for item in success_metrics) / count
    avg_save_ms = sum(item["save_ms_total"] for item in success_metrics) / count
    aggregate_session_init_ms = sum(item["session_init_ms"] for item in success_metrics)
    aggregate_prepare_assets_ms = sum(item["prepare_assets_ms"] for item in success_metrics)
    aggregate_prepare_views_ms = sum(item["prepare_views_ms_total"] for item in success_metrics)
    aggregate_render_ms = sum(item["render_ms_total"] for item in success_metrics)
    aggregate_save_ms = sum(item["save_ms_total"] for item in success_metrics)
    return "\n".join(
        [
            (
                "[Info] Batch Timing: "
                f"total_elapsed={format_duration_ms(duration_ms)}, "
                f"avg_success_session_init={format_duration_ms(avg_session_init_ms)}, "
                f"avg_success_data_loading={format_duration_ms(avg_prepare_assets_ms)}, "
                f"avg_success_scene_prepare={format_duration_ms(avg_prepare_views_ms)}, "
                f"avg_success_render={format_duration_ms(avg_render_ms)}, "
                f"avg_success_save={format_duration_ms(avg_save_ms)}"
            ),
            (
                "[Info] Batch Worker Time: "
                f"aggregate_session_init={format_duration_ms(aggregate_session_init_ms)}, "
                f"aggregate_data_loading={format_duration_ms(aggregate_prepare_assets_ms)}, "
                f"aggregate_scene_prepare={format_duration_ms(aggregate_prepare_views_ms)}, "
                f"aggregate_render={format_duration_ms(aggregate_render_ms)}, "
                f"aggregate_save={format_duration_ms(aggregate_save_ms)}"
            ),
        ]
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch render GLB/GLTF meshes with nvdiffrast on CUDA.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL or CSV manifest")
    parser.add_argument("--manifest-format", choices=["auto", "jsonl", "csv"], default="auto", help="Manifest format")
    parser.add_argument("--output-root", required=True, help="Root directory for batch outputs and metadata")
    parser.add_argument("--gpus", default=None, help="Comma-separated CUDA device indices")
    parser.add_argument("--shard-index", type=int, default=None, help="Optional external shard index")
    parser.add_argument("--num-shards", type=int, default=None, help="Optional external shard count")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit after filtering")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from prior successful outputs and status state")
    parser.add_argument("--overwrite", choices=["never", "failed", "all"], default="never", help="Overwrite policy")
    parser.add_argument("--fail-fast", action="store_true", help="Abort the batch after the first failed job")
    parser.add_argument("--max-failures", type=int, default=None, help="Abort the batch after this many failures")
    parser.add_argument("--mesh-timeout-sec", type=float, default=None, help="Optional per-mesh wall-clock timeout")
    parser.add_argument("--meshes-per-output-shard", type=int, default=1000, help="Mesh directories per output shard")
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png", help="Output image format")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality when image-format=jpg")
    parser.add_argument("--status-db", default=None, help="SQLite status DB path")
    parser.add_argument("--events-jsonl", default=None, help="Batch event log path")
    parser.add_argument("--summary-json", default=None, help="Batch summary JSON path")
    parser.add_argument("--keep-partial-on-failure", action="store_true", help="Keep temp output directories for failed jobs")
    parser.add_argument("--views-json", default=None, help="Inline JSON view spec or @path to a JSON file")
    parser.add_argument("--views-file", default=None, help="Path to a JSON file describing the global view set")
    parser.add_argument("--view-preset", default=None, help="Named global view preset")
    parser.add_argument("--view-chunk-sizes", default="24,8,4,2,1", help="Descending view chunk sizes for CUDA OOM fallback")
    parser.add_argument("--default-view-name-template", default=None, help="Template for unnamed views; defaults to legacy elev/azim naming")
    add_render_arguments(
        parser,
        include_input=False,
        include_output=False,
        include_view_ranges=True,
        include_canonical_six_views=True,
        include_multi_view_chunk_size=False,
        include_render_all=True,
        include_benchmark=False,
        include_display=True,
    )
    return parser


@torch.inference_mode()
def main() -> None:
    # Batch is the multi-process, multi-GPU entrypoint; render execution inside each worker remains sequential.
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batch rendering")
    if args.render_all:
        raise ValueError("--render-all is not supported in batch mode")
    if args.display:
        raise ValueError("--display is not supported in batch mode")
    if (args.shard_index is None) != (args.num_shards is None):
        raise ValueError("--shard-index and --num-shards must be provided together")
    if args.num_shards is not None:
        if args.num_shards <= 0:
            raise ValueError("--num-shards must be positive")
        if args.shard_index is None or args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard-index must be in [0, num_shards)")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.max_failures is not None and args.max_failures <= 0:
        raise ValueError("--max-failures must be positive")
    if args.mesh_timeout_sec is not None and args.mesh_timeout_sec <= 0:
        raise ValueError("--mesh-timeout-sec must be positive")
    if args.meshes_per_output_shard <= 0:
        raise ValueError("--meshes-per-output-shard must be positive")
    if not 1 <= args.jpg_quality <= 100:
        raise ValueError("--jpg-quality must be in [1, 100]")

    manifest_path = pathlib.Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    output_root = pathlib.Path(args.output_root).expanduser().resolve()
    _validate_output_root(output_root)
    batch_logger = RunLogger(_batch_log_path(output_root), echo=bool(args.print_progress), prefix="batch")
    batch_logger.reset()

    base_config = config_from_args(args)
    global_view_source = _build_global_view_source(args)
    gpu_list = _parse_gpu_list(args.gpus)
    view_chunk_sizes = _parse_view_chunk_sizes(args.view_chunk_sizes)
    status_db_path = pathlib.Path(args.status_db).expanduser().resolve() if args.status_db else (output_root / "batch_status.sqlite")
    events_path = pathlib.Path(args.events_jsonl).expanduser().resolve() if args.events_jsonl else (output_root / "batch_events.jsonl")
    summary_path = pathlib.Path(args.summary_json).expanduser().resolve() if args.summary_json else (output_root / "batch_summary.json")
    raw_rows = _load_manifest(manifest_path, args.manifest_format)
    selected_entries = _select_candidate_entries(raw_rows, args.shard_index, args.num_shards, args.limit)
    _ensure_unique_mesh_ids(selected_entries)

    conn = _init_status_db(status_db_path)
    batch_started_at = _utc_now()
    batch_start_perf = time.perf_counter()
    _record_event(
        events_path,
        "batch_started",
        manifest_path=str(manifest_path),
        output_root=str(output_root),
        total_rows=len(raw_rows),
        selected_rows=len(selected_entries),
        gpu_list=gpu_list,
    )
    batch_logger.log(
        f"Started batch run: manifest={manifest_path}, output_root={output_root}, "
        f"selected_rows={len(selected_entries)}, gpus={gpu_list}"
    )

    success_count = 0
    failed_count = 0
    skipped_count = 0
    success_metrics: list[dict[str, Any]] = []
    jobs_to_run: list[JobSpec] = []
    abort_requested = False
    total_selected = len(selected_entries)
    processed_count = 0

    def _progress(status: str, detail: str, *, last_ms: float | None = None) -> None:
        elapsed_ms = (time.perf_counter() - batch_start_perf) * 1000.0
        eta_ms = estimate_remaining_ms(processed_count, total_selected, elapsed_ms)
        batch_logger.log(
            f"Progress {processed_count}/{total_selected}: {status} {detail}",
            console="always",
            console_message=(
                f"[Progress] [{processed_count}/{total_selected}] {status}: {detail} "
                f"(last: {format_duration_ms(last_ms)} / ETA: {format_duration_ms(eta_ms)})"
            ),
        )

    batch_logger.log(
        f"Batch progress initialized for {total_selected} selected row(s).",
        console="always",
        console_message="[Info] Batch Start: Initializing",
    )
    batch_logger.log(
        f"Batch total meshes: {total_selected}, workers: {len(gpu_list)}",
        console="always",
        console_message=f"[Info] Batch Items: total={total_selected}, workers={len(gpu_list)}",
    )

    for entry in selected_entries:
        if entry.error_message is not None:
            failed_count += 1
            _record_event(events_path, "job_failed", mesh_id=entry.mesh_id, row_index=entry.row_index, error_type="ValidationError", error_message=entry.error_message)
            if entry.mesh_id is not None:
                _upsert_status(
                    conn,
                    mesh_id=entry.mesh_id,
                    input_path="",
                    status="failed",
                    output_dir=None,
                    started_at=None,
                    finished_at=_utc_now(),
                    error_type="ValidationError",
                    error_message=entry.error_message,
                    worker_id=None,
                    gpu_index=None,
                    increment_attempt=False,
                )
            processed_count += 1
            mesh_label = entry.mesh_id or f"row-{entry.row_index + 1}"
            _progress("failed", f"{mesh_label} validation")
            if args.fail_fast or (args.max_failures is not None and failed_count >= args.max_failures):
                abort_requested = True
                batch_logger.log("Aborting during candidate validation due to fail-fast/max-failures policy.")
                break
            continue
        assert entry.mesh_id is not None
        try:
            row = _normalize_manifest_row(entry, manifest_path.parent.resolve(), output_root)
            row_config = replace(base_config, input=str(row.input_path), output="", display=False, render_all=False)
            row_config = config_with_overrides(row_config, row.overrides)
            views = _resolve_job_views(row, global_view_source, row_config, args.default_view_name_template)
            if not views:
                raise ValueError("resolved view set is empty")
            output_dir = _build_output_dir(output_root, row, args.meshes_per_output_shard).resolve()
            job = JobSpec(
                row_index=row.row_index,
                selected_index=row.selected_index,
                mesh_id=row.mesh_id,
                input_path=str(row.input_path),
                output_dir=str(output_dir),
                temp_output_dir=str(pathlib.Path(str(output_dir) + ".tmp")),
                render_config=replace(row_config, output=str(output_dir)),
                views=views,
                image_format=args.image_format,
                jpg_quality=args.jpg_quality,
                keep_partial_on_failure=bool(args.keep_partial_on_failure),
                overwrite=args.overwrite,
                view_chunk_sizes=view_chunk_sizes,
                source_row=row.source_row,
                tags=row.tags,
            )
            skip_reason = _skip_reason(conn, job) if args.resume else None
            if skip_reason is not None:
                skipped_count += 1
                finished_at = _utc_now()
                _upsert_status(
                    conn,
                    mesh_id=job.mesh_id,
                    input_path=job.input_path,
                    status="skipped",
                    output_dir=job.output_dir,
                    started_at=None,
                    finished_at=finished_at,
                    error_type=None,
                    error_message=skip_reason,
                    worker_id=None,
                    gpu_index=None,
                    increment_attempt=False,
                )
                _record_event(events_path, "job_skipped", mesh_id=job.mesh_id, row_index=job.row_index, output_dir=job.output_dir, reason=skip_reason)
                batch_logger.log(f"Skipped job {job.mesh_id}: {skip_reason}")
                processed_count += 1
                _progress("skipped", job.mesh_id)
                continue
            jobs_to_run.append(job)
        except Exception as exc:
            failed_count += 1
            finished_at = _utc_now()
            error_message = str(exc)
            _upsert_status(
                conn,
                mesh_id=entry.mesh_id,
                input_path="",
                status="failed",
                output_dir=None,
                started_at=None,
                finished_at=finished_at,
                error_type=type(exc).__name__,
                error_message=error_message,
                worker_id=None,
                gpu_index=None,
                increment_attempt=False,
            )
            _record_event(events_path, "job_failed", mesh_id=entry.mesh_id, row_index=entry.row_index, error_type=type(exc).__name__, error_message=error_message)
            processed_count += 1
            _progress("failed", f"{entry.mesh_id} normalization")
            if args.fail_fast or (args.max_failures is not None and failed_count >= args.max_failures):
                abort_requested = True
                batch_logger.log("Aborting during job normalization due to fail-fast/max-failures policy.")
                break

    if not abort_requested and jobs_to_run:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        log_path = str(_batch_log_path(output_root))
        echo = bool(args.print_progress)
        workers = [_start_worker(ctx, slot, gpu_index, result_queue, log_path, echo) for slot, gpu_index in enumerate(gpu_list)]
        pending_jobs = list(jobs_to_run)
        batch_logger.log(f"Dispatching {len(pending_jobs)} job(s) across {len(workers)} worker(s).")
        try:
            while pending_jobs or any(worker.current_job is not None for worker in workers):
                if not abort_requested:
                    for index, worker in enumerate(workers):
                        if worker.current_job is not None or not pending_jobs:
                            continue
                        if not worker.process.is_alive():
                            workers[index] = worker = _restart_worker(ctx, worker, result_queue, log_path, echo)
                        job = pending_jobs.pop(0)
                        started_at = _utc_now()
                        _upsert_status(
                            conn,
                            mesh_id=job.mesh_id,
                            input_path=job.input_path,
                            status="running",
                            output_dir=job.output_dir,
                            started_at=started_at,
                            finished_at=None,
                            error_type=None,
                            error_message=None,
                            worker_id=f"worker-{worker.slot}-gpu{worker.gpu_index}",
                            gpu_index=worker.gpu_index,
                            increment_attempt=True,
                        )
                        _record_event(events_path, "job_started", mesh_id=job.mesh_id, row_index=job.row_index, gpu_index=worker.gpu_index, worker_id=f"worker-{worker.slot}-gpu{worker.gpu_index}", output_dir=job.output_dir)
                        worker.job_queue.put(job)
                        worker.current_job = job
                        worker.current_started_at = started_at
                        worker.current_started_mono = time.monotonic()
                try:
                    result = result_queue.get(timeout=0.2)
                except queue_module.Empty:
                    result = None
                if result is not None:
                    worker = workers[result["worker_slot"]]
                    if worker.current_job is not None and worker.current_job.mesh_id == result["mesh_id"]:
                        worker.current_job = None
                        worker.current_started_at = None
                        worker.current_started_mono = None
                    _finalize_job_result(conn, events_path, result)
                    if result["status"] == "success":
                        success_count += 1
                        success_metrics.append(result)
                        batch_logger.log(
                            f"Job succeeded: mesh_id={result['mesh_id']}, gpu={result['gpu_index']}, "
                            f"duration_ms={_round(result['duration_ms'])}, output={result['output_dir']}"
                        )
                        processed_count += 1
                        _progress("success", result["mesh_id"], last_ms=float(result["duration_ms"]))
                    else:
                        failed_count += 1
                        batch_logger.log(
                            f"Job failed: mesh_id={result['mesh_id']}, gpu={result['gpu_index']}, "
                            f"error={result['error_type']}: {result['error_message']}"
                        )
                        processed_count += 1
                        _progress("failed", result["mesh_id"], last_ms=float(result["duration_ms"]))
                    if result["status"] == "failed" and (args.fail_fast or (args.max_failures is not None and failed_count >= args.max_failures)):
                        abort_requested = True
                for index, worker in enumerate(workers):
                    if worker.current_job is None:
                        if not worker.process.is_alive():
                            workers[index] = _restart_worker(ctx, worker, result_queue, log_path, echo)
                        continue
                    if not worker.process.is_alive():
                        duration_ms = 0.0
                        if worker.current_started_mono is not None:
                            duration_ms = (time.monotonic() - worker.current_started_mono) * 1000.0
                        result = _external_failure_result(worker.current_job, worker, error_type="WorkerExitError", error_message="worker exited unexpectedly", duration_ms=duration_ms)
                        _finalize_job_result(conn, events_path, result)
                        failed_count += 1
                        batch_logger.log(f"Worker exited unexpectedly for mesh_id={result['mesh_id']}; restarting worker.")
                        processed_count += 1
                        _progress("failed", f"{result['mesh_id']} worker-exit", last_ms=float(result["duration_ms"]))
                        workers[index] = _restart_worker(ctx, worker, result_queue, log_path, echo)
                        if args.fail_fast or (args.max_failures is not None and failed_count >= args.max_failures):
                            abort_requested = True
                            break
                        continue
                    if args.mesh_timeout_sec is not None and worker.current_started_mono is not None:
                        elapsed = time.monotonic() - worker.current_started_mono
                        if elapsed > args.mesh_timeout_sec:
                            result = _external_failure_result(
                                worker.current_job,
                                worker,
                                error_type="TimeoutError",
                                error_message=f"mesh job exceeded timeout of {args.mesh_timeout_sec} seconds",
                                duration_ms=elapsed * 1000.0,
                            )
                            worker.process.terminate()
                            worker.process.join(timeout=5)
                            _finalize_job_result(conn, events_path, result)
                            failed_count += 1
                            batch_logger.log(f"Timed out mesh_id={result['mesh_id']} after {args.mesh_timeout_sec} sec; restarting worker.")
                            processed_count += 1
                            _progress("failed", f"{result['mesh_id']} timeout", last_ms=float(result["duration_ms"]))
                            workers[index] = _start_worker(ctx, worker.slot, worker.gpu_index, result_queue, log_path, echo)
                            if args.fail_fast or (args.max_failures is not None and failed_count >= args.max_failures):
                                abort_requested = True
                                break
                if abort_requested:
                    batch_logger.log("Batch abort requested; terminating in-flight workers.")
                    for worker in workers:
                        if worker.current_job is not None:
                            elapsed_ms = 0.0 if worker.current_started_mono is None else (time.monotonic() - worker.current_started_mono) * 1000.0
                            worker.process.terminate()
                            worker.process.join(timeout=5)
                            result = _external_failure_result(
                                worker.current_job,
                                worker,
                                error_type="BatchAbortedError",
                                error_message="batch aborted before job completion",
                                duration_ms=elapsed_ms,
                            )
                            _finalize_job_result(conn, events_path, result)
                            failed_count += 1
                            processed_count += 1
                            _progress("failed", f"{result['mesh_id']} aborted", last_ms=float(result["duration_ms"]))
                        _stop_worker(worker)
                    break
        finally:
            for worker in workers:
                _stop_worker(worker)

    finished_at = _utc_now()
    duration_sec = time.perf_counter() - batch_start_perf
    duration_ms = duration_sec * 1000.0
    avg_session_init_ms = sum(item["session_init_ms"] for item in success_metrics) / len(success_metrics) if success_metrics else 0.0
    avg_prepare_assets_ms = sum(item["prepare_assets_ms"] for item in success_metrics) / len(success_metrics) if success_metrics else 0.0
    avg_prepare_views_ms = sum(item["prepare_views_ms_total"] for item in success_metrics) / len(success_metrics) if success_metrics else 0.0
    avg_render_ms_per_mesh = sum(item["render_ms_total"] for item in success_metrics) / len(success_metrics) if success_metrics else 0.0
    avg_save_ms_per_mesh = sum(item["save_ms_total"] for item in success_metrics) / len(success_metrics) if success_metrics else 0.0
    aggregate_session_init_ms = sum(item["session_init_ms"] for item in success_metrics)
    aggregate_prepare_assets_ms = sum(item["prepare_assets_ms"] for item in success_metrics)
    aggregate_prepare_views_ms = sum(item["prepare_views_ms_total"] for item in success_metrics)
    aggregate_render_ms = sum(item["render_ms_total"] for item in success_metrics)
    aggregate_save_ms = sum(item["save_ms_total"] for item in success_metrics)
    total_views = sum(item["view_count"] for item in success_metrics)
    avg_render_ms_per_view = sum(item["render_ms_total"] for item in success_metrics) / total_views if total_views else 0.0
    avg_views_per_mesh = total_views / len(success_metrics) if success_metrics else 0.0
    summary = {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "started_at": batch_started_at,
        "finished_at": finished_at,
        "duration_sec": _round(duration_sec),
        "total_rows": len(raw_rows),
        "selected_rows": len(selected_entries),
        "success_count": success_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "avg_session_init_ms": _round(avg_session_init_ms),
        "avg_prepare_assets_ms": _round(avg_prepare_assets_ms),
        "avg_prepare_views_ms_per_mesh": _round(avg_prepare_views_ms),
        "avg_render_ms_per_mesh": _round(avg_render_ms_per_mesh),
        "avg_save_ms_per_mesh": _round(avg_save_ms_per_mesh),
        "avg_render_ms_per_view": _round(avg_render_ms_per_view),
        "avg_views_per_mesh": _round(avg_views_per_mesh),
        "aggregate_session_init_ms": _round(aggregate_session_init_ms),
        "aggregate_prepare_assets_ms": _round(aggregate_prepare_assets_ms),
        "aggregate_prepare_views_ms": _round(aggregate_prepare_views_ms),
        "aggregate_render_ms": _round(aggregate_render_ms),
        "aggregate_save_ms": _round(aggregate_save_ms),
        "gpu_list": gpu_list,
    }
    _write_json(summary_path, summary)
    _record_event(
        events_path,
        "batch_finished",
        manifest_path=str(manifest_path),
        output_root=str(output_root),
        duration_sec=_round(duration_sec),
        success_count=success_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
    )
    batch_logger.log(
        f"Finished batch run: success={success_count}, failed={failed_count}, skipped={skipped_count}, "
        f"duration_sec={_round(duration_sec)}, summary={summary_path}"
    )
    batch_logger.log(
        f"Batch timing summary: duration_ms={_round(duration_ms)}",
        console="always",
        console_message=_format_batch_timing_console(duration_ms, success_metrics),
    )
    batch_logger.log(
        f"Batch progress completed: {processed_count}/{total_selected}",
        console="always",
        console_message=format_path_notice("Info", "Done. Log file:", batch_logger.path),
    )
    conn.close()


if __name__ == "__main__":
    main()
