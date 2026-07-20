nvdiffrast_mesh_renderer/vram_utils.py"""Free-VRAM probe and conservative render-memory estimator.

The estimator does not aim for prediction accuracy; it provides a coarse
upper bound used to decide whether to take the streaming bypass path
before any GPU allocation happens. Coefficients are deliberately
conservative so that meshes near the cliff err toward streaming.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


# Coefficients tuned to over-estimate without triggering streaming for tiny
# meshes. Override via env vars when calibrating on a specific GPU/mesh class.
VRAM_BASE_OVERHEAD_BYTES = _env_int("NVDIFFRAST_VRAM_BASE_OVERHEAD", 512 * 1024 * 1024)
VRAM_BYTES_PER_VERTEX = _env_int("NVDIFFRAST_VRAM_BYTES_PER_VERTEX", 96)
VRAM_BYTES_PER_FACE = _env_int("NVDIFFRAST_VRAM_BYTES_PER_FACE", 64)
VRAM_BYTES_PER_PIXEL_PER_LAYER = _env_int("NVDIFFRAST_VRAM_BYTES_PER_PIXEL_PER_LAYER", 48)
VRAM_TEXTURE_BUDGET_BYTES = _env_int("NVDIFFRAST_VRAM_TEXTURE_BUDGET", 256 * 1024 * 1024)
VRAM_ANTIALIAS_MULTIPLIER = _env_float("NVDIFFRAST_VRAM_ANTIALIAS_MULTIPLIER", 1.25)
VRAM_SAFETY_FACTOR = _env_float("NVDIFFRAST_VRAM_SAFETY_FACTOR", 0.75)
VRAM_STREAM_TEXTURE_CAP = _env_int("NVDIFFRAST_VRAM_STREAM_TEXTURE_CAP", 1024)


@dataclass(frozen=True)
class RenderStrategy:
    force_streaming: bool
    recommended_chunk_size: int
    texture_cap_override: int | None
    estimated_bytes: int
    free_bytes: int
    reason: str


def get_free_vram_bytes(device: torch.device | None = None) -> int:
    """Return free VRAM on the given CUDA device, or 0 if unavailable."""
    if not torch.cuda.is_available():
        return 0
    try:
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif device.type != "cuda":
            return 0
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        return 0


def estimate_render_vram_bytes(
    *,
    vertex_count: int,
    face_count: int,
    view_count: int,
    resolution: int,
    render_mode_count: int = 1,
    depth_peels: int = 2,
    antialias: bool = True,
) -> int:
    """Coarse upper bound on peak VRAM during a multi-view render chunk.

    Sums (a) base context overhead, (b) mesh tensors, (c) per-view
    rasterization and post-process buffers scaled by depth peels and
    render-mode count, plus an antialias multiplier.
    """
    vertex_count = max(int(vertex_count), 0)
    face_count = max(int(face_count), 0)
    view_count = max(int(view_count), 1)
    resolution = max(int(resolution), 64)
    render_mode_count = max(int(render_mode_count), 1)
    depth_peels = max(int(depth_peels), 1)

    mesh_bytes = vertex_count * VRAM_BYTES_PER_VERTEX + face_count * VRAM_BYTES_PER_FACE
    pixels = resolution * resolution
    per_view_bytes = pixels * VRAM_BYTES_PER_PIXEL_PER_LAYER * depth_peels * render_mode_count
    if antialias:
        per_view_bytes = int(per_view_bytes * VRAM_ANTIALIAS_MULTIPLIER)
    chunk_bytes = per_view_bytes * view_count
    return int(VRAM_BASE_OVERHEAD_BYTES + mesh_bytes + VRAM_TEXTURE_BUDGET_BYTES + chunk_bytes)


def _largest_chunk_size_that_fits(
    *,
    vertex_count: int,
    face_count: int,
    resolution: int,
    render_mode_count: int,
    depth_peels: int,
    antialias: bool,
    free_bytes: int,
    initial_chunk_size: int,
    safety_factor: float,
) -> int:
    budget = int(free_bytes * safety_factor)
    for candidate in range(max(int(initial_chunk_size), 1), 0, -1):
        estimated = estimate_render_vram_bytes(
            vertex_count=vertex_count,
            face_count=face_count,
            view_count=candidate,
            resolution=resolution,
            render_mode_count=render_mode_count,
            depth_peels=depth_peels,
            antialias=antialias,
        )
        if estimated <= budget:
            return candidate
    return 1


def pick_safe_render_strategy(
    *,
    vertex_count: int,
    face_count: int,
    initial_chunk_size: int,
    view_count: int,
    resolution: int,
    render_mode_count: int = 1,
    depth_peels: int = 2,
    antialias: bool = True,
    current_texture_cap: int = 0,
    device: torch.device | None = None,
    safety_factor: float | None = None,
) -> RenderStrategy:
    """Decide whether to bypass to streaming based on free VRAM.

    Returns the recommended starting chunk size and (optionally) a texture
    cap to apply. force_streaming becomes True when even chunk_size=1
    against the full mesh would exceed the safety budget — the streaming
    path then trades fast-path performance for mesh-by-mesh memory reuse.
    """
    safety = float(VRAM_SAFETY_FACTOR if safety_factor is None else safety_factor)
    free_bytes = get_free_vram_bytes(device)
    initial_chunk_size = max(int(initial_chunk_size), 1)

    estimated_initial = estimate_render_vram_bytes(
        vertex_count=vertex_count,
        face_count=face_count,
        view_count=initial_chunk_size,
        resolution=resolution,
        render_mode_count=render_mode_count,
        depth_peels=depth_peels,
        antialias=antialias,
    )

    if free_bytes <= 0:
        # No reliable signal from the driver; keep the requested settings.
        return RenderStrategy(
            force_streaming=False,
            recommended_chunk_size=initial_chunk_size,
            texture_cap_override=None,
            estimated_bytes=estimated_initial,
            free_bytes=0,
            reason="vram_probe_unavailable",
        )

    budget = int(free_bytes * safety)
    if estimated_initial <= budget:
        return RenderStrategy(
            force_streaming=False,
            recommended_chunk_size=initial_chunk_size,
            texture_cap_override=None,
            estimated_bytes=estimated_initial,
            free_bytes=free_bytes,
            reason="within_budget",
        )

    safe_chunk = _largest_chunk_size_that_fits(
        vertex_count=vertex_count,
        face_count=face_count,
        resolution=resolution,
        render_mode_count=render_mode_count,
        depth_peels=depth_peels,
        antialias=antialias,
        free_bytes=free_bytes,
        initial_chunk_size=initial_chunk_size,
        safety_factor=safety,
    )

    estimated_at_one = estimate_render_vram_bytes(
        vertex_count=vertex_count,
        face_count=face_count,
        view_count=1,
        resolution=resolution,
        render_mode_count=render_mode_count,
        depth_peels=depth_peels,
        antialias=antialias,
    )
    needs_streaming = estimated_at_one > budget

    texture_override: int | None = None
    if needs_streaming:
        if current_texture_cap == 0 or current_texture_cap > VRAM_STREAM_TEXTURE_CAP:
            texture_override = VRAM_STREAM_TEXTURE_CAP

    if needs_streaming:
        reason = "streaming_required"
    elif safe_chunk < initial_chunk_size:
        reason = "chunk_size_reduced"
    else:
        reason = "within_budget"

    return RenderStrategy(
        force_streaming=needs_streaming,
        recommended_chunk_size=max(safe_chunk, 1),
        texture_cap_override=texture_override,
        estimated_bytes=estimated_initial,
        free_bytes=free_bytes,
        reason=reason,
    )


def format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}" if unit != "B" else f"{num_bytes}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"
