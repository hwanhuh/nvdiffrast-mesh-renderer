import argparse
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np

RENDER_MODE_CHOICES = (
    "beauty",
    "albedo",
    "normal_world",
    "normal_view",
    "face_normal",
    "normal_ogl",
    "depth_ndc",
    "depth_linear",
    "depth_ogl",
    "position_ogl",
    "confidence_ogl",
    "mask",
    "triangle_id",
    "uv",
    "roughness",
    "metallic",
    "ao",
    "emissive",
    "wireframe",
    "beauty_plus_wireframe",
)
ENV_USAGE_CHOICES = ("light", "background", "both")
TONEMAP_CHOICES = ("aces", "reinhard", "none")
CULL_MODE_CHOICES = ("auto", "off", "force")
CAMERA_CHOICES = ("perspective", "orthographic")
LEGACY_IGNORED_OVERRIDE_KEYS = frozenset(
    {
        "geometry_preprocess_device",
        "geometry_cuda_threshold_faces",
        "geometry_cuda_threshold_vertices",
    }
)

BATCH_OVERRIDE_KEYS = frozenset(
    {
        "resolution",
        "render_mode",
        "png_compression",
        "camera",
        "fov",
        "distance",
        "distance_scale",
        "env_map",
        "env_usage",
        "env_light_intensity",
        "env_background_intensity",
        "env_diffuse_samples",
        "background",
        "light_intensity",
        "exposure",
        "tonemap",
        "cull_mode",
        "normalize_depth",
        "wireframe_color",
        "wireframe_opacity",
        "wireframe_thickness_px",
        "double_sided_depth_peels",
        "texture_map_max_size",
    }
)


@dataclass
class RenderConfig:
    input: str
    output: str
    resolution: int
    elev: float
    azim: float
    elev_start: Optional[float]
    elev_end: Optional[float]
    elev_step: Optional[float]
    azim_start: Optional[float]
    azim_end: Optional[float]
    azim_step: Optional[float]
    camera: str
    fov: float
    distance: Optional[float]
    distance_scale: float
    env_map: str
    env_usage: str
    env_light_intensity: float
    env_background_intensity: float
    env_diffuse_samples: int
    background: str
    background_rgba: Optional[np.ndarray]
    background_transparent: bool
    light_intensity: float
    exposure: float
    tonemap: str
    cull_mode: str
    antialias: bool
    display: bool
    print_progress: bool
    render_mode: str
    wireframe_color: np.ndarray
    wireframe_opacity: float
    wireframe_thickness_px: float
    double_sided_depth_peels: int
    normalize_depth: bool
    png_compression: int
    render_all: bool
    render_all_batch_size: int
    canonical_six_views: bool
    canonical_mv_conditions: bool
    canonical_render_cond: bool
    multi_view_chunk_size: int
    texture_map_max_size: int
    benchmark_requested: bool
    benchmark_runs: int
    benchmark_warmup_runs: int


def _parse_float_sequence(value: Any, *, expected_lengths: Sequence[int], transparent_ok: bool = False, label: str) -> np.ndarray | None:
    if isinstance(value, str):
        if transparent_ok and value.lower() == "transparent":
            return None
        parts = [part.strip() for part in value.split(",")]
        if len(parts) not in expected_lengths:
            lengths = ", ".join(str(length) for length in expected_lengths)
            raise ValueError(f"{label} must provide {lengths} value(s)")
        values = np.array([float(part) for part in parts], dtype=np.float32)
    else:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(values) not in expected_lengths:
            lengths = ", ".join(str(length) for length in expected_lengths)
            raise ValueError(f"{label} must provide {lengths} value(s)")
    return values


def parse_background(value: str | Sequence[float]):
    rgba = _parse_float_sequence(
        value,
        expected_lengths=(3, 4),
        transparent_ok=True,
        label="--background",
    )
    if rgba is None:
        return None, True
    if len(rgba) == 3:
        rgba = np.concatenate([rgba, np.ones(1, dtype=np.float32)], axis=0)
    return np.clip(rgba, 0.0, 1.0), False


def parse_rgb(value: str | Sequence[float]) -> np.ndarray:
    rgb = _parse_float_sequence(value, expected_lengths=(3,), label="RGB value")
    assert rgb is not None
    return np.clip(rgb, 0.0, 1.0)


def _parse_range_triplet(args: argparse.Namespace, prefix: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    start = getattr(args, f"{prefix}_start", None)
    end = getattr(args, f"{prefix}_end", None)
    step = getattr(args, f"{prefix}_step", None)
    provided = [value is not None for value in (start, end, step)]
    if any(provided) and not all(provided):
        raise ValueError(f"--{prefix}-start, --{prefix}-end, and --{prefix}-step must be provided together")
    if step == 0.0:
        raise ValueError(f"--{prefix}-step must be non-zero")
    return start, end, step


def _parse_benchmark_args(args: argparse.Namespace) -> tuple[bool, int, int]:
    benchmark_runs_arg = getattr(args, "benchmark_runs", None)
    benchmark_warmup_runs_arg = getattr(args, "benchmark_warmup_runs", None)
    benchmark_requested = benchmark_runs_arg is not None or benchmark_warmup_runs_arg is not None
    if not benchmark_requested:
        return False, 0, 0
    runs = 1 if benchmark_runs_arg is None else max(int(benchmark_runs_arg), 1)
    warmup_runs = 0 if benchmark_warmup_runs_arg is None else max(int(benchmark_warmup_runs_arg), 0)
    return True, runs, warmup_runs


def _validate_multi_view_args(
    args: argparse.Namespace,
    elev_start: Optional[float],
    elev_end: Optional[float],
    elev_step: Optional[float],
    azim_start: Optional[float],
    azim_end: Optional[float],
    azim_step: Optional[float],
) -> None:
    canonical_flags = (
        bool(getattr(args, "canonical_six_views", False)),
        bool(getattr(args, "canonical_mv_conditions", False)),
        bool(getattr(args, "canonical_render_cond", False)),
    )
    if sum(canonical_flags) > 1:
        raise ValueError("--canonical-six-views, --canonical-mv-conditions, and --canonical-render-cond are mutually exclusive")
    if any(canonical_flags) and any(
        value is not None for value in (elev_start, elev_end, elev_step, azim_start, azim_end, azim_step)
    ):
        raise ValueError("--canonical-six-views/--canonical-mv-conditions/--canonical-render-cond cannot be combined with explicit multi-view range arguments")


def add_render_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_input: bool = True,
    include_output: bool = True,
    include_view_ranges: bool = True,
    include_canonical_six_views: bool = True,
    include_canonical_render_cond: bool = True,
    include_multi_view_chunk_size: bool = True,
    include_render_all: bool = True,
    include_benchmark: bool = True,
    include_display: bool = True,
) -> argparse.ArgumentParser:
    if include_input:
        parser.add_argument("input", help="Path to input .glb or .gltf")
    if include_output:
        parser.add_argument("--output", default="outputs/render.png", help="Output image path")
    parser.add_argument("--resolution", type=int, default=2048, help="Square output resolution")
    parser.add_argument("--elev", type=float, default=0.0, help="Camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=0.0, help="Camera azimuth in degrees")
    if include_view_ranges:
        parser.add_argument("--elev-start", type=float, default=None, help="Inclusive multi-view elevation start in degrees")
        parser.add_argument("--elev-end", type=float, default=None, help="Inclusive multi-view elevation end in degrees")
        parser.add_argument("--elev-step", type=float, default=None, help="Multi-view elevation step in degrees")
        parser.add_argument("--azim-start", type=float, default=None, help="Inclusive multi-view azimuth start in degrees")
        parser.add_argument("--azim-end", type=float, default=None, help="Inclusive multi-view azimuth end in degrees")
        parser.add_argument("--azim-step", type=float, default=None, help="Multi-view azimuth step in degrees")
    parser.add_argument("--camera", choices=CAMERA_CHOICES, default="perspective", help="Projection type for rendering camera")
    parser.add_argument("--fov", type=float, default=45.0, help="Vertical field of view in degrees")
    parser.add_argument("--distance", type=float, default=None, help="Optional absolute camera distance override")
    parser.add_argument("--distance-scale", type=float, default=1.15, help="Automatic camera distance multiplier")
    parser.add_argument("--env-map", default="", help="Optional HDR/EXR environment map path")
    parser.add_argument(
        "--env-usage",
        choices=ENV_USAGE_CHOICES,
        default="light",
        help="Use the environment map for lighting only, background only, or both",
    )
    parser.add_argument("--env-light-intensity", type=float, default=0.3, help="Environment lighting multiplier")
    parser.add_argument("--env-background-intensity", type=float, default=1.0, help="Environment background multiplier")
    parser.add_argument("--env-diffuse-samples", type=int, default=16, help="Cosine-weighted env diffuse sample count")
    parser.add_argument("--background", default="transparent", help="transparent or r,g,b[,a] in 0-1 range")
    parser.add_argument("--light-intensity", type=float, default=1.35, help="Directional light multiplier")
    parser.add_argument("--exposure", type=float, default=1.2, help="Linear exposure before tone mapping")
    parser.add_argument("--tonemap", choices=TONEMAP_CHOICES, default="reinhard", help="Tone mapping operator")
    parser.add_argument(
        "--cull-mode",
        choices=CULL_MODE_CHOICES,
        default="auto",
        help="Backface handling. auto: cull iff material.double_sided is false; off: render both winding buckets; force: front faces only.",
    )
    parser.add_argument("--render-mode", choices=RENDER_MODE_CHOICES, default="beauty", help="Named render mode entrypoint.")
    parser.add_argument("--wireframe-color", default="0.2,1.0,0.25", help="Wireframe overlay color as r,g,b in 0-1 range")
    parser.add_argument("--wireframe-opacity", type=float, default=1.0, help="Wireframe overlay opacity multiplier")
    parser.add_argument("--wireframe-thickness-px", type=float, default=0.5, help="Wireframe thickness in pixels")
    parser.add_argument(
        "--double-sided-depth-peels",
        type=int,
        default=4,
        help="Maximum depth layers to peel per winding bucket for double-sided meshes. Use 1 to disable depth peeling.",
    )
    parser.add_argument("--normalize-depth", action="store_true", help="Normalize depth outputs across visible pixels for visualization")
    parser.add_argument(
        "--png-compression",
        type=int,
        default=1,
        help="PNG compression level in [0, 9]. Lower values trade larger files for faster writes.",
    )
    if include_render_all:
        parser.add_argument("--render-all", action="store_true", help="Render every supported mode into a mode-named output directory")
        parser.add_argument(
            "--render-all-batch-size",
            type=int,
            default=4,
            help="Maximum number of render-all modes to process per shared geometry pass before saving outputs",
        )
    if include_canonical_six_views:
        parser.add_argument("--canonical-six-views", action="store_true", help="Render front, back, left, right, top, and bottom views in one multi-view run")
        parser.add_argument(
            "--canonical-mv-conditions",
            action="store_true",
            help="Render canonical six views for both normal_ogl and position_ogl conditions, producing 12 outputs.",
        )
    if include_canonical_render_cond:
        parser.add_argument(
            "--canonical-render-cond",
            action="store_true",
            help="Render a deterministic render_cond-style multi-view set with per-view FOV overrides and filtered top/bottom extremes.",
        )
    if include_multi_view_chunk_size:
        parser.add_argument("--multi-view-chunk-size", type=int, default=4, help="Maximum number of views to stage per sequential multi-view chunk")
    parser.add_argument(
        "--texture-map-max-size",
        type=int,
        default=2048,
        help="If > 0, downscale mesh material texture maps so their longest side does not exceed this size before GPU upload.",
    )
    if include_benchmark:
        parser.add_argument("--benchmark-runs", type=int, default=None, help="Enable render-all benchmarking and set timed runs per mode")
        parser.add_argument("--benchmark-warmup-runs", type=int, default=None, help="Enable render-all benchmarking and set untimed warmup runs per mode")
    parser.add_argument("--no-antialias", action="store_true", help="Disable edge antialiasing")
    if include_display:
        parser.add_argument("--display", action="store_true", help="Display the result in an OpenGL window")
    parser.add_argument("--print", dest="print_progress", action="store_true", help="Echo raw progress/report logs to stdout in addition to writing a .log file")
    return parser


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a GLB/GLTF mesh with nvdiffrast on CUDA.")
    return add_render_arguments(parser)


def _validate_choice(name: str, value: str, choices: Sequence[str]) -> str:
    if value not in choices:
        joined = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {joined}")
    return value


def config_from_args(args: argparse.Namespace) -> RenderConfig:
    background_value = getattr(args, "background", "transparent")
    background_rgba, background_transparent = parse_background(background_value)
    elev_start, elev_end, elev_step = _parse_range_triplet(args, "elev")
    azim_start, azim_end, azim_step = _parse_range_triplet(args, "azim")
    _validate_multi_view_args(args, elev_start, elev_end, elev_step, azim_start, azim_end, azim_step)
    benchmark_requested, benchmark_runs, benchmark_warmup_runs = _parse_benchmark_args(args)
    return RenderConfig(
        input=str(getattr(args, "input", "")),
        output=str(getattr(args, "output", "outputs/render.png")),
        resolution=int(getattr(args, "resolution", 2048)),
        elev=float(getattr(args, "elev", 0.0)),
        azim=float(getattr(args, "azim", 0.0)),
        elev_start=elev_start,
        elev_end=elev_end,
        elev_step=elev_step,
        azim_start=azim_start,
        azim_end=azim_end,
        azim_step=azim_step,
        camera=str(getattr(args, "camera", "perspective")),
        fov=float(getattr(args, "fov", 45.0)),
        distance=getattr(args, "distance", None),
        distance_scale=float(getattr(args, "distance_scale", 1.15)),
        env_map=str(getattr(args, "env_map", "")),
        env_usage=str(getattr(args, "env_usage", "light")),
        env_light_intensity=float(getattr(args, "env_light_intensity", 0.3)),
        env_background_intensity=float(getattr(args, "env_background_intensity", 1.0)),
        env_diffuse_samples=int(getattr(args, "env_diffuse_samples", 16)),
        background=str(background_value) if isinstance(background_value, str) else ",".join(str(v) for v in np.asarray(background_value).reshape(-1)),
        background_rgba=background_rgba,
        background_transparent=background_transparent,
        light_intensity=float(getattr(args, "light_intensity", 1.35)),
        exposure=float(getattr(args, "exposure", 1.2)),
        tonemap=str(getattr(args, "tonemap", "reinhard")),
        cull_mode=str(getattr(args, "cull_mode", "auto")),
        antialias=not bool(getattr(args, "no_antialias", False)),
        display=bool(getattr(args, "display", False)),
        print_progress=bool(getattr(args, "print_progress", False)),
        render_mode=str(getattr(args, "render_mode", "beauty")),
        wireframe_color=parse_rgb(getattr(args, "wireframe_color", "0.2,1.0,0.25")),
        wireframe_opacity=float(np.clip(getattr(args, "wireframe_opacity", 1.0), 0.0, 1.0)),
        wireframe_thickness_px=max(float(getattr(args, "wireframe_thickness_px", 0.5)), 0.0),
        double_sided_depth_peels=max(int(getattr(args, "double_sided_depth_peels", 2)), 1),
        normalize_depth=bool(getattr(args, "normalize_depth", False)),
        png_compression=int(np.clip(int(getattr(args, "png_compression", 1)), 0, 9)),
        render_all=bool(getattr(args, "render_all", False)),
        render_all_batch_size=max(int(getattr(args, "render_all_batch_size", 4)), 1),
        canonical_six_views=bool(getattr(args, "canonical_six_views", False)),
        canonical_mv_conditions=bool(getattr(args, "canonical_mv_conditions", False)),
        canonical_render_cond=bool(getattr(args, "canonical_render_cond", False)),
        multi_view_chunk_size=max(int(getattr(args, "multi_view_chunk_size", 4)), 1),
        texture_map_max_size=max(int(getattr(args, "texture_map_max_size", 0)), 0),
        benchmark_requested=benchmark_requested,
        benchmark_runs=benchmark_runs,
        benchmark_warmup_runs=benchmark_warmup_runs,
    )


def config_with_overrides(base_config: RenderConfig, overrides: dict[str, Any]) -> RenderConfig:
    unknown_keys = sorted(set(overrides) - BATCH_OVERRIDE_KEYS - LEGACY_IGNORED_OVERRIDE_KEYS)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unsupported override key(s): {joined}")
    updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if key == "background":
            background_rgba, background_transparent = parse_background(value)
            if isinstance(value, str):
                background_value = value
            else:
                background_value = ",".join(str(float(part)) for part in np.asarray(value, dtype=np.float32).reshape(-1))
            updates["background"] = background_value
            updates["background_rgba"] = background_rgba
            updates["background_transparent"] = background_transparent
            continue
        if key == "wireframe_color":
            updates[key] = parse_rgb(value)
            continue
        if key == "wireframe_opacity":
            updates[key] = float(np.clip(float(value), 0.0, 1.0))
            continue
        if key == "wireframe_thickness_px":
            updates[key] = max(float(value), 0.0)
            continue
        if key == "render_mode":
            updates[key] = _validate_choice(key, str(value), RENDER_MODE_CHOICES)
            continue
        if key == "camera":
            updates[key] = _validate_choice(key, str(value), CAMERA_CHOICES)
            continue
        if key == "env_usage":
            updates[key] = _validate_choice(key, str(value), ENV_USAGE_CHOICES)
            continue
        if key == "tonemap":
            updates[key] = _validate_choice(key, str(value), TONEMAP_CHOICES)
            continue
        if key == "cull_mode":
            updates[key] = _validate_choice(key, str(value), CULL_MODE_CHOICES)
            continue
        if key in LEGACY_IGNORED_OVERRIDE_KEYS:
            continue
        if key in {"resolution", "env_diffuse_samples", "texture_map_max_size"}:
            updates[key] = int(value)
            continue
        if key == "png_compression":
            updates[key] = int(np.clip(int(value), 0, 9))
            continue
        if key == "double_sided_depth_peels":
            updates[key] = max(int(value), 1)
            continue
        if key == "normalize_depth":
            updates[key] = bool(value)
            continue
        if key in {
            "fov",
            "distance_scale",
            "env_light_intensity",
            "env_background_intensity",
            "light_intensity",
            "exposure",
        }:
            updates[key] = float(value)
            continue
        if key == "distance":
            updates[key] = None if value is None else float(value)
            continue
        if key == "env_map":
            updates[key] = str(value)
            continue
    return replace(base_config, **updates)
