import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np


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
    render_mode: str
    wireframe_color: np.ndarray
    wireframe_opacity: float
    wireframe_thickness_px: float
    normalize_depth: bool
    render_all: bool
    canonical_six_views: bool
    multi_view_chunk_size: int
    geometry_preprocess_device: str
    geometry_cuda_threshold_faces: int
    geometry_cuda_threshold_vertices: int
    benchmark_requested: bool
    benchmark_runs: int
    benchmark_warmup_runs: int


def parse_background(value: str):
    if value.lower() == "transparent":
        return None, True
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (3, 4):
        raise ValueError("--background must be 'transparent' or 'r,g,b[,a]' in 0-1 range")
    rgba = np.array([float(part) for part in parts], dtype=np.float32)
    if len(rgba) == 3:
        rgba = np.concatenate([rgba, np.ones(1, dtype=np.float32)], axis=0)
    return np.clip(rgba, 0.0, 1.0), False


def parse_rgb(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected r,g,b in 0-1 range")
    rgb = np.array([float(part) for part in parts], dtype=np.float32)
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
    if bool(getattr(args, "canonical_six_views", False)) and any(
        value is not None for value in (elev_start, elev_end, elev_step, azim_start, azim_end, azim_step)
    ):
        raise ValueError("--canonical-six-views cannot be combined with explicit multi-view range arguments")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a GLB/GLTF mesh with nvdiffrast on CUDA.")
    parser.add_argument("input", help="Path to input .glb or .gltf")
    parser.add_argument("--output", default="outputs/render.png", help="Output image path")
    parser.add_argument("--resolution", type=int, default=2048, help="Square output resolution")
    parser.add_argument("--elev", type=float, default=0.0, help="Camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=0.0, help="Camera azimuth in degrees")
    parser.add_argument("--elev-start", type=float, default=None, help="Inclusive multi-view elevation start in degrees")
    parser.add_argument("--elev-end", type=float, default=None, help="Inclusive multi-view elevation end in degrees")
    parser.add_argument("--elev-step", type=float, default=None, help="Multi-view elevation step in degrees")
    parser.add_argument("--azim-start", type=float, default=None, help="Inclusive multi-view azimuth start in degrees")
    parser.add_argument("--azim-end", type=float, default=None, help="Inclusive multi-view azimuth end in degrees")
    parser.add_argument("--azim-step", type=float, default=None, help="Multi-view azimuth step in degrees")
    parser.add_argument("--fov", type=float, default=45.0, help="Vertical field of view in degrees")
    parser.add_argument("--distance", type=float, default=None, help="Optional absolute camera distance override")
    parser.add_argument("--distance-scale", type=float, default=1.15, help="Automatic camera distance multiplier")
    parser.add_argument("--env-map", default="", help="Optional HDR/EXR environment map path")
    parser.add_argument("--env-usage", choices=["light", "background", "both"], default="light", help="Use the environment map for lighting only, background only, or both")
    parser.add_argument("--env-light-intensity", type=float, default=0.3, help="Environment lighting multiplier")
    parser.add_argument("--env-background-intensity", type=float, default=1.0, help="Environment background multiplier")
    parser.add_argument("--env-diffuse-samples", type=int, default=16, help="Cosine-weighted env diffuse sample count")
    parser.add_argument("--background", default="transparent", help="transparent or r,g,b[,a] in 0-1 range")
    parser.add_argument("--light-intensity", type=float, default=1.1, help="Directional light multiplier")
    parser.add_argument("--exposure", type=float, default=1.0, help="Linear exposure before tone mapping")
    parser.add_argument("--tonemap", choices=["aces", "reinhard", "none"], default="reinhard", help="Tone mapping operator")
    parser.add_argument("--cull-mode", choices=["auto", "off", "force"], default="auto", help="Backface handling. auto: cull iff material.double_sided is false; off: render both winding buckets; force: front faces only.")
    parser.add_argument(
        "--render-mode",
        choices=[
            "beauty",
            "albedo",
            "normal_world",
            "normal_view",
            "face_normal",
            "depth_ndc",
            "depth_linear",
            "mask",
            "triangle_id",
            "uv",
            "roughness",
            "metallic",
            "ao",
            "emissive",
            "wireframe",
            "beauty_plus_wireframe",
        ],
        default="beauty",
        help="Named render mode entrypoint.",
    )
    parser.add_argument("--wireframe-color", default="0.2,1.0,0.25", help="Wireframe overlay color as r,g,b in 0-1 range")
    parser.add_argument("--wireframe-opacity", type=float, default=1.0, help="Wireframe overlay opacity multiplier")
    parser.add_argument("--wireframe-thickness-px", type=float, default=0.5, help="Wireframe thickness in pixels")
    parser.add_argument("--normalize-depth", action="store_true", help="Normalize depth outputs across visible pixels for visualization")
    parser.add_argument("--render-all", action="store_true", help="Render every supported mode into a mode-named output directory")
    parser.add_argument("--canonical-six-views", action="store_true", help="Render front, back, left, right, top, and bottom views in one multi-view run")
    parser.add_argument("--multi-view-chunk-size", type=int, default=4, help="Maximum number of multi-view jobs to run concurrently per chunk")
    parser.add_argument(
        "--geometry-preprocess-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Where to compute face normals and tangents during mesh loading. auto uses CUDA only for large meshes.",
    )
    parser.add_argument(
        "--geometry-cuda-threshold-faces",
        type=int,
        default=100000,
        help="In auto mode, use CUDA preprocessing when a mesh has at least this many faces.",
    )
    parser.add_argument(
        "--geometry-cuda-threshold-vertices",
        type=int,
        default=100000,
        help="In auto mode, use CUDA preprocessing when a mesh has at least this many vertices.",
    )
    parser.add_argument("--benchmark-runs", type=int, default=None, help="Enable render-all benchmarking and set timed runs per mode")
    parser.add_argument("--benchmark-warmup-runs", type=int, default=None, help="Enable render-all benchmarking and set untimed warmup runs per mode")
    parser.add_argument("--no-antialias", action="store_true", help="Disable edge antialiasing")
    parser.add_argument("--display", action="store_true", help="Display the result in an OpenGL window")
    return parser


def config_from_args(args: argparse.Namespace) -> RenderConfig:
    background_rgba, background_transparent = parse_background(args.background)
    elev_start, elev_end, elev_step = _parse_range_triplet(args, "elev")
    azim_start, azim_end, azim_step = _parse_range_triplet(args, "azim")
    _validate_multi_view_args(args, elev_start, elev_end, elev_step, azim_start, azim_end, azim_step)
    benchmark_requested, benchmark_runs, benchmark_warmup_runs = _parse_benchmark_args(args)
    return RenderConfig(
        input=args.input,
        output=args.output,
        resolution=args.resolution,
        elev=args.elev,
        azim=args.azim,
        elev_start=elev_start,
        elev_end=elev_end,
        elev_step=elev_step,
        azim_start=azim_start,
        azim_end=azim_end,
        azim_step=azim_step,
        fov=args.fov,
        distance=args.distance,
        distance_scale=args.distance_scale,
        env_map=args.env_map,
        env_usage=args.env_usage,
        env_light_intensity=args.env_light_intensity,
        env_background_intensity=args.env_background_intensity,
        env_diffuse_samples=args.env_diffuse_samples,
        background=args.background,
        background_rgba=background_rgba,
        background_transparent=background_transparent,
        light_intensity=args.light_intensity,
        exposure=args.exposure,
        tonemap=args.tonemap,
        cull_mode=args.cull_mode,
        antialias=not args.no_antialias,
        display=args.display,
        render_mode=getattr(args, "render_mode", "beauty"),
        wireframe_color=parse_rgb(getattr(args, "wireframe_color", "0.2,1.0,0.25")),
        wireframe_opacity=float(np.clip(getattr(args, "wireframe_opacity", 1.0), 0.0, 1.0)),
        wireframe_thickness_px=max(float(getattr(args, "wireframe_thickness_px", 0.5)), 0.0),
        normalize_depth=bool(getattr(args, "normalize_depth", False)),
        render_all=bool(getattr(args, "render_all", False)),
        canonical_six_views=bool(getattr(args, "canonical_six_views", False)),
        multi_view_chunk_size=max(int(getattr(args, "multi_view_chunk_size", 4)), 1),
        geometry_preprocess_device=str(getattr(args, "geometry_preprocess_device", "auto")),
        geometry_cuda_threshold_faces=max(int(getattr(args, "geometry_cuda_threshold_faces", 100000)), 0),
        geometry_cuda_threshold_vertices=max(int(getattr(args, "geometry_cuda_threshold_vertices", 100000)), 0),
        benchmark_requested=benchmark_requested,
        benchmark_runs=benchmark_runs,
        benchmark_warmup_runs=benchmark_warmup_runs,
    )
