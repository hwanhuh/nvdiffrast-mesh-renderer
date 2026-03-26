# nvdiffrast_mesh_renderer

## Installation

This package assumes a CUDA-ready environment where `torch` is already installed.

`nvdiffrast` is a required runtime dependency, but it is intentionally not declared in `pyproject.toml` because this project needs it installed from source with `--no-build-isolation`.

Install `nvdiffrast` first:

```bash
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
python -m pip install /tmp/extensions/nvdiffrast --no-build-isolation
```

Equivalent helper script:

```bash
bash scripts/install_nvdiffrast.sh
```

Install the package from this repository:

```bash
pip install .
```

For development, `pip install -e .` also works.

Installed CLI entrypoints:

- `nvdiffrast-mesh-render`
- `nvdiffrast-mesh-render-all`
- `nvdiffrast-mesh-render-multi-view`

You can also invoke the main CLI as a module:

```bash
python -m nvdiffrast_mesh_renderer --help
```

## Usage

Basic beauty render:

```bash
nvdiffrast-mesh-render example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/render.png \
    --resolution 1024
```

Beauty + wireframe overlay:

```bash
nvdiffrast-mesh-render example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/beauty_wire.png \
    --resolution 1024 \
    --render-mode beauty_plus_wireframe \
    --wireframe-color 0.2,1.0,0.25 \
    --wireframe-opacity 0.8 \
    --wireframe-thickness-px 0.5
```

Depth visualization:

```bash
nvdiffrast-mesh-render example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/depth_linear.png \
    --resolution 1024 \
    --render-mode depth_linear \
    --normalize-depth
```

Render every supported mode in one shot:

```bash
nvdiffrast-mesh-render example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/all_modes.png \
    --resolution 512 \
    --render-all
```

This writes one image per mode under `outputs/all_modes/`.

By default, `render-all` just writes the per-mode images. Benchmark timing and `render_all_report.txt` are generated only when you explicitly pass `--benchmark-runs` and/or `--benchmark-warmup-runs`.

Convenience wrapper for the same workflow:

```bash
nvdiffrast-mesh-render-all example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/all_modes \
    --resolution 512
```

If `--output` has an extension, `--render-all` uses the stem as a directory. If it has no extension, that path is treated as the output directory directly.

Chunked multi-view render:

```bash
nvdiffrast-mesh-render-multi-view example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/multiview \
    --resolution 1024 \
    --render-mode beauty \
    --azim-start 0 \
    --azim-end 330 \
    --azim-step 30 \
    --elev-start -15 \
    --elev-end 15 \
    --elev-step 15 \
    --multi-view-chunk-size 4
```

This writes one image per view under `outputs/multiview/` plus a `multiview_report.txt` manifest. If `--output` has an extension, the stem is used as the output directory. If it has no extension, that path is treated as the output directory directly.

Canonical six-view render:

```bash
nvdiffrast-mesh-render-multi-view example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/canonical_views \
    --resolution 1024 \
    --render-mode beauty \
    --canonical-six-views
```

This writes `front`, `back`, `left`, `right`, `top`, and `bottom` view images into the output directory. In this mode the renderer first tries one chunk of 6 views, and if that hits CUDA OOM it retries automatically with chunk size 2.

## Render Modes

Supported `--render-mode` values:

- `beauty`
- `albedo`
- `normal_world`
- `normal_view`
- `face_normal`
- `depth_ndc`
- `depth_linear`
- `mask`
- `triangle_id`
- `uv`
- `roughness`
- `metallic`
- `ao`
- `emissive`
- `wireframe`
- `beauty_plus_wireframe`

Notes:

- normal outputs are encoded from `[-1, 1]` to `[0, 1]`
- `depth_ndc` comes from `rast[..., 2]`
- `depth_linear` comes from `-view_pos.z`
- `--normalize-depth` is intended for visualization/export, not raw metric depth output
- `wireframe` uses analytic barycentric coverage, not ad hoc edge drawing and not `dr.antialias()` for interior edges

## Common Flags

Input / output:

- `input`
- `--output`
- `--resolution`
- `--render-all`
- `--canonical-six-views`
- `--multi-view-chunk-size`

Camera:

- `--elev`
- `--azim`
- `--elev-start`
- `--elev-end`
- `--elev-step`
- `--azim-start`
- `--azim-end`
- `--azim-step`
- `--fov`
- `--distance`
- `--distance-scale`

Environment / background:

- `--env-map`
- `--env-usage {light,background,both}`
- `--env-light-intensity`
- `--env-background-intensity`
- `--env-diffuse-samples`
- `--background`

Shading / visualization:

- `--render-mode`
- `--wireframe-color`
- `--wireframe-opacity`
- `--wireframe-thickness-px`
- `--normalize-depth`

Quality / visibility:

- `--cull-mode {auto,off,force}`
- `--no-antialias`
- `--display`

Large-scale preprocessing:

- `--geometry-preprocess-device {auto,cpu,cuda}`
- `--geometry-cuda-threshold-faces`
- `--geometry-cuda-threshold-vertices`

## Project Overview

This is a CUDA + `nvdiffrast` based GLB/GLTF renderer. The current implementation is organized around a shared geometry pass plus a multi-mode render dispatcher, while preserving the earlier projected-winding front/back classification and premultiplied-alpha compositing fixes.
The installable package is `nvdiffrast_mesh_renderer`, exposed via `nvdiffrast-mesh-render`, `nvdiffrast-mesh-render-all`, `nvdiffrast-mesh-render-multi-view`, and `python -m nvdiffrast_mesh_renderer`. The repository-level `render_glb.py`, `render_all_modes.py`, and `render_multi_view.py` remain as thin wrappers around the package.

## Current Features

- Load `.glb` and `.gltf` assets
- Handle PBR and diffuse materials
- Support base color / metallic-roughness / normal / occlusion / emissive textures
- Support vertex colors
- Automatic orbit camera framing
- Directional lights plus environment lighting / environment background
- Transparent background output
- Premultiplied alpha throughout internal compositing
- Projected-winding based front/back classification
- `auto`, `off`, and `force` cull modes
- Edge antialiasing on/off
- Multiple geometry-buffer and material visualization modes
- Analytic barycentric wireframe rendering
- `beauty_plus_wireframe` overlay composition
- Single render, render-all, and multi-view CLI flows
- Chunked multi-view rendering across azimuth/elevation grids with per-chunk parallel execution
- Canonical six-view rendering with `front/back/left/right/top/bottom` presets and `6 -> 2` CUDA OOM fallback
- Faster scene loading by iterating scene graph instances directly instead of `scene.dump(concatenate=False)`
- Automatic CUDA geometry preprocessing for large meshes to speed up face-normal and tangent generation

## Repository Layout

```text
nvdiffrast_renderer/
├── scripts/
│   └── install_nvdiffrast.sh       # Exact nvdiffrast source install helper
├── render_glb.py                    # Main CLI entrypoint
├── render_all_modes.py              # Convenience wrapper for --render-all
├── render_multi_view.py             # Convenience wrapper for multi-view grids
├── nvdiffrast_mesh_renderer/
│   ├── config.py                    # CLI parsing and RenderConfig
│   ├── renderer.py                  # SceneRenderer and render-mode dispatch
│   ├── scene_builder.py             # Mesh, camera, and light construction
│   ├── geometry_pass.py             # Geometry pass, front/back split, wireframe topology
│   ├── beauty.py                    # Multi-mode rendering and beauty shading
│   ├── compositor.py                # Mesh-layer compositing and overlay composition
│   ├── environment.py               # Environment maps and background rendering
│   ├── ibl.py                       # Image-based lighting helpers
│   ├── textures.py                  # Texture loading, caching, sampling
│   ├── materials.py                 # Material extraction
│   ├── geometry_utils.py            # Tangents, normals, bounds helpers
│   ├── math_utils.py                # Shared math helpers
│   ├── postprocess.py               # Tone mapping and image output
│   └── types.py                     # Shared dataclass contracts
├── example_meshes/                  # Sample GLBs used for validation
├── envmaps/                         # Sample HDR/EXR environment maps
└── outputs/                         # Render outputs and validation outputs
```

## Rendering Pipeline

1. `SceneBuilder`
   - loads GLB/GLTF assets
   - extracts materials
   - computes tangents and face normals
   - builds a bounds-based camera
   - builds directional lights
2. `GeometryPassRenderer.render_geometry_pass(...)`
   - splits triangles into front/back buckets using clip-space projected winding
   - applies single-sided culling according to `--cull-mode`
   - rasterizes and interpolates the shared geometry attributes
   - samples material scalar/color channels needed by downstream modes
3. `RenderModeRenderer.render_mode(...)`
   - evaluates `beauty` shading or visualization outputs from the shared geometry buffer
   - keeps premultiplied-alpha conventions intact
   - applies analytic barycentric wireframe coverage where requested
4. `LayerCompositor`
   - composites mesh layers in depth-sorted order
   - merges front/back passes for double-sided meshes
   - composites wireframe over beauty in premultiplied form
5. `ImagePostprocessor`
   - unpremultiplies
   - applies exposure and tone mapping where appropriate
   - converts selected outputs to display-ready PNG/JPG images

## Geometry Pass Contract

`GeometryPassRenderer.render_geometry_pass(...)` currently produces:

- `rast`
- `rast_db`
- `valid`
- `triangle_id`
- `uvw`
- `barycentric`
- `world_pos`
- `view_pos`
- `normal_world`
- `normal_view`
- `face_normal_world`
- `face_normal_view`
- `uv`
- `uv_da`
- `tangent`
- `vertex_color`
- `base_rgba`
- `emissive`
- `ao`
- `roughness`
- `metallic`
- `clip_pos`
- `tri`
- `side`

There is also a dedicated wireframe raster path that expands triangles into face-varying topology and interpolates per-corner barycentrics `(1,0,0)`, `(0,1,0)`, `(0,0,1)`.

## Front/Back Classification

This renderer does not use `dot(normal, view)` as the primary front/back signal. It uses clip-space projected winding.

That means:

- triangles are transformed into clip space
- signed area is computed in NDC XY
- the sign of that area determines front vs back

That signal remains authoritative. Normal flipping, single-sided culling, and double-sided merging are downstream consequences of that classification.

## Alpha and Compositing Rules

Internal rendering and compositing use premultiplied alpha.

Current rules:

- shaded outputs are kept internally as `rgb * alpha`
- edge antialiasing is applied in premultiplied form for the base geometry pass
- inter-mesh compositing is performed in premultiplied alpha
- double-sided `BLEND` materials merge front/back in premultiplied alpha
- wireframe overlays are also composited as premultiplied overlays

Right before saving, the renderer unpremultiplies and converts each mode into a display-oriented export image.

## Wireframe Overlay

Wireframe rendering is implemented as a dedicated pass on expanded face-varying topology.

- per-corner barycentrics are interpolated explicitly
- edge distance is computed from `min(barycentric)`
- derivative-aware edge width is used to keep thickness roughly pixel-consistent
- `dr.antialias()` is not used for interior wireframe edges
- `beauty_plus_wireframe` composites the wireframe over beauty in premultiplied form

Controls:

- `--wireframe-color r,g,b`
- `--wireframe-opacity`
- `--wireframe-thickness-px`

## Requirements

Required:

- Python 3.10+
- NVIDIA GPU
- CUDA-enabled PyTorch
- `nvdiffrast`

Additional Python packages:

- `numpy`
- `trimesh`
- `imageio`
- `Pillow`
- `opencv-python`

Optional:

- `ffmpeg`, `ffprobe`
  - preferred path for `.hdr` / `.exr` loading
  - otherwise the code falls back to `imageio` / OpenCV
- `glfw`, `PyOpenGL`
  - needed only if you use `--display`

## Installation Notes

This repository is safest to use in an environment where CUDA, PyTorch, and `nvdiffrast` are already aligned correctly.

At minimum, these two pieces must match:

- a CUDA-enabled PyTorch build
- a `nvdiffrast` build compatible with that CUDA environment

Exact `nvdiffrast` install used by this repository:

```bash
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
python -m pip install /tmp/extensions/nvdiffrast --no-build-isolation
```

Or use:

```bash
bash scripts/install_nvdiffrast.sh
```

A typical install for the auxiliary packages looks like this:

```bash
pip install numpy trimesh imageio pillow opencv-python
```

Install `torch`, then install `nvdiffrast` using the commands above, then install this repository with `pip install .` (or `pip install -e .` during development).

## Validation

All-mode inspection render:

```bash
nvdiffrast-mesh-render example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/mode_check.png \
    --resolution 256 \
    --render-all
```

This is a smoke/inspection workflow, not a golden-image comparison.

Benchmark-enabled render-all:

```bash
nvdiffrast-mesh-render-all example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/mode_check \
    --resolution 256 \
    --benchmark-runs 5 \
    --benchmark-warmup-runs 2
```

Multi-view inspection render:

```bash
nvdiffrast-mesh-render-multi-view example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/mode_grid \
    --resolution 256 \
    --render-mode beauty \
    --azim-start 0 \
    --azim-end 180 \
    --azim-step 45 \
    --multi-view-chunk-size 4
```

Canonical six-view inspection render:

```bash
nvdiffrast-mesh-render-multi-view example_meshes/c7fd79edb639400293683095caafff21_1024.glb \
    --output outputs/mode_grid_canonical \
    --resolution 256 \
    --render-mode beauty \
    --canonical-six-views
```

## Programmatic Usage

```python
from argparse import Namespace

from nvdiffrast_mesh_renderer.config import config_from_args
from nvdiffrast_mesh_renderer.renderer import SceneRenderer

args = Namespace(
    input="example_meshes/c7fd79edb639400293683095caafff21_1024.glb",
    output="outputs/from_code.png",
    resolution=512,
    elev=0.0,
    azim=0.0,
    fov=45.0,
    distance=None,
    distance_scale=1.15,
    env_map="",
    env_usage="light",
    env_light_intensity=0.3,
    env_background_intensity=1.0,
    env_diffuse_samples=16,
    background="transparent",
    light_intensity=1.1,
    exposure=1.0,
    tonemap="reinhard",
    cull_mode="auto",
    no_antialias=False,
    display=False,
    render_mode="beauty",
    wireframe_color="0.2,1.0,0.25",
    wireframe_opacity=1.0,
    wireframe_thickness_px=0.5,
    normalize_depth=False,
    render_all=False,
)

config = config_from_args(args)
SceneRenderer(config).render_to_file()
```

## Current Limitations

- CUDA is mandatory.
- Only square output resolutions are currently supported.
- `depth_*`, `triangle_id`, and similar outputs are exported as display-oriented images, not raw float buffers.
- `--render-all` re-renders the scene once per mode, so it is intended as a validation/inspection workflow rather than the fastest batch path.
- Multi-view rendering currently cannot be combined with `--render-all`.
- `--display` is not meaningful for multi-view rendering because the view grid is emitted as files.
- `--canonical-six-views` cannot be combined with explicit multi-view range flags such as `--azim-start` or `--elev-start`.
- There is no support for animation, skinning, morph targets, or multi-camera scenes.
- Hidden-line and multi-layer rendering are not implemented yet, though the raster path is structured so `nvdiffrast` DepthPeeler can be introduced later.

## Troubleshooting

### `RuntimeError: CUDA is required for this renderer`

- Check that PyTorch is a CUDA build.
- Check that a GPU is visible in the runtime environment.
- If you are in a sandbox/container, confirm GPU passthrough is enabled.

### `.hdr` / `.exr` files fail to load

- Check whether `ffmpeg` and `ffprobe` are installed.
- If that path still fails, confirm the `imageio` / OpenCV fallback path works in your environment.

### `--display` does not open a window

- Check whether `glfw` and `PyOpenGL` are installed.
- Check whether the environment can create a GUI/OpenGL context.

### A model renders as missing or empty

- Check how `trimesh` is dumping the scene.
- Confirm the mesh actually contains faces.
- Confirm the material is not fully transparent.

## Reference Files

- `render_glb.py`
- `render_all_modes.py`
- `render_multi_view.py`
- `nvdiffrast_mesh_renderer/`
