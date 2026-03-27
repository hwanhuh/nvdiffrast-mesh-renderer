import math
import pathlib
from typing import Optional, Tuple

import torch

from .config import RenderConfig
from .math_utils import safe_normalize
from .textures import TextureCache, direction_to_latlong_uv, load_image_file, sample_texture
from .types import CameraData, EnvironmentData


def _load_environment_array(path: pathlib.Path):
    env_array = load_image_file(path)
    env_array = env_array[..., :3] if env_array.shape[-1] > 3 else env_array
    return env_array.astype("float32", copy=False)


def sample_environment(
    env: EnvironmentData,
    direction: torch.Tensor,
    mip_level_bias: Optional[torch.Tensor] = None,
    intensity: Optional[float] = None,
) -> torch.Tensor:
    sampled = sample_texture(env.texture, direction_to_latlong_uv(direction), boundary_mode="clamp", mip_level_bias=mip_level_bias)
    if sampled is None:
        raise RuntimeError("Environment map sampling failed")
    return sampled[..., :3] * float(env.light_intensity if intensity is None else intensity)


class BackgroundRayGridCache:
    def __init__(self):
        self._coords: dict[tuple[str, int | None, int], torch.Tensor] = {}

    def get_coords(self, resolution: int, device: torch.device) -> torch.Tensor:
        # Reuse the screen-space coordinate vector; per-view ray directions still depend on FOV and camera rotation.
        key = (device.type, device.index, int(resolution))
        coords = self._coords.get(key)
        if coords is None:
            coords = torch.linspace(-1.0 + 1.0 / resolution, 1.0 - 1.0 / resolution, resolution, device=device, dtype=torch.float32)
            self._coords[key] = coords
        return coords

    def clear(self) -> None:
        self._coords.clear()


class EnvironmentService:
    def __init__(self, cache: TextureCache, *, background_grid_cache: BackgroundRayGridCache | None = None):
        self.cache = cache
        self.background_grid_cache = BackgroundRayGridCache() if background_grid_cache is None else background_grid_cache

    def clear_persistent_caches(self) -> None:
        self.cache.clear()
        self.background_grid_cache.clear()

    def build(self, config: RenderConfig) -> Optional[EnvironmentData]:
        if not config.env_map:
            return None
        return EnvironmentData(
            texture=self.cache.get_file(
                pathlib.Path(config.env_map),
                srgb=False,
                variant="env_rgb",
                loader=_load_environment_array,
            ),
            light_intensity=config.env_light_intensity,
            background_intensity=config.env_background_intensity,
        )

    def render_background(
        self,
        camera: CameraData,
        config: RenderConfig,
        env: Optional[EnvironmentData],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = config.resolution
        if env is not None and config.env_usage in {"background", "both"}:
            if getattr(camera, "projection_type", "perspective") == "orthographic":
                dirs_world = camera.forward.view(1, 1, 1, 3).expand(1, res, res, 3)
            else:
                coords = self.background_grid_cache.get_coords(res, device)
                yv = coords.view(res, 1).expand(res, res)
                xv = coords.view(1, res).expand(res, res)
                tan_half = math.tan(math.radians(config.fov) * 0.5)
                dirs_cam = safe_normalize(
                    torch.stack(
                        [
                            xv * tan_half,
                            yv * tan_half,
                            torch.full((res, res), -1.0, device=device, dtype=torch.float32),
                        ],
                        dim=-1,
                    )
                )
                dirs_world = torch.matmul(dirs_cam, camera.cam_to_world[:3, :3].t()).unsqueeze(0)
            bg = sample_environment(env, dirs_world, intensity=env.background_intensity)
            return bg, torch.ones_like(bg[..., :1])
        if config.background_transparent:
            shape = (1, res, res, 3)
            bg = torch.zeros(shape, device=device, dtype=torch.float32)
            return bg, torch.zeros((1, res, res, 1), device=device, dtype=torch.float32)
        rgba = torch.as_tensor(config.background_rgba, dtype=torch.float32, device=device).view(1, 1, 1, 4)
        bg = (rgba[..., :3] * rgba[..., 3:4]).expand(1, res, res, 3).clone()
        alpha = rgba[..., 3:4].expand(1, res, res, 1).clone()
        return bg, alpha
