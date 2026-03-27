import pathlib

import torch

from .config import RenderConfig
from .image_io import DEFAULT_JPG_QUALITY, HostImage, save_image, stage_host_image
from .math_utils import aces_tonemap, linear_to_srgb, reinhard_tonemap


class ImagePostprocessor:
    def __init__(self, config: RenderConfig, device: torch.device | None = None):
        self.config = config
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self._copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )

    def postprocess(self, rgb: torch.Tensor, alpha: torch.Tensor, render_mode: str | None = None) -> HostImage:
        mode = self.config.render_mode if render_mode is None else render_mode
        rgb = torch.where(alpha > 1e-8, rgb / torch.clamp(alpha, min=1e-8), torch.zeros_like(rgb))
        if mode in {"depth_ndc", "depth_linear"} and self.config.normalize_depth:
            rgb = self._normalize_visible(rgb, alpha)
        if mode in {"beauty", "beauty_plus_wireframe"}:
            rgb = rgb * self.config.exposure
            if self.config.tonemap == "aces":
                rgb = aces_tonemap(rgb)
            elif self.config.tonemap == "reinhard":
                rgb = reinhard_tonemap(rgb)
            elif self.config.tonemap == "none":
                rgb = torch.clamp(rgb, 0.0, 1.0)
            else:
                raise ValueError(f"Unsupported tonemap: {self.config.tonemap}")
            rgb = torch.clamp(linear_to_srgb(torch.clamp(rgb, 0.0, 1.0)), 0.0, 1.0)
        elif mode in {"albedo", "emissive", "wireframe"}:
            rgb = torch.clamp(linear_to_srgb(torch.clamp(rgb, 0.0, 1.0)), 0.0, 1.0)
        else:
            rgb = torch.clamp(rgb, 0.0, 1.0)
        image = torch.cat([rgb, torch.clamp(alpha, 0.0, 1.0)], dim=-1)[0].flip(0)
        image_u8 = torch.round(image * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        return stage_host_image(image_u8, copy_stream=self._copy_stream)

    def _normalize_visible(self, rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        visible = alpha[..., 0] > 1e-5
        if not visible.any():
            return torch.zeros_like(rgb)
        scalar = rgb[..., :1]
        min_val = scalar[visible].min()
        max_val = scalar[visible].max()
        if torch.abs(max_val - min_val) < 1e-8:
            normalized = torch.zeros_like(scalar)
        else:
            normalized = (scalar - min_val) / (max_val - min_val)
        return normalized.expand_as(rgb)

    def save(self, path: pathlib.Path, image: HostImage) -> None:
        save_image(path, image, jpg_quality=DEFAULT_JPG_QUALITY, png_compression=self.config.png_compression)
