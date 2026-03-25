import pathlib

import imageio
import numpy as np
import torch

from .config import RenderConfig
from .math_utils import aces_tonemap, linear_to_srgb, reinhard_tonemap


class ImagePostprocessor:
    def __init__(self, config: RenderConfig):
        self.config = config

    def postprocess(self, rgb: torch.Tensor, alpha: torch.Tensor, render_mode: str | None = None) -> np.ndarray:
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
        image = torch.cat([rgb, torch.clamp(alpha, 0.0, 1.0)], dim=-1)[0].flip(0).cpu().numpy()
        return image

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

    def save(self, path: pathlib.Path, image: np.ndarray) -> None:
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            imageio.imwrite(path, (image[..., :3] * 255.0).round().clip(0, 255).astype(np.uint8))
            return
        imageio.imwrite(path, (image * 255.0).round().clip(0, 255).astype(np.uint8))
