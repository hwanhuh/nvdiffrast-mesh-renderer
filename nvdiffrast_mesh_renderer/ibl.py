import math
from typing import List, Sequence

import numpy as np
import torch

from .environment import sample_environment
from .math_utils import safe_normalize
from .types import EnvironmentData


def radical_inverse_vdc(bits: np.ndarray) -> np.ndarray:
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return bits.astype(np.float64) * 2.3283064365386963e-10


def tangent_basis_from_normal(normal: torch.Tensor):
    up_y = torch.tensor([0.0, 1.0, 0.0], device=normal.device, dtype=normal.dtype)
    up_x = torch.tensor([1.0, 0.0, 0.0], device=normal.device, dtype=normal.dtype)
    up = torch.where((torch.abs(normal[..., 1:2]) > 0.99), up_x.view(1, 1, 1, 3), up_y.view(1, 1, 1, 3))
    tangent = safe_normalize(torch.cross(up, normal, dim=-1))
    return tangent, safe_normalize(torch.cross(normal, tangent, dim=-1))


def reflect(direction: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.sum(normal * direction, dim=-1, keepdim=True) * normal - direction


def fresnel_schlick(cos_theta: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    return f0 + (1.0 - f0) * torch.clamp(1.0 - cos_theta, min=0.0, max=1.0) ** 5.0


def fresnel_schlick_roughness(cos_theta: torch.Tensor, f0: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    return f0 + (torch.maximum(torch.ones_like(f0) - roughness, f0) - f0) * torch.clamp(1.0 - cos_theta, 0.0, 1.0) ** 5.0


def distribution_ggx(n_dot_h: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    alpha2 = (roughness ** 2) ** 2
    denom = (n_dot_h ** 2) * (alpha2 - 1.0) + 1.0
    return alpha2 / torch.clamp(math.pi * denom ** 2, min=1e-6)


def geometry_smith(n_dot_v: torch.Tensor, n_dot_l: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    r = roughness + 1.0
    k = (r * r) / 8.0
    ggx_v = n_dot_v / torch.clamp(n_dot_v * (1.0 - k) + k, min=1e-6)
    ggx_l = n_dot_l / torch.clamp(n_dot_l * (1.0 - k) + k, min=1e-6)
    return ggx_v * ggx_l


def environment_brdf_approx(f0: torch.Tensor, roughness: torch.Tensor, n_dot_v: torch.Tensor) -> torch.Tensor:
    c0 = torch.tensor([-1.0, -0.0275, -0.572, 0.022], device=f0.device, dtype=f0.dtype)
    c1 = torch.tensor([1.0, 0.0425, 1.04, -0.04], device=f0.device, dtype=f0.dtype)
    r = roughness * c0.view(1, 1, 1, 4) + c1.view(1, 1, 1, 4)
    a004 = torch.minimum(r[..., :1] * r[..., :1], torch.exp2(-9.28 * n_dot_v)) * r[..., :1] + r[..., 1:2]
    ab = torch.cat([-1.04 * a004 + r[..., 2:3], 1.04 * a004 + r[..., 3:4]], dim=-1)
    return f0 * ab[..., :1] + ab[..., 1:2]


class ImageBasedLighting:
    def __init__(self, env: EnvironmentData, diffuse_samples: int, device: torch.device):
        self.env = env
        self.samples = self._build_cosine_samples(diffuse_samples, device)

    def _build_cosine_samples(self, count: int, device: torch.device) -> List[torch.Tensor]:
        if count <= 0:
            return []
        idx = np.arange(count, dtype=np.uint32)
        xi1 = (idx.astype(np.float64) + 0.5) / float(count)
        xi2 = radical_inverse_vdc(idx)
        r = np.sqrt(xi1)
        phi = 2.0 * np.pi * xi2
        samples = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(np.clip(1.0 - xi1, 0.0, 1.0))], axis=-1).astype(np.float32)
        return [torch.tensor(sample, device=device, dtype=torch.float32) for sample in samples]

    def diffuse(self, normal: torch.Tensor) -> torch.Tensor:
        if not self.samples:
            return torch.zeros_like(normal)
        tangent, bitangent = tangent_basis_from_normal(normal)
        irradiance = torch.zeros_like(normal)
        for sample in self.samples:
            world_dir = safe_normalize(tangent * sample[0] + bitangent * sample[1] + normal * sample[2])
            irradiance = irradiance + sample_environment(self.env, world_dir, intensity=self.env.light_intensity)
        return irradiance / float(len(self.samples))

    def specular(self, normal: torch.Tensor, view_dir: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        dominant_dir = safe_normalize(torch.lerp(reflect(view_dir, normal), normal, roughness * roughness))
        max_mip = float(self.env.texture.max_mip_level or 0)
        mip_bias = torch.clamp(roughness * (1.5 - 0.5 * roughness) * max_mip, 0.0, max_mip)
        return sample_environment(self.env, dominant_dir, mip_level_bias=mip_bias, intensity=self.env.light_intensity)
