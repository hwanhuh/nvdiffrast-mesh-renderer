import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


def safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


def safe_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), eps, None)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.clamp(x, min=0.0) ** (1.0 / 2.4) - 0.055)


def aces_tonemap(x: torch.Tensor) -> torch.Tensor:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def reinhard_tonemap(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + x)


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    return np.array(
        [[f / aspect, 0.0, 0.0, 0.0], [0.0, f, 0.0, 0.0], [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)], [0.0, 0.0, -1.0, 0.0]],
        dtype=np.float32,
    )


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = safe_normalize_np(target - eye)
    up = safe_normalize_np(up.astype(np.float32))
    if abs(float(np.dot(forward, up))) > 0.999:
        fallback_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(forward, fallback_up))) > 0.999:
            fallback_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = fallback_up
    right = safe_normalize_np(np.cross(forward, up))
    true_up = safe_normalize_np(np.cross(right, forward))
    view = np.eye(4, dtype=np.float32)
    view[0, :3], view[1, :3], view[2, :3] = right, true_up, -forward
    view[:3, 3] = -(view[:3, :3] @ eye)
    return view


def orbit_camera(
    center: np.ndarray,
    radius: float,
    elev_deg: float,
    azim_deg: float,
    fov_y_deg: float,
    distance_scale: float = 1.15,
    distance_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    radius = max(float(radius), 1e-3)
    fov_half = math.radians(fov_y_deg) * 0.5
    distance = distance_override if distance_override is not None else radius / math.sin(max(fov_half, 1e-3))
    distance *= distance_scale
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    direction = np.array([math.cos(elev) * math.sin(azim), math.sin(elev), math.cos(elev) * math.cos(azim)], dtype=np.float32)
    return (center + direction * distance).astype(np.float32), center.astype(np.float32), float(distance)


def to_float_array(value, channels: int, default: Sequence[float]) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=np.float32)
    arr = np.asarray(value)
    if arr.size < channels:
        raise ValueError(f"Expected at least {channels} channels, got {arr}")
    arr = arr.reshape(-1)[:channels].astype(np.float32)
    return arr / 255.0 if arr.max(initial=0.0) > 1.0 else arr
