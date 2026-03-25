from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .math_utils import safe_normalize_np
from .types import MeshData


def compute_vertex_tangents(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, normals: np.ndarray) -> Optional[np.ndarray]:
    if uv is None or len(uv) != len(vertices):
        return None
    tan1 = np.zeros((len(vertices), 3), dtype=np.float32)
    tan2 = np.zeros((len(vertices), 3), dtype=np.float32)
    i1, i2, i3 = faces[:, 0], faces[:, 1], faces[:, 2]
    v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
    w1, w2, w3 = uv[i1], uv[i2], uv[i3]
    x1, x2 = v2 - v1, v3 - v1
    s1, s2 = w2[:, 0] - w1[:, 0], w3[:, 0] - w1[:, 0]
    t1, t2 = w2[:, 1] - w1[:, 1], w3[:, 1] - w1[:, 1]
    denom = s1 * t2 - s2 * t1
    valid = np.abs(denom) > 1e-8
    inv = np.zeros_like(denom, dtype=np.float32)
    inv[valid] = 1.0 / denom[valid]
    sdir = np.zeros_like(x1, dtype=np.float32)
    tdir = np.zeros_like(x1, dtype=np.float32)
    sdir[valid] = (t2[valid, None] * x1[valid] - t1[valid, None] * x2[valid]) * inv[valid, None]
    tdir[valid] = (s1[valid, None] * x2[valid] - s2[valid, None] * x1[valid]) * inv[valid, None]
    np.add.at(tan1, i1, sdir), np.add.at(tan1, i2, sdir), np.add.at(tan1, i3, sdir)
    np.add.at(tan2, i1, tdir), np.add.at(tan2, i2, tdir), np.add.at(tan2, i3, tdir)
    normals = safe_normalize_np(normals.astype(np.float32))
    tangent = safe_normalize_np(tan1 - normals * np.sum(normals * tan1, axis=-1, keepdims=True))
    handedness = np.sign(np.sum(np.cross(normals, tangent) * tan2, axis=-1, keepdims=True))
    handedness[handedness == 0.0] = 1.0
    return np.concatenate([tangent, handedness.astype(np.float32)], axis=-1).astype(np.float32)


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return safe_normalize_np(np.cross(v1 - v0, v2 - v0).astype(np.float32))


def scene_bounds(meshes: Sequence[MeshData]) -> Tuple[np.ndarray, float]:
    vertices = torch.cat([mesh.positions for mesh in meshes], dim=0)
    center = ((vertices.min(dim=0).values + vertices.max(dim=0).values) * 0.5).cpu().numpy().astype(np.float32)
    center_t = torch.as_tensor(center, dtype=vertices.dtype, device=vertices.device)
    radius = float(torch.linalg.norm(vertices - center_t, dim=-1).max().item())
    return center, max(radius, 1e-3)
