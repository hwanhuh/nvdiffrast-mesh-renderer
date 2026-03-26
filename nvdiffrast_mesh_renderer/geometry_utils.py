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


def compute_face_normals_torch(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
    return torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1, eps=1e-8)


def compute_vertex_tangents_torch(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uv: Optional[torch.Tensor],
    normals: torch.Tensor,
) -> Optional[torch.Tensor]:
    if uv is None or uv.shape[0] != vertices.shape[0]:
        return None
    i1, i2, i3 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
    w1, w2, w3 = uv[i1], uv[i2], uv[i3]
    x1, x2 = v2 - v1, v3 - v1
    s1, s2 = w2[:, 0] - w1[:, 0], w3[:, 0] - w1[:, 0]
    t1, t2 = w2[:, 1] - w1[:, 1], w3[:, 1] - w1[:, 1]
    denom = s1 * t2 - s2 * t1
    valid = torch.abs(denom) > 1e-8
    inv = torch.zeros_like(denom)
    inv[valid] = 1.0 / denom[valid]
    sdir = torch.zeros_like(x1)
    tdir = torch.zeros_like(x1)
    sdir[valid] = (t2[valid].unsqueeze(-1) * x1[valid] - t1[valid].unsqueeze(-1) * x2[valid]) * inv[valid].unsqueeze(-1)
    tdir[valid] = (s1[valid].unsqueeze(-1) * x2[valid] - s2[valid].unsqueeze(-1) * x1[valid]) * inv[valid].unsqueeze(-1)
    tan1 = torch.zeros((vertices.shape[0], 3), dtype=vertices.dtype, device=vertices.device)
    tan2 = torch.zeros((vertices.shape[0], 3), dtype=vertices.dtype, device=vertices.device)
    tan1.index_add_(0, i1, sdir)
    tan1.index_add_(0, i2, sdir)
    tan1.index_add_(0, i3, sdir)
    tan2.index_add_(0, i1, tdir)
    tan2.index_add_(0, i2, tdir)
    tan2.index_add_(0, i3, tdir)
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-8)
    tangent = torch.nn.functional.normalize(
        tan1 - normals * torch.sum(normals * tan1, dim=-1, keepdim=True),
        dim=-1,
        eps=1e-8,
    )
    handedness = torch.sign(torch.sum(torch.cross(normals, tangent, dim=-1) * tan2, dim=-1, keepdim=True))
    handedness = torch.where(handedness == 0.0, torch.ones_like(handedness), handedness)
    return torch.cat([tangent, handedness], dim=-1)


def scene_bounds(meshes: Sequence[MeshData]) -> Tuple[np.ndarray, float]:
    vertices = torch.cat([mesh.positions for mesh in meshes], dim=0)
    center = ((vertices.min(dim=0).values + vertices.max(dim=0).values) * 0.5).cpu().numpy().astype(np.float32)
    center_t = torch.as_tensor(center, dtype=vertices.dtype, device=vertices.device)
    radius = float(torch.linalg.norm(vertices - center_t, dim=-1).max().item())
    return center, max(radius, 1e-3)
