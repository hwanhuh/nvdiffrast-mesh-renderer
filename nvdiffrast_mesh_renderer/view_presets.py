import hashlib
import math
import random
from typing import Any

CANONICAL_RENDER_COND_VIEW_COUNT = 16
CANONICAL_RENDER_COND_MAX_ABS_ELEV = 60.0
CANONICAL_RENDER_COND_FOV_MIN = 10.0
CANONICAL_RENDER_COND_FOV_MAX = 70.0
CANONICAL_RENDER_COND_DISTANCE_SCALE = 1.0
_CANONICAL_RENDER_COND_CANDIDATE_COUNTS = (20, 24, 32)


def _stable_seed(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _radical_inverse(base: int, n: int) -> float:
    result = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        result += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return result


def _canonical_render_cond_angles(index: int, num_samples: int, offset: tuple[float, float]) -> tuple[float, float]:
    u = (index / num_samples) + (offset[0] / num_samples)
    v = _radical_inverse(2, index) + offset[1]
    u = 2.0 * u if u < 0.25 else (2.0 / 3.0) * u + (1.0 / 3.0)
    elev = math.degrees(math.acos(1.0 - 2.0 * u) - (math.pi / 2.0))
    yaw = math.degrees(v * 2.0 * math.pi)
    yaw = ((yaw + 180.0) % 360.0) - 180.0
    azim = ((90.0 - yaw + 180.0) % 360.0) - 180.0
    return elev, azim


def _sample_render_cond_fov(rng: random.Random) -> float:
    fov_min = CANONICAL_RENDER_COND_FOV_MIN
    fov_max = CANONICAL_RENDER_COND_FOV_MAX
    radius_min = math.sqrt(3.0) / 2.0 / math.sin(math.radians(fov_max) * 0.5)
    radius_max = math.sqrt(3.0) / 2.0 / math.sin(math.radians(fov_min) * 0.5)
    k_min = 1.0 / (radius_max * radius_max)
    k_max = 1.0 / (radius_min * radius_min)
    k = rng.uniform(k_min, k_max)
    radius = 1.0 / math.sqrt(k)
    return math.degrees(2.0 * math.asin(min(math.sqrt(3.0) / (2.0 * radius), 1.0)))


def _candidate_render_cond_views(seed_key: str, candidate_count: int) -> list[dict[str, Any]]:
    rng = random.Random(_stable_seed(seed_key))
    offset = (rng.random(), rng.random())
    accepted: list[dict[str, Any]] = []
    for source_index in range(candidate_count):
        elev, azim = _canonical_render_cond_angles(source_index, candidate_count, offset)
        fov = _sample_render_cond_fov(rng)
        if abs(elev) > CANONICAL_RENDER_COND_MAX_ABS_ELEV:
            continue
        accepted.append(
            {
                "name": f"render_cond_{len(accepted):02d}",
                "elev": elev,
                "azim": azim,
                "distance": None,
                "distance_scale": CANONICAL_RENDER_COND_DISTANCE_SCALE,
                "fov": fov,
                "light_seed": _stable_seed(f"{seed_key}:light:{source_index}"),
            }
        )
        if len(accepted) >= CANONICAL_RENDER_COND_VIEW_COUNT:
            return accepted
    return accepted


def generate_canonical_render_cond_views(seed_key: str) -> list[dict[str, Any]]:
    normalized_key = seed_key.strip()
    if not normalized_key:
        raise ValueError("seed_key must not be empty")
    for candidate_count in _CANONICAL_RENDER_COND_CANDIDATE_COUNTS:
        views = _candidate_render_cond_views(normalized_key, candidate_count)
        if len(views) >= CANONICAL_RENDER_COND_VIEW_COUNT:
            return views[:CANONICAL_RENDER_COND_VIEW_COUNT]
    raise RuntimeError("Failed to generate enough canonical render_cond views")
