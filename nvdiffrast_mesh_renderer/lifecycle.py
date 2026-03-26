import gc
import json
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from .config import RenderConfig
from .renderer import SceneRenderer

_CACHE_KEY_EXCLUDED_FIELDS = frozenset(
    {
        "input",
        "output",
        "elev",
        "azim",
        "elev_start",
        "elev_end",
        "elev_step",
        "azim_start",
        "azim_end",
        "azim_step",
        "canonical_six_views",
        "multi_view_chunk_size",
        "render_all",
        "display",
        "benchmark_requested",
        "benchmark_runs",
        "benchmark_warmup_runs",
    }
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_cuda_oom(exc: BaseException) -> bool:
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        if "out of memory" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def is_cuda_failure(exc: BaseException) -> bool:
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        if "cuda error" in message or "cuda runtime error" in message or "out of memory" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def renderer_cache_key(config: RenderConfig) -> str:
    payload = {
        key: _json_ready(value)
        for key, value in asdict(config).items()
        if key not in _CACHE_KEY_EXCLUDED_FIELDS
    }
    return json.dumps(payload, sort_keys=True)


class RendererCache:
    def __init__(self, *, device: torch.device | None = None):
        self.device = device
        self._renderers: dict[str, SceneRenderer] = {}

    def get(self, config: RenderConfig) -> SceneRenderer:
        key = renderer_cache_key(config)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = SceneRenderer(config, device=self.device)
            self._renderers[key] = renderer
        return renderer

    def clear_texture_caches(self) -> None:
        for renderer in self._renderers.values():
            renderer.clear_texture_cache()

    def drop(self, config: RenderConfig) -> None:
        renderer = self._renderers.pop(renderer_cache_key(config), None)
        if renderer is not None:
            renderer.clear_texture_cache()

    def drop_all(self) -> None:
        for renderer in self._renderers.values():
            renderer.clear_texture_cache()
        self._renderers.clear()

    def release_cuda_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_after_cuda_failure(self, config: RenderConfig | None = None) -> None:
        if config is None:
            self.drop_all()
        else:
            self.drop(config)
        self.clear_texture_caches()
        self.release_cuda_memory()

    def close(self) -> None:
        self.drop_all()
        self.release_cuda_memory()
