import gc
import json
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from .config import RenderConfig
from .environment import EnvironmentService
from .logging_utils import RunLogger
from .renderer import SceneRenderer
from .textures import TextureCache

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
        "canonical_mv_conditions",
        "canonical_render_cond",
        "multi_view_chunk_size",
        "render_all",
        "render_all_batch_size",
        "display",
        "print_progress",
        "png_compression",
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
    def __init__(self, *, device: torch.device | None = None, logger: RunLogger | None = None):
        self.device = torch.device("cuda") if device is None else torch.device(device)
        self.logger = logger
        # Env maps and background helpers are process/GPU-scoped, while mesh textures stay renderer/job-scoped.
        self._environment_service = EnvironmentService(TextureCache(self.device, max_file_entries=4))
        self._active_key: str | None = None
        self._active_renderer: SceneRenderer | None = None

    def _discard_active_renderer(self) -> None:
        renderer = self._active_renderer
        self._active_renderer = None
        self._active_key = None
        if renderer is not None:
            renderer.clear_texture_cache()

    def get_with_status(self, config: RenderConfig) -> tuple[SceneRenderer, bool]:
        key = renderer_cache_key(config)
        if self._active_renderer is not None and self._active_key == key:
            return self._active_renderer, False
        self._discard_active_renderer()
        renderer = SceneRenderer(config, device=self.device, environment_service=self._environment_service, logger=self.logger)
        self._active_renderer = renderer
        self._active_key = key
        return renderer, True

    def get(self, config: RenderConfig) -> SceneRenderer:
        renderer, _created = self.get_with_status(config)
        return renderer

    def clear_texture_caches(self) -> None:
        if self._active_renderer is not None:
            self._active_renderer.clear_texture_cache()

    def drop(self, config: RenderConfig) -> None:
        if self._active_key == renderer_cache_key(config):
            self._discard_active_renderer()

    def drop_all(self) -> None:
        self._discard_active_renderer()

    def clear_process_caches(self) -> None:
        self._environment_service.clear_persistent_caches()

    def release_cuda_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_after_cuda_failure(self, config: RenderConfig | None = None) -> None:
        del config
        # One execution lane keeps at most one live renderer/context, so any CUDA failure taints the active one.
        self.drop_all()
        self.clear_process_caches()
        self.release_cuda_memory()

    def close(self) -> None:
        self.drop_all()
        self.clear_process_caches()
        self.release_cuda_memory()
