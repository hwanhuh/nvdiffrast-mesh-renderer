from collections import OrderedDict
import pathlib
import subprocess
import weakref
from typing import Callable, Optional

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

import nvdiffrast.torch as dr

from .math_utils import safe_normalize, srgb_to_linear
from .types import GpuTexture


def load_image_file(path: pathlib.Path) -> np.ndarray:
    if path.suffix.lower() not in {".hdr", ".exr"}:
        image = np.asarray(iio.imread(path))
        image = image[..., None] if image.ndim == 2 else image
        return image.astype(np.float32) / np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else image.astype(np.float32)
    try:
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)], text=True).strip()
        width, height = [int(value) for value in probe.split("x")]
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-vframes", "1", "-"])
        return np.frombuffer(raw, dtype=np.float32).reshape(3, height, width).transpose(1, 2, 0).copy()
    except Exception:
        try:
            image = np.asarray(iio.imread(path))
        except Exception:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        image = image[..., None] if image.ndim == 2 else image
        return image.astype(np.float32)


def image_to_numpy(image: Image.Image, mode: str) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")
    array = np.asarray(image.convert(mode))
    return (array[..., None] if array.ndim == 2 else array).astype(np.float32) / 255.0


def mip_level_count(height: int, width: int) -> int:
    levels, h, w = 0, int(height), int(width)
    while h > 1 or w > 1:
        if (h > 1 and h % 2 != 0) or (w > 1 and w % 2 != 0):
            return 0
        h, w, levels = max(1, h // 2), max(1, w // 2), levels + 1
    return levels


def make_gpu_texture(array: np.ndarray, device: torch.device, srgb: bool) -> GpuTexture:
    array = array[..., None] if array.ndim == 2 else array
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device, dtype=torch.float32)
    tensor = torch.flip(tensor, dims=(0,))
    if srgb and tensor.shape[-1] >= 3:
        rgb = srgb_to_linear(tensor[..., :3])
        tensor = torch.cat([rgb, tensor[..., 3:]], dim=-1) if tensor.shape[-1] > 3 else rgb
    tensor = tensor.unsqueeze(0).contiguous()
    max_mip_level = mip_level_count(tensor.shape[1], tensor.shape[2])
    mip = dr.texture_construct_mip(tensor, max_mip_level=max_mip_level) if max_mip_level > 0 else None
    return GpuTexture(tex=tensor, mip=mip, can_mip=max_mip_level > 0, max_mip_level=max_mip_level or None)


class TextureCache:
    def __init__(self, device: torch.device, *, max_file_entries: int = 0):
        self.device = device
        self._cache: dict[tuple[str, int, bool, object], tuple[weakref.ReferenceType[object], GpuTexture]] = {}
        self._max_file_entries = max(int(max_file_entries), 0)
        self._file_cache: OrderedDict[tuple[str, str, bool, object, tuple[int, int, int, int]], GpuTexture] = OrderedDict()

    def _cache_key(self, kind: str, source: object, srgb: bool, variant: object) -> tuple[str, int, bool, object]:
        return (kind, id(source), srgb, variant)

    def _get_cached(self, key: tuple[str, int, bool, object], source: object) -> Optional[GpuTexture]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        source_ref, texture = entry
        if source_ref() is not source:
            self._cache.pop(key, None)
            return None
        return texture

    def _store_cached(self, key: tuple[str, int, bool, object], source: object, texture: GpuTexture) -> GpuTexture:
        def _remove(_ref: weakref.ReferenceType[object], *, cache=self._cache, cache_key=key) -> None:
            cache.pop(cache_key, None)

        self._cache[key] = (weakref.ref(source, _remove), texture)
        return texture

    def _make_texture(self, array: np.ndarray, srgb: bool) -> GpuTexture:
        return make_gpu_texture(array, self.device, srgb=srgb)

    def _file_signature(self, path: pathlib.Path) -> tuple[pathlib.Path, tuple[int, int, int, int]]:
        resolved = path.expanduser().resolve()
        stat = resolved.stat()
        return resolved, (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))

    def _trim_file_cache(self) -> None:
        while self._max_file_entries > 0 and len(self._file_cache) > self._max_file_entries:
            self._file_cache.popitem(last=False)

    def get_pil(self, image: Optional[Image.Image], srgb: bool, mode: str) -> Optional[GpuTexture]:
        if image is None:
            return None
        key = self._cache_key("pil", image, srgb, mode)
        cached = self._get_cached(key, image)
        if cached is not None:
            return cached
        return self._store_cached(key, image, self._make_texture(image_to_numpy(image, mode=mode), srgb=srgb))

    def get_array(self, array: np.ndarray, srgb: bool = False) -> GpuTexture:
        key = self._cache_key("array", array, srgb, tuple(array.shape))
        cached = self._get_cached(key, array)
        if cached is not None:
            return cached
        return self._store_cached(key, array, self._make_texture(array, srgb=srgb))

    def get_file(
        self,
        path: pathlib.Path,
        srgb: bool = False,
        *,
        variant: object = None,
        loader: Callable[[pathlib.Path], np.ndarray] | None = None,
    ) -> GpuTexture:
        resolved, signature = self._file_signature(path)
        if self._max_file_entries <= 0:
            array = load_image_file(resolved) if loader is None else loader(resolved)
            return self._make_texture(array, srgb=srgb)
        key = ("file", str(resolved), srgb, variant, signature)
        cached = self._file_cache.get(key)
        if cached is not None:
            self._file_cache.move_to_end(key)
            return cached
        array = load_image_file(resolved) if loader is None else loader(resolved)
        texture = self._make_texture(array, srgb=srgb)
        self._file_cache[key] = texture
        self._file_cache.move_to_end(key)
        self._trim_file_cache()
        return texture

    def clear(self) -> None:
        self._cache.clear()
        self._file_cache.clear()


def sample_texture(
    texture: Optional[GpuTexture],
    uv: Optional[torch.Tensor],
    uv_da: Optional[torch.Tensor] = None,
    boundary_mode: str = "wrap",
    mip_level_bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if texture is None or uv is None:
        return None
    kwargs = {}
    filter_mode = "linear"
    if texture.can_mip and (uv_da is not None or mip_level_bias is not None):
        filter_mode = "linear-mipmap-linear"
        if uv_da is not None:
            kwargs["uv_da"] = uv_da
        if mip_level_bias is not None:
            kwargs["mip_level_bias"] = mip_level_bias.squeeze(-1)
        if texture.mip is not None:
            kwargs["mip"] = texture.mip
            kwargs["max_mip_level"] = texture.max_mip_level
    return dr.texture(texture.tex, uv, filter_mode=filter_mode, boundary_mode=boundary_mode, **kwargs)


def direction_to_latlong_uv(direction: torch.Tensor) -> torch.Tensor:
    direction = safe_normalize(direction)
    u = torch.atan2(direction[..., 0], direction[..., 2]) / (2.0 * np.pi) + 0.5
    v = torch.acos(direction[..., 1].clamp(-1.0, 1.0)) / np.pi
    return torch.stack([torch.remainder(u, 1.0), v.clamp(0.0, 1.0)], dim=-1)
