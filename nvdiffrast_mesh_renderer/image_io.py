from __future__ import annotations

import pathlib
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Deque

import cv2
import numpy as np
import torch

DEFAULT_JPG_QUALITY = 95
DEFAULT_PNG_COMPRESSION = 1


@dataclass
class HostImage:
    tensor: torch.Tensor
    ready_event: torch.cuda.Event | None = None
    device_tensor: torch.Tensor | None = None
    _array: np.ndarray | None = None

    def wait(self) -> None:
        if self.ready_event is not None:
            self.ready_event.synchronize()
            self.ready_event = None
        self.device_tensor = None

    def numpy(self) -> np.ndarray:
        self.wait()
        if self._array is None:
            self._array = self.tensor.numpy()
        return self._array

    def __array__(self, dtype=None):
        array = self.numpy()
        if dtype is None:
            return array
        return array.astype(dtype, copy=False)


def stage_host_image(image: torch.Tensor, *, copy_stream: torch.cuda.Stream | None = None) -> HostImage:
    image = image.contiguous()
    if image.device.type != "cuda":
        return HostImage(tensor=image.detach().to(device="cpu", dtype=torch.uint8).contiguous())
    host = torch.empty(image.shape, dtype=image.dtype, device="cpu", pin_memory=True)
    current_stream = torch.cuda.current_stream(device=image.device)
    stream = copy_stream if copy_stream is not None else current_stream
    if stream is not current_stream:
        stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        host.copy_(image, non_blocking=True)
        ready_event = torch.cuda.Event()
        ready_event.record(stream)
    return HostImage(tensor=host, ready_event=ready_event, device_tensor=image)


def to_numpy_image(image: HostImage | np.ndarray) -> np.ndarray:
    if isinstance(image, HostImage):
        return image.numpy()
    return np.asarray(image)


def to_uint8_image(image: HostImage | np.ndarray) -> np.ndarray:
    array = to_numpy_image(image)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(np.rint(array * 255.0).clip(0, 255).astype(np.uint8))
    raise TypeError(f"Unsupported image dtype: {array.dtype}")


def _cv_image_for_path(array: np.ndarray, suffix: str) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected HWC image array, got shape {array.shape}")
    channels = array.shape[-1]
    if suffix in {".jpg", ".jpeg"}:
        if channels == 4:
            array = array[..., :3]
            channels = 3
        if channels != 3:
            raise ValueError("JPEG output requires RGB or RGBA input")
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if suffix == ".png":
        if channels == 4:
            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
        if channels == 3:
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        if channels == 1:
            return array
        raise ValueError("PNG output requires 1, 3, or 4 channels")
    raise ValueError(f"Unsupported output format: {suffix}")


def save_image(
    path: pathlib.Path,
    image: HostImage | np.ndarray,
    *,
    jpg_quality: int = DEFAULT_JPG_QUALITY,
    png_compression: int = DEFAULT_PNG_COMPRESSION,
) -> pathlib.Path:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    array = to_uint8_image(image)
    encoded = _cv_image_for_path(array, suffix)
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    elif suffix == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
    if not cv2.imwrite(str(output_path), encoded, params):
        raise RuntimeError(f"Failed to write image: {output_path}")
    return output_path


def encode_png_bytes(image: HostImage | np.ndarray, *, png_compression: int = DEFAULT_PNG_COMPRESSION) -> bytes:
    array = to_uint8_image(image)
    encoded = _cv_image_for_path(array, ".png")
    ok, payload = cv2.imencode(".png", encoded, [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)])
    if not ok:
        raise RuntimeError("Failed to encode PNG bytes")
    return payload.tobytes()


def _save_image_job(
    image: HostImage | np.ndarray,
    output_path: pathlib.Path,
    *,
    jpg_quality: int,
    png_compression: int,
) -> tuple[pathlib.Path, float]:
    started = time.perf_counter()
    save_image(output_path, image, jpg_quality=jpg_quality, png_compression=png_compression)
    return output_path, (time.perf_counter() - started) * 1000.0


@dataclass
class PendingSave:
    future: Future[tuple[pathlib.Path, float]]


class AsyncImageSaver:
    def __init__(
        self,
        *,
        max_pending: int = 4,
        jpg_quality: int = DEFAULT_JPG_QUALITY,
        png_compression: int = DEFAULT_PNG_COMPRESSION,
    ):
        self.max_pending = max(int(max_pending), 1)
        self.jpg_quality = int(jpg_quality)
        self.png_compression = int(png_compression)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvdiffrast-image-save")
        self._pending: Deque[PendingSave] = deque()

    def submit(self, image: HostImage | np.ndarray, output_path: pathlib.Path) -> list[tuple[pathlib.Path, float]]:
        future = self._executor.submit(
            _save_image_job,
            image,
            pathlib.Path(output_path),
            jpg_quality=self.jpg_quality,
            png_compression=self.png_compression,
        )
        self._pending.append(PendingSave(future=future))
        completed: list[tuple[pathlib.Path, float]] = []
        while len(self._pending) > self.max_pending:
            completed.append(self._pending.popleft().future.result())
        return completed

    def finish(self) -> list[tuple[pathlib.Path, float]]:
        completed: list[tuple[pathlib.Path, float]] = []
        while self._pending:
            completed.append(self._pending.popleft().future.result())
        return completed

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
