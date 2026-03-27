import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from nvdiffrast_mesh_renderer.config import build_argparser, config_from_args
from nvdiffrast_mesh_renderer.image_io import HostImage, encode_jpg_bytes, encode_png_bytes, save_image
from nvdiffrast_mesh_renderer.postprocess import ImagePostprocessor


class ImageIoTests(unittest.TestCase):
    def test_postprocess_returns_host_uint8_image(self):
        config = config_from_args(build_argparser().parse_args(["example.glb", "--png-compression", "2"]))
        postprocessor = ImagePostprocessor(config, device=torch.device("cpu"))
        rgb = torch.tensor([[[[0.25, 0.5, 0.75]]]], dtype=torch.float32)
        alpha = torch.ones((1, 1, 1, 1), dtype=torch.float32)

        image = postprocessor.postprocess(rgb, alpha, render_mode="mask")

        self.assertIsInstance(image, HostImage)
        array = image.numpy()
        self.assertEqual(array.dtype, np.uint8)
        self.assertEqual(array.shape, (1, 1, 4))
        np.testing.assert_array_equal(array[0, 0], np.array([64, 128, 191, 255], dtype=np.uint8))

    def test_save_image_and_encode_png_bytes_accept_host_image(self):
        tensor = torch.tensor([[[255, 64, 32, 255]]], dtype=torch.uint8)
        image = HostImage(tensor=tensor)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            png_path = output_dir / "sample.png"
            jpg_path = output_dir / "sample.jpg"

            save_image(png_path, image, png_compression=1)
            save_image(jpg_path, image, jpg_quality=90)
            payload = encode_png_bytes(image, png_compression=1)

            self.assertTrue(png_path.is_file())
            self.assertTrue(jpg_path.is_file())
            self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_encode_jpg_bytes_accepts_rgba_host_image(self):
        tensor = torch.tensor([[[255, 64, 32, 128]]], dtype=torch.uint8)
        image = HostImage(tensor=tensor)

        payload = encode_jpg_bytes(image, jpg_quality=85, background_rgb=(255, 255, 255))

        self.assertTrue(payload.startswith(b"\xff\xd8\xff"))
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)[0, 0]
        self.assertGreaterEqual(int(rgb[0]), 200)
        self.assertGreaterEqual(int(rgb[1]), 140)
        self.assertGreaterEqual(int(rgb[2]), 120)


if __name__ == "__main__":
    unittest.main()
