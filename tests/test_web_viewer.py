import unittest

from web_viewer import DEFAULT_BACKGROUND_HEX, PAGE_HTML, _parse_background_hex


class WebViewerTests(unittest.TestCase):
    def test_parse_background_hex_accepts_valid_color(self):
        normalized, rgb = _parse_background_hex("#12abEf")

        self.assertEqual(normalized, "#12abef")
        self.assertEqual(rgb, (0x12, 0xAB, 0xEF))

    def test_parse_background_hex_falls_back_to_default(self):
        normalized, rgb = _parse_background_hex(None)

        self.assertEqual(normalized, DEFAULT_BACKGROUND_HEX)
        self.assertEqual(rgb, (0xBC, 0xBC, 0xBC))

    def test_parse_background_hex_rejects_invalid_color(self):
        with self.assertRaises(ValueError):
            _parse_background_hex("white")

    def test_page_exposes_camera_selector(self):
        self.assertIn('id="cameraInput"', PAGE_HTML)
        self.assertIn('value="perspective"', PAGE_HTML)
        self.assertIn('value="orthographic"', PAGE_HTML)


if __name__ == "__main__":
    unittest.main()
