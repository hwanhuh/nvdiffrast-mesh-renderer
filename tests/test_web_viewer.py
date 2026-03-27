import unittest

from web_viewer import DEFAULT_BACKGROUND_HEX, _parse_background_hex


class WebViewerTests(unittest.TestCase):
    def test_parse_background_hex_accepts_valid_color(self):
        normalized, rgb = _parse_background_hex("#12abEf")

        self.assertEqual(normalized, "#12abef")
        self.assertEqual(rgb, (0x12, 0xAB, 0xEF))

    def test_parse_background_hex_falls_back_to_default(self):
        normalized, rgb = _parse_background_hex(None)

        self.assertEqual(normalized, DEFAULT_BACKGROUND_HEX)
        self.assertEqual(rgb, (255, 255, 255))

    def test_parse_background_hex_rejects_invalid_color(self):
        with self.assertRaises(ValueError):
            _parse_background_hex("white")


if __name__ == "__main__":
    unittest.main()
