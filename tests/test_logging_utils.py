import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from nvdiffrast_mesh_renderer import logging_utils
from nvdiffrast_mesh_renderer.logging_utils import RunLogger


class _FakeStdout(io.StringIO):
    def __init__(self, *, is_tty: bool):
        super().__init__()
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


class RunLoggerConsoleTests(unittest.TestCase):
    def setUp(self):
        logging_utils._ACTIVE_PROGRESS_WIDTH = 0

    def tearDown(self):
        logging_utils._ACTIVE_PROGRESS_WIDTH = 0

    def test_progress_updates_in_place_on_tty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(path=pathlib.Path(temp_dir) / "run.log")
            logger.reset()
            fake_stdout = _FakeStdout(is_tty=True)
            with mock.patch("sys.stdout", fake_stdout):
                with mock.patch(
                    "nvdiffrast_mesh_renderer.logging_utils.shutil.get_terminal_size",
                    return_value=os.terminal_size((120, 20)),
                ):
                    logger.log("progress 1", console="always", console_message="[Progress] [1/2] render-all: beauty")
                    logger.log("progress 2", console="always", console_message="[Progress] [2/2] render-all: mask")
                    logger.log("done", console="always", console_message="[Info] Done")

        output = fake_stdout.getvalue()
        self.assertIn("\r[Progress] [1/2] render-all: beauty", output)
        self.assertIn("\r[Progress] [2/2] render-all: mask", output)
        self.assertTrue(output.endswith("[Info] Done\n"))
        self.assertEqual(logging_utils._ACTIVE_PROGRESS_WIDTH, 0)

    def test_progress_falls_back_to_line_output_without_tty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(path=pathlib.Path(temp_dir) / "run.log")
            logger.reset()
            fake_stdout = _FakeStdout(is_tty=False)
            with mock.patch("sys.stdout", fake_stdout):
                logger.log("progress", console="always", console_message="[Progress] [1/1] batch: mesh")
                logger.log("done", console="always", console_message="[Info] Done")

        output = fake_stdout.getvalue()
        self.assertNotIn("\r", output)
        self.assertIn("[Progress] [1/1] batch: mesh\n", output)
        self.assertTrue(output.endswith("[Info] Done\n"))


if __name__ == "__main__":
    unittest.main()
