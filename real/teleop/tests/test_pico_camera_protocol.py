from __future__ import annotations

import os
import sys
import unittest

import numpy as np


TELEOP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TELEOP_ROOT not in sys.path:
    sys.path.insert(0, TELEOP_ROOT)

from pico_camera_view import prepare_stereo_frame  # noqa: E402


class PicoCameraFrameTest(unittest.TestCase):
    def test_letterboxes_complete_rgb_frame_for_both_eyes(self) -> None:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        rgb[-1, -1] = [0, 255, 0]

        stereo = prepare_stereo_frame(rgb, output_width=2560, output_height=720)

        self.assertEqual(stereo.shape, (720, 2560, 3))
        left, right = np.hsplit(stereo, 2)
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(left[:, :160], 0)
        np.testing.assert_array_equal(left[:, 1120:], 0)
        self.assertEqual(tuple(left[0, 160]), (0, 0, 255))
        self.assertEqual(tuple(left[-1, 1119]), (0, 255, 0))

    def test_rejects_non_rgb_input(self) -> None:
        with self.assertRaises(ValueError):
            prepare_stereo_frame(np.zeros((10, 10), dtype=np.uint8), 1280, 720)


if __name__ == "__main__":
    unittest.main()
