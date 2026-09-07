"""RealDex11Driver 纯逻辑单测(不依赖 unitree_sdk2py / DDS)。"""

import unittest

import numpy as np

from scripts.offline.dex3_to_dex1 import Dex11Command
from scripts.offline.real_dex1_1_client import (
    Dex11Calibration,
    RealDex11Driver,
    rate_limit,
)


class FakeController:
    """注入用假控制器,记录命令并可回报状态。"""

    def __init__(self, state=(0.0, 0.0)):
        self.cmds = []
        self._state = state  # (left_q, right_q)

    def get_current_dual_gripper_q(self):
        return self._state

    def ctrl_dual_gripper(self, left_q, right_q):
        self.cmds.append((left_q, right_q))


class TestCalibration(unittest.TestCase):
    def test_norm_to_q_endpoints(self):
        c = Dex11Calibration(q_closed=0.0, q_open=5.5)
        self.assertAlmostEqual(c.norm_to_q(0.0), 0.0)
        self.assertAlmostEqual(c.norm_to_q(1.0), 5.5)
        self.assertAlmostEqual(c.norm_to_q(0.5), 2.75)

    def test_clamp_out_of_range(self):
        c = Dex11Calibration(q_closed=0.0, q_open=5.5)
        self.assertAlmostEqual(c.norm_to_q(-1.0), 0.0)
        self.assertAlmostEqual(c.norm_to_q(2.0), 5.5)

    def test_invert(self):
        c = Dex11Calibration(q_closed=0.0, q_open=5.5, invert=True)
        self.assertAlmostEqual(c.norm_to_q(0.0), 5.5)
        self.assertAlmostEqual(c.norm_to_q(1.0), 0.0)

    def test_round_trip(self):
        for invert in (False, True):
            c = Dex11Calibration(q_closed=0.1, q_open=5.4, invert=invert)
            for n in (0.0, 0.25, 0.5, 0.9, 1.0):
                self.assertAlmostEqual(c.q_to_norm(c.norm_to_q(n)), n, places=5)


class TestRateLimit(unittest.TestCase):
    def test_limits_step(self):
        self.assertAlmostEqual(rate_limit(0.0, 5.0, 0.5), 0.5)
        self.assertAlmostEqual(rate_limit(5.0, 0.0, 0.5), 4.5)

    def test_within_step_passes(self):
        self.assertAlmostEqual(rate_limit(1.0, 1.2, 0.5), 1.2)

    def test_zero_disables(self):
        self.assertAlmostEqual(rate_limit(0.0, 9.9, 0.0), 9.9)


class TestRealDex11DriverDryRun(unittest.TestCase):
    def test_dry_run_no_controller_needed(self):
        drv = RealDex11Driver(dry_run=True, max_step_rad=0.0)
        drv.send(Dex11Command(left=1.0, right=0.0))
        self.assertTrue(drv.last_command["dry_run"])
        self.assertAlmostEqual(drv.last_command["q_left"], 5.5)
        self.assertAlmostEqual(drv.last_command["q_right"], 0.0)
        self.assertIsNone(drv.controller)


class TestRealDex11DriverWithController(unittest.TestCase):
    def test_rate_limited_from_measured_state(self):
        fake = FakeController(state=(0.0, 0.0))
        drv = RealDex11Driver(dry_run=False, controller=fake, max_step_rad=0.5)
        drv.send(Dex11Command(left=1.0, right=1.0))  # 目标 5.5,但首步从 0 限速到 0.5
        self.assertEqual(len(fake.cmds), 1)
        self.assertAlmostEqual(fake.cmds[0][0], 0.5)
        self.assertAlmostEqual(fake.cmds[0][1], 0.5)
        # 第二步再进 0.5
        drv.send(Dex11Command(left=1.0, right=1.0))
        self.assertAlmostEqual(fake.cmds[1][0], 1.0)

    def test_state_norm_readback(self):
        fake = FakeController(state=(5.5, 0.0))
        drv = RealDex11Driver(dry_run=False, controller=fake)
        ln, rn = drv.get_state_norm()
        self.assertAlmostEqual(ln, 1.0)
        self.assertAlmostEqual(rn, 0.0)


if __name__ == "__main__":
    unittest.main()
