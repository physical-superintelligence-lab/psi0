from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real.SONIC.dex1_virtual_runtime import (
    Dex1VirtualMapper,
    Dex1VirtualTeleopBridge,
    patch_sonic_proprio_hands,
)
from real.SONIC.run_data_exporter_dex1 import (
    select_virtual_hand_targets,
)
from scripts.data.raw_sonic_to_psi_lerobot import (
    MOTION_TOKEN_DIM,
    Sonic2LeRobotConverter,
    pack_psi_state,
    source_hand7_from_actuated,
)


class FakeController:
    def __init__(self):
        self.target = None
        self.started = False
        self.closed = False

    def get_current_dual_gripper_q(self):
        return 2.75, 2.75

    def start_publishing(self):
        self.started = True

    def ctrl_dual_gripper(self, left, right):
        self.target = (left, right)

    def close(self):
        self.closed = True


class Dex1Virtual14Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper = Dex1VirtualMapper()

    def test_openness_roundtrip(self):
        for left in np.linspace(0.0, 1.0, 5):
            for right in np.linspace(0.0, 1.0, 5):
                hand = self.mapper.hand14(float(left), float(right))
                command = self.mapper.command_from_hand14(hand)
                self.assertAlmostEqual(command.left, left, places=5)
                self.assertAlmostEqual(command.right, right, places=5)

    def test_virtual_hands_are_dense_and_converter_preserves_them(self):
        left, right = self.mapper.hand7_pair(0.2, 0.8)
        self.assertGreater(np.count_nonzero(np.abs(left) > 1e-6), 1)
        self.assertGreater(np.count_nonzero(np.abs(right) > 1e-6), 1)
        source = np.zeros(43, dtype=np.float32)
        source[22:29] = source_hand7_from_actuated(left)
        source[36:43] = source_hand7_from_actuated(right)
        packed = pack_psi_state(source)
        np.testing.assert_allclose(packed[29:36], left)
        np.testing.assert_allclose(packed[36:43], right)

    def test_converter_virtual_action_prefers_teleop_hand_target(self):
        converter = Sonic2LeRobotConverter()
        converter.end_effector = "dex1_virtual14"
        target = self.mapper.hand14(0.15, 0.85)
        wbc = np.zeros(43, dtype=np.float32)
        action = np.asarray(
            converter.build_act(np.zeros(MOTION_TOKEN_DIM), wbc, hand14_override=target)
        )
        np.testing.assert_allclose(action[MOTION_TOKEN_DIM:], target)

    def test_exporter_pose_uses_sonic_target_and_planner_mode_uses_planner(self):
        sonic = {
            "left_hand_joints": np.full(7, 1.0),
            "right_hand_joints": np.full(7, 2.0),
        }
        planner = {
            "left_hand_joints": np.full(7, 3.0),
            "right_hand_joints": np.full(7, 4.0),
        }
        left, right = select_virtual_hand_targets(1, sonic, planner)
        np.testing.assert_allclose(left, 1.0)
        np.testing.assert_allclose(right, 2.0)
        left, right = select_virtual_hand_targets(5, sonic, planner)
        np.testing.assert_allclose(left, 3.0)
        np.testing.assert_allclose(right, 4.0)
        self.assertIsNone(select_virtual_hand_targets(1, None, planner))

    def test_trigger_bridge_uses_trigger_not_recording_grip(self):
        controller = FakeController()
        bridge = Dex1VirtualTeleopBridge(self.mapper, controller, max_step_rad=10.0)
        left, right = bridge.compute_hand_joints_from_inputs(
            None, None, 0.0, 1.0, 1.0, 0.0
        )
        expected_left, expected_right = self.mapper.hand7_pair(1.0, 0.0)
        np.testing.assert_allclose(left, expected_left)
        np.testing.assert_allclose(right, expected_right)
        self.assertEqual(controller.target, (5.5, 0.0))
        bridge.close()
        self.assertTrue(controller.closed)

    def test_exporter_patch_uses_state_for_observation_and_target_for_action(self):
        left_action, right_action = self.mapper.hand7_pair(0.1, 0.9)
        original = {"body_q": np.arange(29), "left_hand_q": np.ones(7)}
        patched = patch_sonic_proprio_hands(
            original,
            left_q=5.5,
            right_q=0.0,
            left_action7=left_action,
            right_action7=right_action,
            mapper=self.mapper,
        )
        expected_left_state, expected_right_state = self.mapper.hand7_pair(1.0, 0.0)
        np.testing.assert_allclose(patched["left_hand_q"], expected_left_state)
        np.testing.assert_allclose(patched["right_hand_q"], expected_right_state)
        np.testing.assert_allclose(patched["last_left_hand_action"], left_action)
        np.testing.assert_allclose(patched["last_right_hand_action"], right_action)
        self.assertIsNot(patched, original)
        np.testing.assert_array_equal(original["left_hand_q"], np.ones(7))


if __name__ == "__main__":
    unittest.main()
