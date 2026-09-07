"""Psi0-side Dex1 adapter for the official SONIC runtime.

The physical grippers expose two scalar positions.  Psi0/SONIC keeps the
official 14-D hand interface by projecting those scalars through the same
reversible virtual-Dex3 map used by native Psi0 collection.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.offline.dex3_to_dex1 import Dex3ToDex1, load_from_stats_file


DEFAULT_STATS = Path(__file__).resolve().parent / "assets/dex1_virtual_mapping_stats.json"
Q_MIN = 0.0
Q_MAX = 5.5


class Dex1VirtualMapper:
    def __init__(self, stats_path: str | Path = DEFAULT_STATS):
        self.stats_path = Path(stats_path).expanduser().resolve()
        self.dex1_map: Dex3ToDex1 = load_from_stats_file(str(self.stats_path))

    @staticmethod
    def q_to_openness(q: float | None) -> float:
        if q is None:
            raise RuntimeError("Dex1 state is not available")
        return float(np.clip((float(q) - Q_MIN) / (Q_MAX - Q_MIN), 0.0, 1.0))

    @staticmethod
    def openness_to_q(openness: float) -> float:
        return float(Q_MIN + np.clip(openness, 0.0, 1.0) * (Q_MAX - Q_MIN))

    def hand14(self, left_openness: float, right_openness: float) -> np.ndarray:
        return self.dex1_map.state_hand14(left_openness, right_openness)

    def hand7_pair(self, left_openness: float, right_openness: float) -> tuple[np.ndarray, np.ndarray]:
        hand = self.hand14(left_openness, right_openness)
        return hand[:7].copy(), hand[7:].copy()

    def command_from_hand14(self, hand14: np.ndarray):
        return self.dex1_map.hand14_to_command(np.asarray(hand14, dtype=np.float32))


def patch_sonic_proprio_hands(
    proprio: dict[str, Any],
    *,
    left_q: float,
    right_q: float,
    left_action7: np.ndarray,
    right_action7: np.ndarray,
    mapper: Dex1VirtualMapper,
) -> dict[str, Any]:
    """Return a copied SONIC proprio message with physical/virtual Dex1 hands."""
    left_action = np.asarray(left_action7, dtype=np.float64)
    right_action = np.asarray(right_action7, dtype=np.float64)
    if left_action.shape != (7,) or right_action.shape != (7,):
        raise ValueError("virtual hand actions must both have shape (7,)")
    left_state, right_state = mapper.hand7_pair(
        mapper.q_to_openness(left_q), mapper.q_to_openness(right_q)
    )
    patched = dict(proprio)
    patched["left_hand_q"] = left_state.astype(np.float64)
    patched["right_hand_q"] = right_state.astype(np.float64)
    patched["last_left_hand_action"] = left_action
    patched["last_right_hand_action"] = right_action
    return patched


class Dex1VirtualTeleopBridge:
    """Convert PICO triggers to virtual hand7 targets and physical Dex1 q."""

    def __init__(
        self,
        mapper: Dex1VirtualMapper,
        controller: Any,
        max_step_rad: float = 0.5,
    ):
        self.mapper = mapper
        self.controller = controller
        self.max_step_rad = float(max_step_rad)
        left_q, right_q = controller.get_current_dual_gripper_q()
        self.left_q = float(left_q if left_q is not None else Q_MAX)
        self.right_q = float(right_q if right_q is not None else Q_MAX)
        controller.start_publishing()

    def compute_hand_joints_from_inputs(
        self,
        _left_solver,
        _right_solver,
        left_trigger: float,
        _left_grip: float,
        right_trigger: float,
        _right_grip: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Trigger 0=open and 1=closed.  Grip buttons remain available for the
        # official recording shortcuts and therefore do not drive Dex1.
        left_open = 1.0 - float(np.clip(left_trigger, 0.0, 1.0))
        right_open = 1.0 - float(np.clip(right_trigger, 0.0, 1.0))
        left_target = self.mapper.openness_to_q(left_open)
        right_target = self.mapper.openness_to_q(right_open)
        self.left_q += float(np.clip(left_target - self.left_q, -self.max_step_rad, self.max_step_rad))
        self.right_q += float(np.clip(right_target - self.right_q, -self.max_step_rad, self.max_step_rad))
        self.controller.ctrl_dual_gripper(self.left_q, self.right_q)
        return self.mapper.hand7_pair(left_open, right_open)

    def close(self) -> None:
        self.controller.close()


class Dex1StateReader:
    """Read both physical Dex1 states without publishing commands."""

    def __init__(self, network: str = "enp4s0"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

        ChannelFactoryInitialize(0, network)
        self._subs = {
            "left": ChannelSubscriber("rt/dex1/left/state", MotorStates_),
            "right": ChannelSubscriber("rt/dex1/right/state", MotorStates_),
        }
        for sub in self._subs.values():
            sub.Init()
        self._q: dict[str, float | None] = {"left": None, "right": None}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            for side, sub in self._subs.items():
                msg = sub.Read()
                if msg is not None and len(msg.states) > 0:
                    with self._lock:
                        self._q[side] = float(msg.states[0].q)
            time.sleep(0.002)

    def get_q(self) -> tuple[float | None, float | None]:
        with self._lock:
            return self._q["left"], self._q["right"]

    def wait(self, timeout: float = 5.0) -> tuple[float, float]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            left, right = self.get_q()
            if left is not None and right is not None:
                return float(left), float(right)
            time.sleep(0.01)
        raise TimeoutError("timed out waiting for Dex1 state")

    def close(self) -> None:
        self._stop.set()
