"""Map normalized policy commands to the physical Dex1-1 grippers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scripts.offline.dex3_to_dex1 import Dex11Command

Q_CLOSED_DEFAULT = 0.0
Q_OPEN_DEFAULT = 5.5


@dataclass(frozen=True)
class Dex11Calibration:
    """Convert between openness (1=open) and gripper position in radians."""

    q_closed: float = Q_CLOSED_DEFAULT
    q_open: float = Q_OPEN_DEFAULT
    invert: bool = False

    def norm_to_q(self, norm: float) -> float:
        n = float(np.clip(norm, 0.0, 1.0))
        if self.invert:
            n = 1.0 - n
        return self.q_closed + n * (self.q_open - self.q_closed)

    def q_to_norm(self, q: float) -> float:
        span = self.q_open - self.q_closed
        if span == 0:
            return 0.0
        n = (float(q) - self.q_closed) / span
        if self.invert:
            n = 1.0 - n
        return float(np.clip(n, 0.0, 1.0))


def rate_limit(prev_q: float, target_q: float, max_step_rad: float) -> float:
    """Limit a command to one bounded position step."""
    if max_step_rad is None or max_step_rad <= 0:
        return float(target_q)
    delta = float(np.clip(target_q - prev_q, -max_step_rad, max_step_rad))
    return float(prev_q + delta)


@dataclass
class RealDex11Driver:
    """Send rate-limited scalar commands through the Dex1 DDS controller."""

    network: str = "enp4s0"
    calib: Dex11Calibration = field(default_factory=Dex11Calibration)
    kp: float = 5.0
    kd: float = 0.05
    max_step_rad: float = 0.5
    dry_run: bool = True
    controller: object | None = None
    _last_q: dict = field(default_factory=lambda: {"left": None, "right": None}, init=False)
    last_command: dict | None = field(default=None, init=False)

    def _ensure_controller(self):
        if self.controller is not None or self.dry_run:
            return
        # 懒加载真实 DDS 控制器
        from real.teleop.robot_control.robot_hand_dex1_1 import Dex1_1_Controller

        self.controller = Dex1_1_Controller(
            network=self.network, kp=self.kp, kd=self.kd
        )
        self.controller.start_publishing()

    def _seed_last_q(self, side: str, target_q: float) -> float:
        """Seed rate limiting from measured state when available."""
        if self._last_q[side] is not None:
            return self._last_q[side]
        if self.controller is not None and hasattr(self.controller, "get_current_dual_gripper_q"):
            lq, rq = self.controller.get_current_dual_gripper_q()
            cur = lq if side == "left" else rq
            if cur is not None:
                return float(cur)
        return float(target_q)

    def send(self, command: Dex11Command) -> None:
        self._ensure_controller()
        target = {
            "left": self.calib.norm_to_q(command.left),
            "right": self.calib.norm_to_q(command.right),
        }
        out = {}
        for side in ("left", "right"):
            base = self._seed_last_q(side, target[side])
            q = rate_limit(base, target[side], self.max_step_rad)
            self._last_q[side] = q
            out[side] = q

        self.last_command = {
            "norm_left": float(command.left),
            "norm_right": float(command.right),
            "q_left": out["left"],
            "q_right": out["right"],
            "dry_run": self.dry_run,
        }

        if not self.dry_run and self.controller is not None:
            self.controller.ctrl_dual_gripper(out["left"], out["right"])

    def get_state_norm(self) -> tuple[float | None, float | None]:
        if self.controller is None or not hasattr(self.controller, "get_current_dual_gripper_q"):
            return None, None
        lq, rq = self.controller.get_current_dual_gripper_q()
        ln = self.calib.q_to_norm(lq) if lq is not None else None
        rn = self.calib.q_to_norm(rq) if rq is not None else None
        return ln, rn

    def close(self):
        if self.controller is not None and hasattr(self.controller, "close"):
            self.controller.close()
