#!/usr/bin/env python3
"""Run the official SONIC PICO manager with the Psi0 Dex1 adapter."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import sys
import threading
import time
from pathlib import Path


PSI_ROOT = Path(__file__).resolve().parents[2]
SONIC_DIR = Path(
    os.environ.get("SONIC_DIR", PSI_ROOT / "third_party/GR00T-WholeBodyControl")
).resolve()
for path in (PSI_ROOT, SONIC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real.SONIC.dex1_virtual_runtime import (  # noqa: E402
    DEFAULT_STATS,
    Dex1VirtualMapper,
    Dex1VirtualTeleopBridge,
)
from real.teleop.robot_control.robot_hand_dex1_1 import Dex1_1_Controller  # noqa: E402
from gear_sonic.scripts import pico_manager_thread_server as official_manager  # noqa: E402


class StartChordReleaseFilter:
    """Arbitrate asynchronous face-button chords before exposing them.

    A/B and X/Y arrive from different controllers and are not sampled
    atomically. Hold partial chords until the input has settled so the Remote
    Vision four-button chord cannot leak A+X, B+Y, A+B, or X+Y to SONIC.
    """

    ZERO = (False, False, False, False)

    def __init__(
        self,
        read_buttons,
        release_frames: int = 3,
        settle_sec: float = 0.15,
        clock=time.monotonic,
    ):
        self._read_buttons = read_buttons
        self._release_frames = max(1, int(release_frames))
        self._settle_sec = max(0.0, float(settle_sec))
        self._clock = clock
        self._reset()

    def _reset(self):
        self._active = False
        self._accumulated = self.ZERO
        self._committed = None
        self._suppress = False
        self._last_change = 0.0
        self._release_count = 0

    def __call__(self, *args, **kwargs):
        raw_buttons = tuple(bool(value) for value in self._read_buttons(*args, **kwargs))
        now = self._clock()

        if not self._active:
            if not any(raw_buttons):
                return self.ZERO
            self._active = True
            self._accumulated = raw_buttons
            self._last_change = now
            self._suppress = all(raw_buttons)
            if not self._suppress and self._settle_sec == 0.0:
                self._committed = raw_buttons
            return self._committed or self.ZERO

        if any(raw_buttons):
            self._release_count = 0
            accumulated = tuple(
                before or current
                for before, current in zip(self._accumulated, raw_buttons)
            )
            if accumulated != self._accumulated:
                self._accumulated = accumulated
                self._last_change = now
            if all(self._accumulated):
                self._suppress = True
                self._committed = None
            if self._suppress:
                return self.ZERO
            if self._committed is None and now - self._last_change >= self._settle_sec:
                self._committed = self._accumulated
            return self._committed or self.ZERO

        self._release_count += 1
        if self._suppress:
            if self._release_count >= self._release_frames:
                self._reset()
            return self.ZERO
        if self._committed is not None:
            if self._release_count >= self._release_frames:
                self._reset()
                return self.ZERO
            return self._committed
        if self._release_count >= self._release_frames:
            result = self._accumulated
            self._reset()
            return result
        return self.ZERO


class ControllerInputRouter:
    """Keep official controls while making recording grip chords exclusive.

    The official manager checks ``left_grip + A/B`` independently from its
    multi-button mode chords. Remote Vision owns physical B, so map physical
    ``left_grip + Y`` to the official internal B discard command. Mask the grip
    for every other face-button sample so modes cannot become recording
    commands and physical B cannot discard data.
    """

    def __init__(
        self,
        read_buttons,
        read_inputs,
        read_axis_clicks=None,
        release_frames: int = 3,
        button_settle_sec: float = 0.15,
        policy_hold_sec: float = 0.6,
        clock=time.monotonic,
    ):
        self._read_buttons = read_buttons
        self._read_inputs = read_inputs
        self._read_axis_clicks = read_axis_clicks
        self._release_frames = release_frames
        self._button_settle_sec = float(button_settle_sec)
        self._policy_hold_sec = float(policy_hold_sec)
        self._clock = clock
        self._local = threading.local()
        self._policy_hold_started = None

    @contextmanager
    def planner_input_scope(self):
        """Hide the synthetic policy chord from PlannerStreamer reads.

        The official manager and PlannerStreamer call ``get_abxy_buttons``
        sequentially on the same thread, so thread identity cannot isolate the
        internal start/stop chord from locomotion controls.
        """

        previous = getattr(self._local, "planner_input", False)
        self._local.planner_input = True
        try:
            yield
        finally:
            self._local.planner_input = previous

    def _button_filter(self) -> StartChordReleaseFilter:
        button_filter = getattr(self._local, "button_filter", None)
        if button_filter is None:
            button_filter = StartChordReleaseFilter(
                self._read_buttons,
                release_frames=self._release_frames,
                settle_sec=self._button_settle_sec,
                clock=self._clock,
            )
            self._local.button_filter = button_filter
        return button_filter

    def get_abxy_buttons(self, *args, **kwargs):
        buttons = self._button_filter()(*args, **kwargs)
        # Remote Vision owns the physical four-face-button chord.  Mask it and
        # its debounced release aliases from SONIC; policy toggle is synthesized
        # below from a deliberate right-stick hold on the manager main thread.
        if all(buttons):
            buttons = (False, False, False, False)

        recording_grip_allowed = buttons == (True, False, False, False)
        if buttons == (False, False, False, True):
            try:
                inputs = self._read_inputs(*args, **kwargs)
                left_grip = inputs[3] if len(inputs) == 5 else 0.0
            except Exception:
                left_grip = 0.0
            if float(left_grip) > 0.5:
                buttons = (False, True, False, False)
                recording_grip_allowed = True

        if (
            self._read_axis_clicks is not None
            and threading.current_thread() is threading.main_thread()
            and not getattr(self._local, "planner_input", False)
        ):
            try:
                _left_click, right_click = self._read_axis_clicks(*args, **kwargs)
            except Exception:
                right_click = False
            if bool(right_click):
                now = self._clock()
                if self._policy_hold_started is None:
                    self._policy_hold_started = now
                if now - self._policy_hold_started >= self._policy_hold_sec:
                    buttons = (True, True, True, True)
            else:
                self._policy_hold_started = None

        self._local.last_buttons = buttons
        self._local.recording_grip_allowed = recording_grip_allowed
        return buttons

    def get_controller_inputs(self, *args, **kwargs):
        inputs = self._read_inputs(*args, **kwargs)
        if len(inputs) != 5:
            return inputs
        left_menu, left_trigger, right_trigger, left_grip, right_grip = inputs
        if not getattr(self._local, "recording_grip_allowed", False):
            left_grip = 0.0
        return left_menu, left_trigger, right_trigger, left_grip, right_grip


def require_dex1_state_pair(controller) -> tuple[float, float]:
    """Fail closed before publishing when either physical Dex1 state is absent."""
    left_q, right_q = controller.get_current_dual_gripper_q()
    if left_q is None or right_q is None:
        controller.close()
        raise RuntimeError(
            "Dex1 manager requires both left and right state before publishing; "
            f"got left={left_q} right={right_q}"
        )
    return float(left_q), float(right_q)


def enter_sonic_runtime_dir(sonic_dir: Path = SONIC_DIR) -> None:
    """Resolve official SONIC runtime assets from the upstream repository root."""

    os.chdir(sonic_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--buffer-size", type=int, default=15)
    parser.add_argument("--num-frames-to-send", type=int, default=5)
    parser.add_argument("--target-fps", type=int, default=50)
    parser.add_argument("--zmq-feedback-host", default="localhost")
    parser.add_argument("--zmq-feedback-port", type=int, default=5557)
    parser.add_argument("--network", default="enp4s0")
    parser.add_argument("--stats", default=os.environ.get("DEX1_VIRTUAL_STATS", str(DEFAULT_STATS)))
    parser.add_argument("--max-step-rad", type=float, default=0.5)
    parser.add_argument("--no-dex1-hardware", action="store_true")
    parser.add_argument("--vis-vr3pt", action="store_true")
    parser.add_argument("--waist-tracking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enter_sonic_runtime_dir()
    mapper = Dex1VirtualMapper(args.stats)

    if args.no_dex1_hardware:
        class NoopController:
            def get_current_dual_gripper_q(self):
                return 5.5, 5.5

            def start_publishing(self):
                pass

            def ctrl_dual_gripper(self, _left, _right):
                pass

            def close(self):
                pass

        controller = NoopController()
    else:
        controller = Dex1_1_Controller(network=args.network, init_dds=True)
        require_dex1_state_pair(controller)
    bridge = Dex1VirtualTeleopBridge(mapper, controller, max_step_rad=args.max_step_rad)
    official_manager.compute_hand_joints_from_inputs = bridge.compute_hand_joints_from_inputs
    input_router = ControllerInputRouter(
        official_manager.get_abxy_buttons,
        official_manager.get_controller_inputs,
        official_manager.get_axis_clicks,
    )
    official_manager.get_abxy_buttons = input_router.get_abxy_buttons
    official_manager.get_controller_inputs = input_router.get_controller_inputs
    official_planner_run_once = official_manager.PlannerStreamer.run_once

    def planner_run_once_without_policy_toggle(planner, *args, **kwargs):
        with input_router.planner_input_scope():
            return official_planner_run_once(planner, *args, **kwargs)

    official_manager.PlannerStreamer.run_once = planner_run_once_without_policy_toggle
    print(
        f"[Psi0 Dex1] virtual14 enabled: stats={mapper.stats_path} "
        f"network={args.network} hardware={not args.no_dex1_hardware}"
    )
    print(
        "[Psi0 Controls] hold right stick click for 0.6s to start/stop policy; "
        "Left Grip+A records, Left Grip+Y discards; physical B and A+B+X+Y "
        "remain Remote Vision only"
    )
    try:
        official_manager.run_pico_manager(
            port=args.port,
            buffer_size=args.buffer_size,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            zmq_feedback_host=args.zmq_feedback_host,
            zmq_feedback_port=args.zmq_feedback_port,
            enable_vis_vr3pt=args.vis_vr3pt,
            enable_waist_tracking=args.waist_tracking,
            input_source="xrt",
        )
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
