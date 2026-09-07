#!/usr/bin/env python3
"""Run the official SONIC exporter with Dex1 state and hand targets."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np


PSI_ROOT = Path(__file__).resolve().parents[2]
SONIC_DIR = Path(
    os.environ.get("SONIC_DIR", PSI_ROOT / "third_party" / "GR00T-WholeBodyControl")
).resolve()
for path in (PSI_ROOT, SONIC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real.SONIC.dex1_virtual_runtime import (  # noqa: E402
    DEFAULT_STATS,
    Dex1StateReader,
    Dex1VirtualMapper,
    patch_sonic_proprio_hands,
)


POSE_STREAM_MODES = frozenset((1, 4))


def select_virtual_hand_targets(
    stream_mode: int,
    sonic_msg: dict | None,
    planner_msg: dict | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Select the same hand source used by the active official SONIC mode."""
    source = sonic_msg if stream_mode in POSE_STREAM_MODES else planner_msg
    if not isinstance(source, dict):
        return None
    left = np.asarray(source.get("left_hand_joints"), dtype=np.float64)
    right = np.asarray(source.get("right_hand_joints"), dtype=np.float64)
    if left.shape != (7,) or right.shape != (7,):
        return None
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return None
    return left.copy(), right.copy()


def main() -> None:
    import tyro
    from gear_sonic.scripts import run_data_exporter as official

    config = tyro.cli(official.SonicDataExporterConfig)
    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    mapper = Dex1VirtualMapper(
        os.environ.get("DEX1_VIRTUAL_STATS", str(DEFAULT_STATS))
    )
    reader = Dex1StateReader(
        network=os.environ.get("G1_NETWORK_INTERFACE", "enp4s0")
    )
    left_q, right_q = reader.wait(timeout=5.0)
    original_collector = official.GrootDataCollector
    original_poll_config = official.poll_robot_config_zmq

    def poll_robot_config(*args, **kwargs):
        robot_config = dict(original_poll_config(*args, **kwargs))
        robot_config["end_effector"] = "dex1_virtual14"
        return robot_config

    class Dex1Collector(original_collector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._dex1_left, self._dex1_right = mapper.hand7_pair(
                mapper.q_to_openness(left_q), mapper.q_to_openness(right_q)
            )

        def _add_data_frame_sonic(self, t_start: float) -> bool:
            targets = select_virtual_hand_targets(
                self.current_stream_mode,
                self.latest_sonic_msg,
                self.latest_planner_msg,
            )
            if targets is not None:
                self._dex1_left, self._dex1_right = targets

            proprio = self.latest_proprio_msg
            if proprio is None:
                return super()._add_data_frame_sonic(t_start)
            measured_left, measured_right = reader.get_q()
            if measured_left is None or measured_right is None:
                raise RuntimeError("Dex1 state stream stopped during recording")

            self.latest_proprio_msg = patch_sonic_proprio_hands(
                proprio,
                left_q=measured_left,
                right_q=measured_right,
                left_action7=self._dex1_left,
                right_action7=self._dex1_right,
                mapper=mapper,
            )
            try:
                return super()._add_data_frame_sonic(t_start)
            finally:
                self.latest_proprio_msg = proprio

    official.poll_robot_config_zmq = poll_robot_config
    official.GrootDataCollector = Dex1Collector
    try:
        official.main(config)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
