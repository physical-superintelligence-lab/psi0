#!/usr/bin/env python3
"""Launch Dex1 collection by reusing SONIC's official tmux launcher."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys


PSI_ROOT = Path(__file__).resolve().parents[2]
SONIC_DIR = Path(
    os.environ.get("SONIC_DIR", PSI_ROOT / "third_party" / "GR00T-WholeBodyControl")
).resolve()
for path in (PSI_ROOT, SONIC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tyro  # noqa: E402
from gear_sonic.scripts import launch_data_collection as official  # noqa: E402


@dataclass
class Dex1LaunchConfig(official.DataCollectionLaunchConfig):
    deploy_checkpoint: str = "policy/sonic_v1_1/model"
    deploy_obs_config: str = "policy/sonic_v1_1/observation_config.yaml"
    deploy_output_type: str = "zmq"
    task_prompt: str = "demo"
    dataset_name: str = ""
    data_exporter_frequency: int = 30
    camera_host: str = "192.168.123.164"
    text_to_speech: bool = False
    root_output_dir: str = "outputs"
    network: str = "enp4s0"
    mapping_stats: str = str(
        PSI_ROOT / "real" / "SONIC" / "assets" / "dex1_virtual_mapping_stats.json"
    )


def main(config: Dex1LaunchConfig) -> None:
    if config.sim:
        raise SystemExit("Use SONIC's official launcher directly for simulation")
    if not config.dataset_name:
        raise SystemExit("--dataset-name is required for Dex1 collection")

    sonic_python = Path(
        os.environ.get("SONIC_PYTHON", Path.home() / "miniconda3/envs/sonic/bin/python")
    ).expanduser().resolve()
    if not sonic_python.is_file():
        raise SystemExit(f"SONIC Python not found: {sonic_python}")

    def check_prerequisites(sim: bool = False) -> None:
        if not shutil.which("tmux"):
            raise SystemExit("tmux is required")
        if not (SONIC_DIR / "gear_sonic_deploy/deploy.sh").is_file():
            raise SystemExit(f"SONIC deploy tree not found: {SONIC_DIR}")

    original_send = official._send_to_pane
    env_script = PSI_ROOT / "real/SONIC/scripts/sonic_conda_env.sh"

    def python_command(script: Path, *args: str) -> str:
        import shlex

        argv = " ".join(shlex.quote(str(value)) for value in (script, *args))
        return (
            f"cd {shlex.quote(str(SONIC_DIR))} && "
            f"export SONIC_DIR={shlex.quote(str(SONIC_DIR))} && "
            f"source {shlex.quote(str(env_script))} && "
            f"export G1_NETWORK_INTERFACE={shlex.quote(config.network)} && "
            f"export DEX1_VIRTUAL_STATS={shlex.quote(str(Path(config.mapping_stats).resolve()))} && "
            f"{shlex.quote(str(sonic_python))} -u {argv}"
        )

    def send_to_pane(pane_index: int, command: str, wait: float = 1.0) -> None:
        if "pico_manager_thread_server.py" in command:
            command = python_command(
                PSI_ROOT / "real/SONIC/run_pico_manager_dex1.py",
                "--network",
                config.network,
                "--stats",
                config.mapping_stats,
            )
        elif "run_data_exporter.py" in command:
            command = python_command(
                PSI_ROOT / "real/SONIC/run_data_exporter_dex1.py",
                "--camera-host",
                config.camera_host,
                "--camera-port",
                str(config.camera_port),
                "--task-prompt",
                config.task_prompt,
                "--dataset-name",
                config.dataset_name,
                "--root-output-dir",
                config.root_output_dir,
                "--data-collection-frequency",
                str(config.data_exporter_frequency),
                "--no-text-to-speech",
            )
        elif "run_camera_viewer.py" in command:
            command = python_command(
                PSI_ROOT / "real/teleop/pico_camera_view.py",
                "--camera-ip",
                config.camera_host,
                "--camera-port",
                str(config.camera_port),
            )
        original_send(pane_index, command, wait)

    os.environ["SONIC_DIR"] = str(SONIC_DIR)
    os.environ["G1_NETWORK_INTERFACE"] = config.network
    os.environ["DEX1_VIRTUAL_STATS"] = str(Path(config.mapping_stats).resolve())
    official._check_prerequisites = check_prerequisites
    official._send_to_pane = send_to_pane
    official.main(config)


if __name__ == "__main__":
    main(tyro.cli(Dex1LaunchConfig))
