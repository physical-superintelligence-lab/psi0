"""Run Psi0's official SONIC RTC client with a physical Dex1-1 adapter."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from real.SONIC.dex1_virtual_runtime import DEFAULT_STATS, Dex1VirtualMapper
from real.SONIC.psi0_vla_dex1_bridge import (
    Dex1ActionRouter,
    Dex1PolicyBridge,
    Dex1StateSubscriberAdapter,
)
from real.teleop.robot_control.robot_hand_dex1_1 import Dex1_1_Controller
from scripts.offline.real_dex1_1_client import RealDex11Driver


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_CLIENT = (
    REPO_ROOT / "third_party" / "GR00T-WholeBodyControl" / "psi_rtc_sonic_client.py"
)
class _ControllerStateReader:
    def __init__(self, controller: Dex1_1_Controller) -> None:
        self.controller = controller

    def get_q(self):
        return self.controller.get_current_dual_gripper_q()


def load_official_client(path: str | Path):
    client_path = Path(path).expanduser().resolve()
    if not client_path.is_file():
        raise FileNotFoundError(f"official Psi0 SONIC client not found: {client_path}")
    root = str(client_path.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("psi0_official_rtc_sonic_client", client_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official Psi0 SONIC client: {client_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_dex1_adapters(official, bridge: Dex1PolicyBridge) -> None:
    original_state_subscriber = official.RobotStateSubscriber
    original_websocket_client = official.RTCWebSocketClient
    router = Dex1ActionRouter(bridge)

    def state_subscriber_factory(*args, **kwargs):
        return Dex1StateSubscriberAdapter(
            original_state_subscriber(*args, **kwargs), bridge
        )

    class Dex1RTCWebSocketClient(original_websocket_client):
        def execute_action(self, action):
            policy_action = np.asarray(action, dtype=np.float32)
            if policy_action.ndim > 1:
                policy_action = policy_action[0]
            sonic_action = router.route(policy_action)
            sonic_action[:64] = official.fsq_quantize(sonic_action[:64])
            self._token_publisher.publish_token(sonic_action)

    official.RobotStateSubscriber = state_subscriber_factory
    official.RTCWebSocketClient = Dex1RTCWebSocketClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official Psi0 SONIC RTC client with Dex1-1 state/action mapping"
    )
    parser.add_argument("--official-client", default=str(DEFAULT_OFFICIAL_CLIENT))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--zmq-host", default="localhost")
    parser.add_argument("--zmq-pub-port", type=int, default=5556)
    parser.add_argument("--zmq-sub-port", type=int, default=5557)
    parser.add_argument("--zmq-topic", default="pose")
    parser.add_argument("--zmq-sub-topic", default="g1_debug")
    parser.add_argument("--camera-address", default="tcp://192.168.123.164:5558")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--state-history-length", type=int, default=1)
    parser.add_argument("--network", default="enp4s0")
    parser.add_argument("--stats", default=str(DEFAULT_STATS))
    parser.add_argument("--enable-dex1-live", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Import the official client and validate dependencies without DDS/ZMQ control",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official = load_official_client(args.official_client)
    if args.check:
        print(f"OFFICIAL CLIENT READY: {Path(args.official_client).resolve()}")
        print("DEX1 LIVE CONTROL: disabled")
        return
    if not args.enable_dex1_live:
        raise SystemExit("refusing real deployment without --enable-dex1-live")

    mapper = Dex1VirtualMapper(args.stats)
    controller = Dex1_1_Controller(network=args.network)
    controller.start_publishing()
    driver = RealDex11Driver(
        network=args.network,
        controller=controller,
        dry_run=False,
    )
    bridge = Dex1PolicyBridge(
        mapper=mapper,
        state_reader=_ControllerStateReader(controller),
        driver=driver,
    )
    install_dex1_adapters(official, bridge)
    official.TASK_INSTRUCTION = args.instruction
    try:
        official.main(
            server_url=f"ws://{args.host}:{args.port}/ws",
            zmq_host=args.zmq_host,
            zmq_pub_port=args.zmq_pub_port,
            zmq_sub_port=args.zmq_sub_port,
            zmq_topic=args.zmq_topic,
            zmq_sub_topic=args.zmq_sub_topic,
            camera_address=args.camera_address,
            history_length=args.state_history_length,
        )
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
