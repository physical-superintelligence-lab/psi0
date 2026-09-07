#!/usr/bin/env python3
"""Relay SONIC's ego camera through Psi0's existing Pico Remote Vision stack."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import threading
import time

import cv2
import numpy as np


PSI_ROOT = Path(__file__).resolve().parents[2]
SIMPLE_SRC = PSI_ROOT / "third_party" / "SIMPLE" / "src"
SONIC_ROOT = Path(
    os.environ.get("SONIC_DIR", PSI_ROOT / "third_party" / "GR00T-WholeBodyControl")
).resolve()
for path in (SIMPLE_SRC, SONIC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor  # noqa: E402
from simple.teleop.pico.streaming import FrameBuffer, StreamingThread  # noqa: E402
from simple.teleop.pico.tcp_server import TCPControlServer  # noqa: E402
from simple.teleop.pico.tcp_video_sender import TCPVideoSender  # noqa: E402


def prepare_stereo_frame(
    image_rgb: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Letterbox the complete RGB image independently for each Pico eye."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image_rgb.shape}")
    eye_width = output_width // 2
    if eye_width <= 0 or output_height <= 0:
        raise ValueError(f"invalid output size {output_width}x{output_height}")

    source_height, source_width = image_rgb.shape[:2]
    scale = min(eye_width / source_width, output_height / source_height)
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    eye = np.zeros((output_height, eye_width, 3), dtype=np.uint8)
    x = (eye_width - width) // 2
    y = (output_height - height) // 2
    eye[y : y + height, x : x + width] = resized
    return np.hstack((eye, eye))


class PicoEgoRelay:
    """Connect SIMPLE's official Remote Vision transport to a frame buffer."""

    def __init__(self, listen_host: str, listen_port: int) -> None:
        self._server = TCPControlServer(f"{listen_host}:{listen_port}")
        self._server.on_open_camera = self._open
        self._server.on_close_camera = self._close_stream
        self._lock = threading.Lock()
        self._stream: StreamingThread | None = None
        self._buffer: FrameBuffer | None = None
        self._shape: tuple[int, int] | None = None

    def start(self) -> None:
        self._server.start()

    def _open(self, request: dict) -> None:
        self._close_stream()
        required = ("ip", "port", "width", "height", "fps")
        if any(not request.get(key) for key in required):
            print(f"[PICO_VIEW] incomplete OPEN_CAMERA request: {request}")
            return
        try:
            sender = TCPVideoSender(
                ip=request["ip"],
                port=int(request["port"]),
                width=int(request["width"]),
                height=int(request["height"]),
                fps=int(request["fps"]),
                bitrate=int(request.get("bitrate") or 4_000_000),
                hevc=bool(request.get("enableMvHevc")),
            )
        except OSError as exc:
            print(f"[PICO_VIEW] cannot connect to Pico video receiver: {exc}")
            self._server.close_client()
            return

        buffer = FrameBuffer()
        stream = StreamingThread(
            frame_buffer=buffer,
            fps=int(request["fps"]),
            publishers=[sender],
            on_ended=self._server.close_client,
        )
        with self._lock:
            self._buffer = buffer
            self._stream = stream
            self._shape = (int(request["width"]), int(request["height"]))
        stream.start()

    def _close_stream(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            self._buffer = None
            self._shape = None
        if stream is not None:
            stream.stop()

    def submit(self, image_rgb: np.ndarray) -> None:
        with self._lock:
            buffer = self._buffer
            shape = self._shape
        if buffer is not None and shape is not None:
            buffer.put(prepare_stereo_frame(image_rgb, *shape))

    def close(self) -> None:
        self._close_stream()
        self._server.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-ip", default="192.168.123.164")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--camera-key", default="ego_view")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=13579)
    args = parser.parse_args()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    relay = PicoEgoRelay(args.listen_host, args.listen_port)
    camera = ComposedCameraClientSensor(server_ip=args.camera_ip, port=args.camera_port)
    relay.start()
    print("[PICO_VIEW] video only; no DDS publisher or robot controller is started")
    last_timestamp = None
    try:
        while not stop.is_set():
            message = camera.read(blocking=False)
            if message is None:
                time.sleep(0.002)
                continue
            timestamp = message.get("timestamps", {}).get(args.camera_key)
            image = message.get("images", {}).get(args.camera_key)
            if image is not None and timestamp != last_timestamp:
                relay.submit(image)
                last_timestamp = timestamp
    finally:
        camera.close()
        relay.close()


if __name__ == "__main__":
    main()
