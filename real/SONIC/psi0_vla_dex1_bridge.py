"""Dex1-1 state/action adapter for Psi0's official SONIC RTC client."""

from __future__ import annotations

from typing import Any

import numpy as np

from real.SONIC.dex1_virtual_runtime import Dex1VirtualMapper


class Dex1PolicyBridge:
    """Keep live Dex1 state and policy hand actions on one virtual14 map."""

    def __init__(
        self,
        *,
        mapper: Dex1VirtualMapper,
        state_reader: Any,
        driver: Any,
    ) -> None:
        self.mapper = mapper
        self.state_reader = state_reader
        self.driver = driver

    def patch_state_message(self, state: dict[str, Any]) -> dict[str, Any]:
        left_q, right_q = self.state_reader.get_q()
        if left_q is None or right_q is None:
            raise RuntimeError("Dex1 state is not available")
        left, right = self.mapper.hand7_pair(
            self.mapper.q_to_openness(left_q),
            self.mapper.q_to_openness(right_q),
        )
        patched = dict(state)
        patched["left_hand_q"] = left.astype(np.float32)
        patched["right_hand_q"] = right.astype(np.float32)
        return patched

    def send_hand14(self, hand14: np.ndarray):
        command = self.mapper.command_from_hand14(
            np.asarray(hand14, dtype=np.float32)
        )
        self.driver.send(command)
        return command

    def close(self) -> None:
        for resource in (self.driver, self.state_reader):
            close = getattr(resource, "close", None)
            if callable(close):
                close()


class Dex1StateSubscriberAdapter:
    """Patch every official SONIC state sample with measured Dex1 hands."""

    def __init__(self, subscriber: Any, bridge: Dex1PolicyBridge) -> None:
        self.subscriber = subscriber
        self.bridge = bridge

    def get_state(self):
        state = self.subscriber.get_state()
        return None if state is None else self.bridge.patch_state_message(state)

    def get_all_states(self):
        return [
            self.bridge.patch_state_message(state)
            for state in self.subscriber.get_all_states()
        ]

    def __getattr__(self, name: str):
        return getattr(self.subscriber, name)


class Dex1ActionRouter:
    """Route virtual hand14 to Dex1 and leave token-only action for SONIC."""

    def __init__(self, bridge: Dex1PolicyBridge) -> None:
        self.bridge = bridge

    def route(self, action78: np.ndarray) -> np.ndarray:
        action = np.asarray(action78, dtype=np.float32)
        if action.shape != (78,):
            raise ValueError(f"Psi0 SONIC action must have shape (78,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Psi0 SONIC action contains NaN or Inf")
        self.bridge.send_hand14(action[64:78])
        sonic_action = action.copy()
        sonic_action[64:78] = 0.0
        return sonic_action
