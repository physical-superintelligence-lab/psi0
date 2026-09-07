from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from real.SONIC.dex1_virtual_runtime import Dex1VirtualMapper
from real.SONIC.psi0_vla_dex1_bridge import (
    Dex1ActionRouter,
    Dex1PolicyBridge,
    Dex1StateSubscriberAdapter,
)
from real.SONIC.run_psi0_rtc_sonic_dex1 import install_dex1_adapters


class FakeStateReader:
    def __init__(self, left_q: float, right_q: float):
        self.state = (left_q, right_q)

    def get_q(self):
        return self.state


class FakeDriver:
    def __init__(self):
        self.commands = []

    def send(self, command):
        self.commands.append(command)


def test_physical_state_and_policy_hand_share_one_mapping():
    mapper = Dex1VirtualMapper()
    reader = FakeStateReader(left_q=1.375, right_q=4.125)
    driver = FakeDriver()
    bridge = Dex1PolicyBridge(mapper=mapper, state_reader=reader, driver=driver)
    source = {
        "body_q": np.arange(29, dtype=np.float64),
        "left_hand_q": np.zeros(7),
        "right_hand_q": np.zeros(7),
    }

    patched = bridge.patch_state_message(source)

    expected_left, expected_right = mapper.hand7_pair(0.25, 0.75)
    np.testing.assert_allclose(patched["left_hand_q"], expected_left)
    np.testing.assert_allclose(patched["right_hand_q"], expected_right)
    np.testing.assert_array_equal(patched["body_q"], source["body_q"])
    assert patched is not source

    command = bridge.send_hand14(np.concatenate([expected_left, expected_right]))

    assert len(driver.commands) == 1
    assert command is driver.commands[0]
    assert abs(command.left - 0.25) < 1e-5
    assert abs(command.right - 0.75) < 1e-5


def test_missing_dex1_state_blocks_model_observation():
    mapper = Dex1VirtualMapper()
    bridge = Dex1PolicyBridge(
        mapper=mapper,
        state_reader=FakeStateReader(left_q=None, right_q=4.125),
        driver=SimpleNamespace(send=lambda _command: None),
    )

    try:
        bridge.patch_state_message({"body_q": np.zeros(29)})
    except RuntimeError as exc:
        assert "Dex1 state" in str(exc)
    else:
        raise AssertionError("missing Dex1 state was accepted")


def test_state_subscriber_patches_every_history_frame():
    mapper = Dex1VirtualMapper()
    bridge = Dex1PolicyBridge(
        mapper=mapper,
        state_reader=FakeStateReader(left_q=1.375, right_q=4.125),
        driver=FakeDriver(),
    )

    class FakeSubscriber:
        def get_state(self):
            return {"body_q": np.zeros(29)}

        def get_all_states(self):
            return [{"body_q": np.full(29, i)} for i in range(3)]

    subscriber = Dex1StateSubscriberAdapter(FakeSubscriber(), bridge)

    latest = subscriber.get_state()
    history = subscriber.get_all_states()

    assert latest["left_hand_q"].shape == (7,)
    assert latest["right_hand_q"].shape == (7,)
    assert len(history) == 3
    for index, frame in enumerate(history):
        np.testing.assert_array_equal(frame["body_q"], np.full(29, index))
        assert frame["left_hand_q"].shape == (7,)
        assert frame["right_hand_q"].shape == (7,)


def test_action_router_sends_dex1_and_keeps_only_motion_token_for_sonic():
    mapper = Dex1VirtualMapper()
    driver = FakeDriver()
    bridge = Dex1PolicyBridge(
        mapper=mapper,
        state_reader=FakeStateReader(left_q=1.375, right_q=4.125),
        driver=driver,
    )
    router = Dex1ActionRouter(bridge)
    hand14 = mapper.hand14(0.2, 0.8)
    action = np.concatenate([np.arange(64, dtype=np.float32), hand14])

    sonic_action = router.route(action)

    np.testing.assert_array_equal(sonic_action[:64], np.arange(64))
    np.testing.assert_array_equal(sonic_action[64:], np.zeros(14))
    assert len(driver.commands) == 1
    assert abs(driver.commands[0].left - 0.2) < 1e-5
    assert abs(driver.commands[0].right - 0.8) < 1e-5


def test_official_client_extension_routes_state_and_action():
    mapper = Dex1VirtualMapper()
    driver = FakeDriver()
    bridge = Dex1PolicyBridge(
        mapper=mapper,
        state_reader=FakeStateReader(left_q=1.375, right_q=4.125),
        driver=driver,
    )

    class OfficialStateSubscriber:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_state(self):
            return {"body_q": np.zeros(29)}

        def get_all_states(self):
            return [{"body_q": np.zeros(29)}]

    class OfficialRTCClient:
        def __init__(self, token_publisher):
            self._token_publisher = token_publisher

    class FakeTokenPublisher:
        def __init__(self):
            self.actions = []

        def publish_token(self, action):
            self.actions.append(action.copy())

    official = SimpleNamespace(
        RobotStateSubscriber=OfficialStateSubscriber,
        RTCWebSocketClient=OfficialRTCClient,
        fsq_quantize=lambda token: token,
    )
    install_dex1_adapters(official, bridge)
    subscriber = official.RobotStateSubscriber()
    token_publisher = FakeTokenPublisher()
    client = official.RTCWebSocketClient(token_publisher)
    hand14 = mapper.hand14(0.2, 0.8)

    state = subscriber.get_state()
    client.execute_action(np.concatenate([np.arange(64), hand14]))

    assert state["left_hand_q"].shape == (7,)
    assert state["right_hand_q"].shape == (7,)
    assert len(driver.commands) == 1
    assert len(token_publisher.actions) == 1
    np.testing.assert_array_equal(token_publisher.actions[0][:64], np.arange(64))
    np.testing.assert_array_equal(token_publisher.actions[0][64:], np.zeros(14))
