"""Unitree Dex1-1 双夹爪 DDS 控制器。

接口取自 PC2 上的 `dex1_1_service`(serial2dds,unitreerobotics/dex1_1_service):

* 右爪 motor_id=0:cmd `rt/dex1/right/cmd`,state `rt/dex1/right/state`
* 左爪 motor_id=1:cmd `rt/dex1/left/cmd`,state `rt/dex1/left/state`
* 消息类型:unitree_go IDL `MotorCmds_` / `MotorStates_`,左右各为独立 topic,
  每条消息只含 1 个电机(用 `cmds[0]` / `states[0]`)。
* 命令字段:`cmds[0].{mode, kp, kd, q, dq, tau}`;服务端内部按 gear_ratio 换算,
  用户侧 `q` 为夹爪输出空间弧度。官方测试用 kp=5.0 kd=0.05。
* q 范围(夹爪空间 rad):标定零 q≈0=闭合,张开 ~5.5(标定上界 322°≈5.62)。
* 服务端 cmd 订阅超时会让电机进入 BRAKE,故需后台**持续发布**当前目标以保持位姿。

本控制器只负责「目标 q -> DDS 持续发布 + 状态回读」。归一化([0,1])与方向标定
由上层 `scripts/offline/real_dex1_1_client.py` 的 RealDex11Driver 负责。
"""

from __future__ import annotations

import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import (  # dds
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl

# side -> (cmd topic, state topic);右=motor_id 0,左=motor_id 1
TOPIC_CMD = {"right": "rt/dex1/right/cmd", "left": "rt/dex1/left/cmd"}
TOPIC_STATE = {"right": "rt/dex1/right/state", "left": "rt/dex1/left/state"}

# 夹爪空间安全范围(rad)。详见模块 docstring。
Q_MIN = 0.0
Q_MAX = 5.5

DEFAULT_KP = 5.0
DEFAULT_KD = 0.05
DEFAULT_FPS = 100.0


class Dex1_1_Controller:
    """双爪 DDS 控制器:后台线程以固定频率持续发布目标 q,并回读状态。"""

    def __init__(
        self,
        network: str | None = "enp4s0",
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        fps: float = DEFAULT_FPS,
        q_min: float = Q_MIN,
        q_max: float = Q_MAX,
        init_dds: bool = True,
        wait_state_timeout: float = 5.0,
    ):
        print("Initialize Dex1_1_Controller...")
        self.kp = float(kp)
        self.kd = float(kd)
        self.fps = float(fps)
        self.q_min = float(q_min)
        self.q_max = float(q_max)

        if init_dds:
            ChannelFactoryInitialize(0, network)

        # 左右各一对 publisher / subscriber(独立 topic)
        self._pub = {}
        self._sub = {}
        for side in ("left", "right"):
            pub = ChannelPublisher(TOPIC_CMD[side], MotorCmds_)
            pub.Init()
            self._pub[side] = pub
            sub = ChannelSubscriber(TOPIC_STATE[side], MotorStates_)
            sub.Init()
            self._sub[side] = sub

        # 共享状态/目标(rad)
        self._lock = threading.Lock()
        self._state_q = {"left": None, "right": None}
        self._target_q = {"left": None, "right": None}
        self._publishing = False  # 是否真正向总线发布(安全门)

        # 状态订阅线程
        self._state_thread = threading.Thread(target=self._subscribe_state, daemon=True)
        self._state_thread.start()

        # 等首帧状态,拿到当前位作为初始目标,避免上电瞬间跳变
        t0 = time.time()
        while time.time() - t0 < wait_state_timeout:
            with self._lock:
                if self._state_q["left"] is not None and self._state_q["right"] is not None:
                    self._target_q["left"] = self._state_q["left"]
                    self._target_q["right"] = self._state_q["right"]
                    break
            time.sleep(0.01)
            print("[Dex1_1_Controller] waiting dds state...")
        else:
            # 没收到状态也给安全默认(闭合),但不自动发布
            with self._lock:
                self._target_q["left"] = self.q_min
                self._target_q["right"] = self.q_min
            print("[Dex1_1_Controller] WARNING: 未收到状态,目标置为闭合(q_min),未发布。")

        # 发布线程(默认不发布,需 start_publishing() 显式开启)
        self._running = True
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()
        print("Initialize Dex1_1_Controller OK!\n")

    # ---- 状态 ----
    def _subscribe_state(self):
        while True:
            for side in ("left", "right"):
                msg = self._sub[side].Read()
                if msg is not None and len(msg.states) > 0:
                    with self._lock:
                        self._state_q[side] = float(msg.states[0].q)
            time.sleep(0.002)

    def get_current_dual_gripper_q(self) -> tuple[float | None, float | None]:
        with self._lock:
            return self._state_q["left"], self._state_q["right"]

    # ---- 目标 / 发布 ----
    def _clip(self, q: float) -> float:
        return float(np.clip(q, self.q_min, self.q_max))

    def ctrl_dual_gripper(self, left_q: float, right_q: float):
        """设置左右目标 q(rad)。仅更新目标;实际发布由后台线程完成。"""
        with self._lock:
            self._target_q["left"] = self._clip(left_q)
            self._target_q["right"] = self._clip(right_q)

    def start_publishing(self):
        """开启向总线持续发布目标(安全门:默认关闭)。"""
        with self._lock:
            self._publishing = True
        print("[Dex1_1_Controller] publishing ENABLED")

    def stop_publishing(self):
        """停止发布。服务端 cmd 超时后电机进入 BRAKE(安全)。"""
        with self._lock:
            self._publishing = False
        print("[Dex1_1_Controller] publishing DISABLED (servo will BRAKE on timeout)")

    def _make_cmd(self, q: float) -> MotorCmds_:
        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        c = msg.cmds[0]
        c.mode = 1
        c.kp = self.kp
        c.kd = self.kd
        c.q = float(q)
        c.dq = 0.0
        c.tau = 0.0
        return msg

    def _control_loop(self):
        period = 1.0 / self.fps
        while self._running:
            start = time.time()
            with self._lock:
                publishing = self._publishing
                tl = self._target_q["left"]
                tr = self._target_q["right"]
            if publishing and tl is not None and tr is not None:
                self._pub["left"].Write(self._make_cmd(tl))
                self._pub["right"].Write(self._make_cmd(tr))
            time.sleep(max(0.0, period - (time.time() - start)))

    def reset(self):
        """Resume publishing for a new episode without restarting the worker thread.

        Latch the latest measured positions before enabling DDS writes so an episode
        transition cannot replay the previous episode's final target.
        """
        self.stop_publishing()
        with self._lock:
            for side in ("left", "right"):
                state_q = self._state_q[side]
                if state_q is not None:
                    self._target_q[side] = state_q
        self.start_publishing()

    def close(self):
        self.stop_publishing()
        self._running = False
        print("Dex1_1_Controller closed.")
