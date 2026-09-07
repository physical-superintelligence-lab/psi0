"""Reversible mapping between a virtual Dex3 hand14 and Dex1 openness."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

HAND_DIM = 14          # 左7 + 右7
LEFT_SLICE = (0, 7)
RIGHT_SLICE = (7, 14)
DEFAULT_RANGE_THRESHOLD = 0.3   # rad;量程小于此的关节视为"基本不动",不计入抓握


@dataclass(frozen=True)
class Dex11Command:
    left: float
    right: float


@dataclass
class Dex3GraspMap:
    """单手 7 关节 → openness[0,1] 的标定。"""

    j_min: np.ndarray              # (7,) 数据集该手 7 关节 min
    j_max: np.ndarray              # (7,)
    active: np.ndarray             # (7,) bool,参与抓握聚合的关节
    invert: bool = False           # True: 翻转开/合方向(待真机标定)

    def openness(self, hand7) -> float:
        h = np.asarray(hand7, np.float32)
        if h.shape[-1] != 7:
            raise ValueError(f"hand7 must be length 7, got {h.shape}")
        rng = np.maximum(self.j_max - self.j_min, 1e-6)
        frac = np.clip((h - self.j_min) / rng, 0.0, 1.0)   # 每关节在量程内的位置 [0,1]
        sel = self.active if self.active.any() else np.ones(7, bool)
        g = float(frac[sel].mean())                        # 聚合 = 平均位置
        return float(1.0 - g) if self.invert else g

    def hand7_from_openness(self, openness: float) -> np.ndarray:
        """反向:openness[0,1] -> 7 维 Dex3 手关节(部署时把 Dex1 夹爪状态填成假 Dex3 手状态)。

        活动关节按 openness 在量程内插值;非活动关节置量程中点(基本不动)。
        与 openness() 互逆:openness(hand7_from_openness(o)) ≈ o。
        """
        o = float(np.clip(openness, 0.0, 1.0))
        frac = (1.0 - o) if self.invert else o             # 每关节归一位置
        h = 0.5 * (self.j_min + self.j_max)                # 非活动关节用中点
        act = self.active if self.active.any() else np.ones(7, bool)
        h = np.where(act, self.j_min + frac * (self.j_max - self.j_min), h)
        return h.astype(np.float32)


@dataclass
class Dex3ToDex1:
    left: Dex3GraspMap
    right: Dex3GraspMap

    def hand14_to_command(self, hand14) -> Dex11Command:
        """将明确的虚拟 Dex3 hand14 映射回 Dex1 开度。

        与 :meth:`to_command` 不同，这个接口不会假设 hand14 位于向量开头，
        适用于 SONIC ``action[64:78]`` 和 ``state[29:43]``。
        """
        hand = np.asarray(hand14, np.float32)
        if hand.shape != (HAND_DIM,):
            raise ValueError(f"hand14 must have shape ({HAND_DIM},), got {hand.shape}")
        if not np.all(np.isfinite(hand)):
            raise ValueError("hand14 contains NaN or Inf")
        return Dex11Command(
            left=self.left.openness(hand[LEFT_SLICE[0]:LEFT_SLICE[1]]),
            right=self.right.openness(hand[RIGHT_SLICE[0]:RIGHT_SLICE[1]]),
        )

    def state_hand14(self, left_openness: float, right_openness: float) -> np.ndarray:
        """反向:左右夹爪 openness -> 14 维 Dex3 手状态(部署时填进模型输入 states)。"""
        return np.concatenate([
            self.left.hand7_from_openness(left_openness),
            self.right.hand7_from_openness(right_openness),
        ]).astype(np.float32)


def build_from_stats(action_min, action_max, *, range_threshold: float = DEFAULT_RANGE_THRESHOLD,
                     invert_left: bool = False, invert_right: bool = False) -> Dex3ToDex1:
    """从 action 的 per-dim min/max(>=14 维)构建左右手塌缩标定。"""
    amin = np.asarray(action_min, np.float32)
    amax = np.asarray(action_max, np.float32)
    if amin.shape[0] < HAND_DIM:
        raise ValueError(f"action stats must have >= {HAND_DIM} dims")
    rng = amax - amin

    def mk(s):
        jmin, jmax = amin[s[0]:s[1]], amax[s[0]:s[1]]
        active = (rng[s[0]:s[1]] > range_threshold)
        return jmin, jmax, active

    lmin, lmax, lact = mk(LEFT_SLICE)
    rmin, rmax, ract = mk(RIGHT_SLICE)
    return Dex3ToDex1(
        left=Dex3GraspMap(lmin, lmax, lact, invert_left),
        right=Dex3GraspMap(rmin, rmax, ract, invert_right),
    )


def load_from_stats_file(stats_path: str, **kw) -> Dex3ToDex1:
    with open(stats_path, encoding="utf-8") as f:
        s = json.load(f)
    a = s["action"]
    return build_from_stats(a["min"], a["max"], **kw)
