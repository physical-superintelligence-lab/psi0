"""dex3_to_dex1 塌缩纯逻辑单测(不依赖 DDS/模型)。"""
import unittest

import numpy as np

from scripts.offline.dex3_to_dex1 import build_from_stats, Dex3ToDex1


def _stats():
    # 14 维 action 手部:左[0:7] 右[7:14]
    # 关节 3-6 / 10-13 给大量程(活动);0-2 / 7-9 给小量程(基本不动)
    amin = np.zeros(14, np.float32)
    amax = np.zeros(14, np.float32)
    for s in (0, 7):  # 每手
        amax[s + 0] = 0.1; amax[s + 1] = 0.1; amax[s + 2] = 0.1     # 小量程 -> 不活动
        for j in (3, 4, 5, 6):
            amin[s + j] = 0.0; amax[s + j] = 1.7                    # 大量程 -> 活动
    return amin, amax


class TestBuild(unittest.TestCase):
    def test_active_joints_by_range(self):
        m = build_from_stats(*_stats(), range_threshold=0.3)
        self.assertEqual(np.where(m.left.active)[0].tolist(), [3, 4, 5, 6])
        self.assertEqual(np.where(m.right.active)[0].tolist(), [3, 4, 5, 6])


class TestOpenness(unittest.TestCase):
    def setUp(self):
        self.m: Dex3ToDex1 = build_from_stats(*_stats(), range_threshold=0.3)

    def test_endpoints(self):
        amin, amax = _stats()
        # 手在 min(活动关节=0)-> openness 0;在 max(活动关节=1.7)-> openness 1
        self.assertAlmostEqual(self.m.left.openness(amin[0:7]), 0.0, places=5)
        self.assertAlmostEqual(self.m.left.openness(amax[0:7]), 1.0, places=5)

    def test_monotonic(self):
        amax = _stats()[1][0:7]
        vals = [self.m.left.openness(amax * f) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
        self.assertTrue(all(vals[i] <= vals[i + 1] + 1e-6 for i in range(len(vals) - 1)))

    def test_invert(self):
        amin, amax = _stats()
        mi = build_from_stats(amin, amax, range_threshold=0.3, invert_left=True)
        self.assertAlmostEqual(mi.left.openness(amin[0:7]), 1.0, places=5)
        self.assertAlmostEqual(mi.left.openness(amax[0:7]), 0.0, places=5)

    def test_clip_out_of_range(self):
        amax = _stats()[1][0:7]
        self.assertAlmostEqual(self.m.left.openness(amax * 2.0), 1.0, places=5)  # 超出 -> 钳到 1


class TestInverse(unittest.TestCase):
    def setUp(self):
        self.m = build_from_stats(*_stats(), range_threshold=0.3)

    def test_roundtrip_openness(self):
        for o in (0.0, 0.25, 0.5, 0.8, 1.0):
            h = self.m.left.hand7_from_openness(o)
            self.assertEqual(h.shape, (7,))
            self.assertAlmostEqual(self.m.left.openness(h), o, places=4)

    def test_roundtrip_invert(self):
        mi = build_from_stats(*_stats(), range_threshold=0.3, invert_left=True)
        for o in (0.0, 0.5, 1.0):
            h = mi.left.hand7_from_openness(o)
            self.assertAlmostEqual(mi.left.openness(h), o, places=4)

    def test_state_hand14(self):
        s = self.m.state_hand14(1.0, 0.0)
        self.assertEqual(s.shape, (14,))
        self.assertAlmostEqual(self.m.left.openness(s[:7]), 1.0, places=4)
        self.assertAlmostEqual(self.m.right.openness(s[7:]), 0.0, places=4)

if __name__ == "__main__":
    unittest.main()
