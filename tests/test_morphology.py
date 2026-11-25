"""区间形态学原语的单元测试与暴力参照实现。"""
import unittest

import numpy as np

from cracklib.model import LayerSpec
from cracklib.morphology import (
    _gather,
    interval_close_fused,
    interval_dilate,
    interval_erode,
)


def brute_erode(lower, upper, offsets, outside):
    """输出 (r,c) 取输入 (r+offset)，腐蚀 = min。"""
    h, w = lower.shape
    lo = np.full((h, w), np.inf)
    hi = np.full((h, w), np.inf)
    for r in range(h):
        for c in range(w):
            for dr, dc in offsets:
                r2, c2 = r + dr, c + dc
                a = outside if not (0 <= r2 < h and 0 <= c2 < w) else lower[r2, c2]
                b = outside if not (0 <= r2 < h and 0 <= c2 < w) else upper[r2, c2]
                lo[r, c] = min(lo[r, c], a)
                hi[r, c] = min(hi[r, c], b)
    return lo, hi


def brute_dilate(lower, upper, offsets, outside):
    """膨胀在反射偏移上 = max。"""
    h, w = lower.shape
    lo = np.full((h, w), -np.inf)
    hi = np.full((h, w), -np.inf)
    for r in range(h):
        for c in range(w):
            for dr, dc in offsets:
                r2, c2 = r + dr, c + dc
                a = outside if not (0 <= r2 < h and 0 <= c2 < w) else lower[r2, c2]
                b = outside if not (0 <= r2 < h and 0 <= c2 < w) else upper[r2, c2]
                lo[r, c] = max(lo[r, c], a)
                hi[r, c] = max(hi[r, c], b)
    return lo, hi


def brute_close(lower, upper, spec: LayerSpec, outside):
    e_lo, e_hi = brute_erode(lower, upper, spec.footprint, outside)
    d_lo, d_hi = brute_dilate(e_lo, e_hi, spec.reflected_offsets, outside)
    return d_lo, d_hi


class PrimitiveTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.lower = rng.random((6, 8))
        width = rng.random((6, 8)) * 0.2
        self.upper = np.minimum(1.0, self.lower + width)
        self.blocked = np.zeros((6, 8), dtype=bool)
        self.blocked[2, 3] = True
        self.spec = LayerSpec(0.5, ((-1, 0), (0, 0), (0, 1), (1, 0)))
        self.outside = 0.13

    def test_gather_origin_and_outside(self):
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        shifted = _gather(arr, (0, 1), (0, 0), fill=-7.0, global_shape=(3, 4))
        # 输出 (r,c) 取输入 (r,c+1)，最后一列落原图外。
        self.assertEqual(shifted[0, 0], arr[0, 1])
        self.assertEqual(shifted[2, 3], -7.0)
        # 带 halo 的画布：左侧多一列，origin=(0,-1)，输出第 0 列引用全局列 0。
        halo = np.pad(arr, ((0, 0), (1, 0)), constant_values=-9.0)
        shifted_inside = _gather(halo, (0, 1), (0, -1), fill=-7.0,
                                 global_shape=(3, 4))
        self.assertEqual(shifted_inside[0, 0], arr[0, 0])
        self.assertEqual(shifted_inside[2, 0], arr[2, 0])

    def test_erode_matches_brute_force(self):
        lo, hi, blk = interval_erode(self.lower, self.upper, self.blocked,
                                     self.spec, self.outside)
        ref_lo, ref_hi = brute_erode(self.lower, self.upper, self.spec.footprint, self.outside)
        np.testing.assert_allclose(lo, ref_lo)
        np.testing.assert_allclose(hi, ref_hi)
        # 阻断掩码沿支撑传播：(2,3) 被阻断，其邻居输出也阻断。
        self.assertTrue(blk[2, 2] and blk[1, 3] and blk[3, 3])
        self.assertFalse(blk[0, 0])

    def test_dilate_matches_brute_force(self):
        lo, hi, _ = interval_dilate(self.lower, self.upper, self.blocked,
                                    self.spec, self.outside)
        ref_lo, ref_hi = brute_dilate(self.lower, self.upper,
                                      self.spec.reflected_offsets, self.outside)
        np.testing.assert_allclose(lo, ref_lo)
        np.testing.assert_allclose(hi, ref_hi)

    def test_close_matches_brute_force_and_keeps_interval_order(self):
        lo, hi, blk = interval_close_fused(self.lower, self.upper, self.blocked,
                                           self.spec, self.outside)
        ref_lo, ref_hi = brute_close(self.lower, self.upper, self.spec, self.outside)
        np.testing.assert_allclose(lo, ref_lo)
        np.testing.assert_allclose(hi, ref_hi)
        self.assertTrue(np.all(lo <= hi + 1e-12))
        # 外边界常量参与所有 min/max，响应值不会越出 [outside, 1] 的数据范围。
        self.assertTrue(np.all(lo[~blk] >= min(self.outside, 0.0) - 1e-12))
        self.assertTrue(np.all(hi[~blk] <= 1.0 + 1e-12))

    def test_exact_input_collapses_to_classical_close(self):
        # 上下界相同（完全实测）时，区间闭运算退化为经典灰度闭运算。
        # footprint={(0,0),(0,1)}，反射={(0,0),(0,-1)}；outside=0。
        spec = LayerSpec(0.5, ((0, 0), (0, 1)))
        x = np.array([[0.9, 0.2, 0.8], [0.1, 0.7, 0.3]])
        lo, hi, _ = interval_close_fused(x, x.copy(), np.zeros_like(x, bool),
                                         spec, outside=0.0)
        ref_lo, ref_hi = brute_close(x, x, spec, 0.0)
        np.testing.assert_allclose(lo, hi)
        np.testing.assert_allclose(lo, ref_lo)
        # 手算 (0,0)：腐蚀取 min(x(0,0),x(0,1))=0.2；
        # 膨胀在 (0,0) 取反射 max(e(0,0),e(0,-1)=outside=0)=0.2。
        self.assertAlmostEqual(lo[0, 0], 0.2)
        # (0,1)：e(0,1)=min(0.2,0.8)=0.2，膨胀取 max(e(0,1),e(0,0))=0.2。
        self.assertAlmostEqual(lo[0, 1], 0.2)


if __name__ == "__main__":
    unittest.main()
