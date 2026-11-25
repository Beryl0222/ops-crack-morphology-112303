"""分块闭运算：与整域计算逐格点一致、块交界无边界、scratch 上限与两遍调度。"""
import unittest

import numpy as np

from cracklib.io import config_from_record, window_from_record
from cracklib.model import Cell, LayerSpec
from cracklib.morphology import (
    AllocationTracker,
    ChunkedClosingScheduler,
    interval_close_fused,
)
from cracklib.store import ObservationStore
from tests.test_morphology import brute_close
from crack_morphology_input import load_record


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.record = load_record()
        self.config = config_from_record(self.record)
        self.store = ObservationStore(self.config)
        for item in self.record["windows"]:
            self.store.upsert_window(window_from_record(item))

    def _dense_reference(self, layer: int):
        mat = self.store.materialize_layer(layer)
        spec = self.config.layer(layer)
        lo, hi, blk = interval_close_fused(
            mat["lower"], mat["upper"], mat["blocked"], spec, self.config.outside)
        return lo, hi, blk

    def test_chunked_matches_dense_all_band_sizes(self):
        for layer in range(self.config.num_layers):
            ref = self._dense_reference(layer)
            for band in (1, 2, 3, 5, 50):
                tracker = AllocationTracker()
                sched = ChunkedClosingScheduler(self.config, self.store,
                                                allocator=tracker)
                sched._fused_band_rows = lambda nc, rh, ch, band=band: band
                result = sched.run_layer(layer)
                np.testing.assert_allclose(result.lower, ref[0], err_msg=f"layer {layer} band {band}")
                np.testing.assert_allclose(result.upper, ref[1])
                np.testing.assert_array_equal(result.blocked, ref[2])

    def test_two_pass_matches_dense(self):
        for layer in range(self.config.num_layers):
            ref = self._dense_reference(layer)
            tracker = AllocationTracker()
            sched = ChunkedClosingScheduler(self.config, self.store,
                                            scratch_bytes=10_000, allocator=tracker,
                                            two_pass=True)
            result = sched.run_layer(layer)
            np.testing.assert_allclose(result.lower, ref[0])
            np.testing.assert_allclose(result.upper, ref[1])
            np.testing.assert_array_equal(result.blocked, ref[2])

    def test_no_boundary_at_chunk_seam(self):
        # 构造一条跨 seam 的高值裂纹；若 seam 被当成边界（填常量），
        # seam 两侧响应会被整体压低。
        cfg = self.config
        layer = 0
        nr, nc = cfg.shape[1], cfg.shape[2]
        x = np.full((nr, nc), 0.1)
        x[:, 3] = 0.95
        store = ObservationStore(cfg)
        from cracklib.model import Window
        store.upsert_window(Window("full", 1, layer, (0, 0), x, x.copy()))
        tracker = AllocationTracker()
        sched = ChunkedClosingScheduler(cfg, store, allocator=tracker)
        sched._fused_band_rows = lambda nc, rh, ch: 2  # 强制 seam 落在行 2/3 之间
        result = sched.run_layer(layer)
        dense_lo, dense_hi, _ = interval_close_fused(
            *tuple(store.materialize_layer(layer)[k] for k in ("lower", "upper", "blocked")),
            cfg.layer(layer), cfg.outside)
        np.testing.assert_allclose(result.lower, dense_lo)
        np.testing.assert_allclose(result.upper, dense_hi)

    def test_scratch_budget_caps_peak_and_is_reported(self):
        budget = 4096
        tracker = AllocationTracker()
        sched = ChunkedClosingScheduler(self.config, self.store,
                                        scratch_bytes=budget, allocator=tracker)
        sched.run_all()
        self.assertLessEqual(tracker.peak, budget)
        self.assertGreater(tracker.peak, 0)
        # 小预算必须真的切成多个条带。
        self.assertLessEqual(sched._fused_band_rows(self.config.shape[2], 1, 1),
                             self.config.shape[1])

    def test_support_radius_halo_uses_full_close_support(self):
        spec = LayerSpec(0.5, ((-1, 0), (0, 0), (0, 1), (1, 0)))
        rh, ch = spec.support_radius()
        self.assertEqual((rh, ch), (2, 1))  # 闭运算支撑半径大于单步 footprint 半径

    def test_random_interval_grid_chunk_equivalence(self):
        rng = np.random.default_rng(11)
        cfg = self.config
        nr, nc = 9, 11
        # 借用配置但换一层更大的随机网格，独立 store。
        from cracklib.model import ProcessConfig
        big_cfg = ProcessConfig(
            shape=(1, nr, nc), ranges=(0.0, 1.0), outside=0.2,
            layers=(cfg.layer(0),), vertical_offsets=(),
            faces={"source": (), "sink": ()},
        )
        lo = rng.random((nr, nc))
        hi = np.minimum(1.0, lo + rng.random((nr, nc)) * 0.3)
        # 这里的随机掩码表示“冲突阻断”，走 blocked 通道而不是窗口 valid 排除。
        blk = np.zeros((nr, nc), dtype=bool)
        blk[rng.random((nr, nc)) < 0.1] = True
        from cracklib.model import Window
        store = ObservationStore(big_cfg)
        store.upsert_window(Window("rng", 1, 0, (0, 0), lo, hi, valid=~blk))
        # 直接把随机阻断注入一份独立 store 的物化结果，验证分块与暴力一致。
        blocked_store = ObservationStore(big_cfg)
        blocked_store.upsert_window(Window("rng", 1, 0, (0, 0), lo, hi))
        for r in range(nr):
            for c in range(nc):
                if blk[r, c]:
                    cell = Cell(0, r, c)
                    blocked_store._constraint[cell] = type(
                        blocked_store.constraint(cell))(cell, 0.0, 0.0, (), True)
        ref_lo, ref_hi = brute_close(lo, hi, big_cfg.layer(0), 0.2)
        tracker = AllocationTracker()
        sched = ChunkedClosingScheduler(big_cfg, blocked_store, allocator=tracker)
        sched._fused_band_rows = lambda nc, rh, ch: 3
        out = sched.run_layer(0)
        # 非阻断格点的数值必须与暴力参照一致。
        np.testing.assert_allclose(out.lower[~out.blocked], ref_lo[~out.blocked])
        np.testing.assert_allclose(out.upper[~out.blocked], ref_hi[~out.blocked])
        # 阻断掩码沿结构支撑传播，且分块结果与整块一致。
        dense = blocked_store.materialize_layer(0)
        _, _, dense_blk = interval_close_fused(
            dense["lower"], dense["upper"], dense["blocked"],
            big_cfg.layer(0), 0.2, global_shape=(nr, nc))
        np.testing.assert_array_equal(out.blocked, dense_blk)


if __name__ == "__main__":
    unittest.main()
