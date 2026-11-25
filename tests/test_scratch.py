"""部署规格（scale）下的内存上限与分块调度。"""
import unittest

import numpy as np

from cracklib.io import config_from_record
from cracklib.model import ProcessConfig, Window
from cracklib.morphology import AllocationTracker, ChunkedClosingScheduler
from cracklib.store import ObservationStore
from crack_morphology_input import load_record


class ScratchBudgetTests(unittest.TestCase):
    def test_record_scratch_budget_is_honored(self):
        record = load_record()
        cfg = config_from_record(record)
        self.assertIsNotNone(cfg.scratch_bytes)
        # 小样本下 scratch 上限远大于整层，调度仍应正常完成且不越界。
        store = ObservationStore(cfg)
        for item in record["windows"]:
            from cracklib.io import window_from_record
            store.upsert_window(window_from_record(item))
        tracker = AllocationTracker()
        sched = ChunkedClosingScheduler(cfg, store, allocator=tracker)
        results = sched.run_all()
        self.assertEqual(set(results), set(range(cfg.num_layers)))
        self.assertLessEqual(tracker.peak, cfg.scratch_bytes)

    def test_large_grid_never_materializes_full_tensor(self):
        # 模拟 scale 规格：12 层、540x960，但只给一个很小的 scratch 上限。
        nr, nc, nlayers = 64, 96, 3
        scratch = 64 * 1024  # 64 KiB，远小于整层 (64*96*17 字节 ≈ 104 KiB)
        base = config_from_record(load_record())
        layers = base.layers[:1] * nlayers
        cfg = ProcessConfig(
            shape=(nlayers, nr, nc), ranges=base.ranges, outside=base.outside,
            layers=layers, vertical_offsets=base.vertical_offsets,
            faces={"source": (), "sink": ()}, scratch_bytes=scratch,
        )
        store = ObservationStore(cfg)
        rng = np.random.default_rng(5)
        # 只放两个小窗口，证明调度器按需物化而非拼整张量。
        for layer in range(nlayers):
            patch = rng.random((8, 8))
            store.upsert_window(Window(f"p{layer}", 1, layer, (4, 4),
                                       patch, patch.copy()))
        tracker = AllocationTracker()
        sched = ChunkedClosingScheduler(cfg, store, allocator=tracker)
        result = sched.run_layer(0)
        self.assertEqual(result.lower.shape, (nr, nc))
        self.assertLessEqual(tracker.peak, scratch)
        # 条带高度确实被预算压低（没有一次性吃下整层）。
        band = sched._fused_band_rows(nc, *cfg.layer(0).support_radius())
        self.assertLess(band, nr)

    def test_tight_budget_forces_many_bands_but_keeps_equivalence(self):
        record = load_record()
        cfg = config_from_record(record)
        store = ObservationStore(cfg)
        from cracklib.io import window_from_record
        for item in record["windows"]:
            store.upsert_window(window_from_record(item))
        # 极小预算下的两遍调度仍须与融合结果逐格点一致。
        ref_tracker = AllocationTracker()
        ref = ChunkedClosingScheduler(cfg, store, allocator=ref_tracker).run_all()
        tight_tracker = AllocationTracker()
        tight = ChunkedClosingScheduler(cfg, store, scratch_bytes=2048,
                                        allocator=tight_tracker, two_pass=True)
        out = tight.run_all()
        for layer in ref:
            np.testing.assert_allclose(out[layer].lower, ref[layer].lower)
            np.testing.assert_allclose(out[layer].upper, ref[layer].upper)
            np.testing.assert_array_equal(out[layer].blocked, ref[layer].blocked)
        self.assertLessEqual(tight_tracker.peak, 2048)


if __name__ == "__main__":
    unittest.main()
