"""训练可微性：确定区域梯度、平局裁决、区间阻断与有限差分验证。"""
import unittest

import numpy as np

from cracklib.io import config_from_record, probe_from_record, window_from_record
from cracklib.model import Cell, ProcessConfig, Window
from cracklib.pipeline import CrackEngine
from cracklib.training import MorphologyTrainer
from crack_morphology_input import load_record


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.record = load_record()
        self.config = config_from_record(self.record)
        self.engine = CrackEngine.from_record(
            __import__("crack_morphology_input").SAMPLE)
        _ = self.engine.responses

    def test_probe_with_interval_dependency_is_not_trainable(self):
        plan = self.engine.training_plan(probe_from_record(self.record))
        self.assertFalse(plan.trainable)
        # 必须给出不能训练的具体原因与格点，而不是笼统报错。
        self.assertTrue(plan.reasons["uncertain"] or plan.reasons["blocked"])
        self.assertEqual(plan.gradients, {})

    def test_fully_determined_region_trains_and_matches_finite_difference(self):
        # 构造一个完全实测的小配方，probe 只依赖确定格点。
        spec = self.config.layer(0)
        nr, nc = 5, 6
        rng = np.random.default_rng(3)
        x = rng.random((nr, nc)) * 0.5 + 0.4
        cfg = ProcessConfig(
            shape=(1, nr, nc), ranges=(0.0, 1.0), outside=0.0,
            layers=(spec,), vertical_offsets=(),
            faces={"source": (Cell(0, 2, 0),), "sink": (Cell(0, 2, 5),)},
        )
        engine = CrackEngine(cfg)
        engine.upsert_window(Window("exact", 1, 0, (0, 0), x, x.copy()))
        probe = [(0, 2, 3, 1.0), (0, 3, 2, 0.7)]
        plan = engine.training_plan(probe)
        self.assertTrue(plan.trainable, plan.reasons)

        # 有限差分：对每个被选中的输入格点扰动，检查响应梯度之和。
        trainer = MorphologyTrainer(cfg, engine.store, engine.responses)
        eps = 1e-6

        def response_values(values):
            from tests.test_morphology import brute_close
            lo, hi = brute_close(values, values, spec, 0.0)
            return np.array([lo[r, c] for _, r, c, _ in probe])

        base = response_values(x)
        for cell, grad in plan.gradients.items():
            xp = x.copy()
            xp[cell.row, cell.col] += eps
            diff = (response_values(xp) - base) / eps
            upstream = np.array([g for _, _, _, g in probe])
            self.assertAlmostEqual(float(diff @ upstream), grad, places=5)

    def test_tie_break_uses_global_cell_order_and_is_unique_identity(self):
        spec = self.config.layer(0)
        nr, nc = 4, 4
        x = np.full((nr, nc), 0.5)
        cfg = ProcessConfig(
            shape=(1, nr, nc), ranges=(0.0, 1.0), outside=0.0,
            layers=(spec,), vertical_offsets=(),
            faces={"source": (), "sink": ()},
        )
        engine = CrackEngine(cfg)
        # 同一物理格点用两个重叠窗口各给一次相同确定值。
        engine.upsert_window(Window("w1", 1, 0, (0, 0), x, x.copy()))
        engine.upsert_window(Window("w2", 1, 0, (0, 0), x, x.copy()))
        trainer = MorphologyTrainer(cfg, engine.store, engine.responses)
        probe_cell = Cell(0, 1, 1)
        diag = trainer.check_cell(probe_cell)
        self.assertTrue(diag["trainable"])
        chain = trainer.route_cell(probe_cell)
        # 极值完全相同时按全局 C 序只选一个身份，重复窗口不产生重复贡献。
        self.assertEqual(len(chain), 1)
        expected = min(
            (Cell(0, r, c) for r in range(nr) for c in range(nc)),
            key=lambda cell: cell.linear_index(cfg.shape),
        )
        self.assertEqual(chain[0], expected)

    def test_boundary_constant_has_no_gradient(self):
        spec = self.config.layer(0)
        nr, nc = 3, 3
        x = np.full((nr, nc), 0.9)
        cfg = ProcessConfig(
            shape=(1, nr, nc), ranges=(0.0, 1.0), outside=0.0,
            layers=(spec,), vertical_offsets=(),
            faces={"source": (), "sink": ()},
        )
        engine = CrackEngine(cfg)
        engine.upsert_window(Window("edge", 1, 0, (0, 0), x, x.copy()))
        trainer = MorphologyTrainer(cfg, engine.store, engine.responses)
        # 角落格点的闭运算支撑含原图外常量；常量不产生输入格点梯度。
        plan = trainer.gradients([(0, 0, 0, 1.0)])
        self.assertTrue(plan.trainable)
        self.assertTrue(all(0 <= c.row < nr and 0 <= c.col < nc for c in plan.gradients))
        self.assertIn("boundary_constant", plan.provenance)

    def test_conflict_cell_blocks_training(self):
        engine = CrackEngine.from_record(
            __import__("crack_morphology_input").SAMPLE, use_conflict_window=True)
        plan = engine.training_plan([(0, 3, 2, 1.0)])
        self.assertFalse(plan.trainable)
        self.assertIn(Cell(0, 3, 2), plan.reasons["blocked"])


if __name__ == "__main__":
    unittest.main()
