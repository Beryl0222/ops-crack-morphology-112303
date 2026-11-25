"""观测存储：交集、冲突、整窗替换与物化。"""
import unittest

import numpy as np

from cracklib.io import config_from_record, window_from_record
from cracklib.model import Cell, Window
from cracklib.store import ObservationStore
from crack_morphology_input import load_record


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.record = load_record()
        self.config = config_from_record(self.record)
        self.store = ObservationStore(self.config)

    def test_overlap_intersection_and_default_ranges(self):
        w0 = window_from_record(self.record["windows"][0])  # id a, origin (1,0)
        self.store.upsert_window(w0)
        cons = self.store.constraint(Cell(0, 1, 0))  # a 的 (0,0)=0.9 确定格点
        self.assertTrue(cons.is_exact())
        self.assertEqual(cons.sources[0][0], "a")
        # 未被任何窗口覆盖的格点退化为物理允许区间。
        untouched = self.store.constraint(Cell(0, 0, 6))
        self.assertEqual((untouched.lower, untouched.upper), self.config.ranges)
        self.assertEqual(untouched.sources, ())

        w1 = window_from_record(self.record["windows"][1])  # id b rev1, origin (2,2)
        self.store.upsert_window(w1)
        # 重叠点 (0,2,2)：a(1,2)=[0,1] 缺测，b(0,0)=[0.3,0.67] -> 交集 [0.3,0.67]
        cons = self.store.constraint(Cell(0, 2, 2))
        self.assertAlmostEqual(cons.lower, 0.3)
        self.assertAlmostEqual(cons.upper, 0.67)
        self.assertFalse(cons.is_exact())
        self.assertEqual(len(cons.sources), 2)
        # 重叠点 (0,2,3)：两侧都给确定值 0.9，交集仍确定且双来源。
        exact = self.store.constraint(Cell(0, 2, 3))
        self.assertTrue(exact.is_exact())
        self.assertEqual(len(exact.sources), 2)

    def test_synthetic_intersection_empty_is_local_conflict(self):
        # 两个小窗口在同一格点给出不相交区间，冲突只标记该格点。
        self.store.upsert_window(Window(
            "x", 1, 0, (0, 0), np.array([[0.9, 0.2]]), np.array([[0.9, 0.2]])))
        fresh = self.store.upsert_window(Window(
            "y", 1, 0, (0, 0), np.array([[0.1, 0.2]]), np.array([[0.2, 0.2]])))
        self.assertEqual(fresh, (Cell(0, 0, 0),))
        self.assertEqual(self.store.conflict_cells(), (Cell(0, 0, 0),))
        self.assertTrue(self.store.constraint(Cell(0, 0, 0)).conflict)
        # 相邻独立格点不受影响。
        self.assertFalse(self.store.constraint(Cell(0, 0, 1)).conflict)

    def test_window_replacement_recomputes_constraint(self):
        for item in (self.record["windows"][0], self.record["windows"][1]):
            self.store.upsert_window(window_from_record(item))
        cons = self.store.constraint(Cell(0, 2, 2))
        self.assertAlmostEqual(cons.lower, 0.3)
        v_before = self.store.version
        # 用同 id 的 rev2 整窗替换 rev1：(0,2,2) 收紧为确定值 0.99。
        self.store.upsert_window(window_from_record(self.record["windows"][2]))
        self.assertGreater(self.store.version, v_before)
        cons = self.store.constraint(Cell(0, 2, 2))
        self.assertTrue(cons.is_exact())
        self.assertAlmostEqual(cons.lower, 0.99)
        source_ids = {src[0]: src[1] for src in cons.sources}
        self.assertEqual(source_ids.get("b"), 2)
        # a 的缺测区间 [0,1] 与确定值 0.99 交集就是 0.99。
        self.assertEqual(len(cons.sources), 2)

    def test_conflict_materialization_marks_blocked_and_isolates_layers(self):
        for item in self.record["windows"]:
            self.store.upsert_window(window_from_record(item))
        self.assertEqual(self.store.conflict_cells(), ())
        self.store.upsert_window(window_from_record(self.record["alternate_conflict"]))
        self.assertEqual(self.store.conflict_cells(), (Cell(0, 3, 2),))

        mat = self.store.materialize_layer(0)
        self.assertTrue(mat["blocked"][3, 2])
        self.assertTrue(mat["observed"][3, 2])
        self.assertFalse(mat["blocked"][0, 0])
        # 层 1 独立区域不受层 0 冲突影响。
        mat1 = self.store.materialize_layer(1)
        self.assertFalse(np.any(mat1["blocked"]))

    def test_valid_mask_excludes_entries(self):
        w = window_from_record(self.record["windows"][0])
        valid = np.ones(w.shape, dtype=bool)
        valid[0, 0] = False
        masked = window_from_record({**self.record["windows"][0], "valid": valid.tolist()})
        self.store.upsert_window(masked)
        self.assertEqual(self.store.constraint(Cell(0, 1, 0)).sources, ())

    def test_alternate_conflict_window(self):
        self.store.upsert_window(window_from_record(self.record["windows"][0]))
        fresh = self.store.upsert_window(
            window_from_record(self.record["alternate_conflict"]))
        # d origin(3,2)=[0.07,0.18] 与 a 局部 (2,2) 全局 (0,3,2)=0.8 冲突。
        self.assertIn(Cell(0, 3, 2), fresh)
        self.assertTrue(self.store.constraint(Cell(0, 3, 2)).conflict)

    def test_remove_window_restores_prior_constraint(self):
        self.store.upsert_window(Window(
            "x", 1, 0, (0, 0), np.array([[0.5]]), np.array([[0.5]])))
        self.store.upsert_window(Window(
            "y", 1, 0, (0, 0), np.array([[0.7]]), np.array([[0.7]])))
        self.assertTrue(self.store.constraint(Cell(0, 0, 0)).conflict)
        self.store.remove_window("y")
        cons = self.store.constraint(Cell(0, 0, 0))
        self.assertFalse(cons.conflict)
        self.assertAlmostEqual(cons.lower, 0.5)
        self.store.remove_window("x")
        # 所有观测移除后回到物理允许默认区间。
        gone = self.store.constraint(Cell(0, 0, 0))
        self.assertEqual((gone.lower, gone.upper), self.config.ranges)


if __name__ == "__main__":
    unittest.main()
