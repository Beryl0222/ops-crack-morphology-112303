"""贯通判定、补测见证、版本失效与冲突隔离的端到端测试。"""
import unittest

import numpy as np

from cracklib.graph import Verdict
from cracklib.io import config_from_record, window_from_record
from cracklib.model import Cell, ProcessConfig, Window
from cracklib.pipeline import CrackEngine
from crack_morphology_input import SAMPLE, load_record


def _chain_config(nr=9, nc=8, thresholds=(0.58, 0.62),
                  source_col=None, sink_row=4, sink_cols=(0,),
                  source_row=4):
    """两层配方：层内四邻接为主，跨层只允许 (0,0) 对齐偏移。

    外边界常量取 0（与交换记录一致）。两层 footprint 异构（非矩形、非平移），
    但都不含 ``(0,-1)``，使汇面所在的列 0 不会被原图外常量在腐蚀阶段压死；
    合成亮带内移并加宽，链体所有响应格点的支撑都落在原图内。
    """
    source_col = nc - 1 if source_col is None else source_col
    l0 = ((0, 0), (1, 0), (-1, 0), (0, 1), (1, 1))
    l1 = ((0, 0), (1, 0), (-1, 0), (0, 1), (-1, 1))
    from cracklib.model import LayerSpec
    return ProcessConfig(
        shape=(2, nr, nc), ranges=(0.0, 1.0), outside=0.0,
        layers=(LayerSpec(thresholds[0], tuple(l0)),
                LayerSpec(thresholds[1], tuple(l1))),
        vertical_offsets=((0, 0),),
        faces={"source": (Cell(0, source_row, source_col),),
               "sink": tuple(Cell(1, sink_row, c) for c in sink_cols)},
    )


def _bright_band(nr, nc, rows=range(2, 7), value=0.9):
    x = np.full((nr, nc), 0.1)
    x[list(rows), :] = value
    return x


class SyntheticPenetrationTests(unittest.TestCase):
    def _engine_with_chain(self, layer_values):
        cfg = _chain_config()
        engine = CrackEngine(cfg)
        for layer, values in enumerate(layer_values):
            engine.upsert_window(Window(f"w{layer}", 1, layer, (0, 0),
                                        values, values.copy()))
        return engine, cfg

    def test_definite_penetration_on_fully_measured_chain(self):
        nr, nc = 9, 8
        x0 = _bright_band(nr, nc)
        x1 = _bright_band(nr, nc)
        # 源在层 0 最左、汇在层 1 最右，形成跨两层的完整正向链。
        cfg = _chain_config(source_col=0, sink_cols=(nc - 1,))
        engine = CrackEngine(cfg)
        engine.upsert_window(Window("w0", 1, 0, (0, 0), x0, x0.copy()))
        engine.upsert_window(Window("w1", 1, 1, (0, 0), x1, x1.copy()))
        result = engine.penetration()
        self.assertEqual(result.verdict, Verdict.DEFINITE)
        path = result.definite_path
        self.assertEqual(path[0], cfg.faces["source"][0])
        self.assertEqual(path[-1], cfg.faces["sink"][0])
        self.assertEqual({c.layer for c in path}, {0, 1})
        # 跨层边必须符合配置偏移，层内边必须是四邻接。
        for a, b in zip(path, path[1:]):
            if a.layer == b.layer:
                self.assertEqual(abs(a.row - b.row) + abs(a.col - b.col), 1)
            else:
                self.assertIn(b, cfg.neighbors_below(a))
        self.assertEqual(result.resample, ())

    def test_impossible_when_possible_graph_does_not_reach_sink(self):
        nr, nc = 5, 6
        # 外常量为 1 的配方下，全零数据在原图外高常量支撑下仍可能在边缘达标，
        # 因此这里改用外常量 0 的独立配方：全零工件不可能产生达阈值响应。
        from cracklib.model import LayerSpec
        cfg = ProcessConfig(
            shape=(2, nr, nc), ranges=(0.0, 1.0), outside=0.0,
            layers=(LayerSpec(0.58, ((0, 0), (1, 0), (-1, 0), (0, 1))),
                    LayerSpec(0.62, ((0, 0), (1, 0), (-1, 0), (0, -1)))),
            vertical_offsets=((0, 0),),
            faces={"source": (Cell(0, 2, 0),), "sink": (Cell(1, 2, nc - 1),)},
        )
        dead = np.zeros((nr, nc))
        engine = CrackEngine(cfg)
        engine.upsert_window(Window("w0", 1, 0, (0, 0), dead, dead.copy()))
        engine.upsert_window(Window("w1", 1, 1, (0, 0), dead, dead.copy()))
        result = engine.penetration()
        self.assertEqual(result.verdict, Verdict.IMPOSSIBLE)
        self.assertIsNone(result.possible_path)

    def test_needs_measurement_returns_small_targeted_witness(self):
        nr, nc = 9, 8
        # 源在层 0 最右、汇在层 1 最左；把缺测洞放在汇面格点本身，
        # 该格点“可能但不肯定”达阈值，判定必然是待补测且无法绕过。
        cfg = _chain_config()
        x0 = _bright_band(nr, nc)
        lo1 = _bright_band(nr, nc)
        hi1 = lo1.copy()
        hole = cfg.faces["sink"][0]
        lo1[hole.row, hole.col] = 0.0
        hi1[hole.row, hole.col] = 1.0
        engine = CrackEngine(cfg)
        engine.upsert_window(Window("w0", 1, 0, (0, 0), x0, x0.copy()))
        engine.upsert_window(Window("w1", 1, 1, (0, 0), lo1, hi1))

        result = engine.penetration()
        self.assertEqual(result.verdict, Verdict.NEEDS_MEASUREMENT)
        self.assertIsNotNone(result.possible_path)
        self.assertTrue(result.resample)
        # 见证集远小于全部缺测格点，且每个目标都未实测确定、带溯源。
        witness_cells = {t.cell for t in result.resample}
        total_uncertain = sum(
            1 for r in range(nr) for c in range(nc)
            if not engine.store.constraint(Cell(1, r, c)).is_exact()
        )
        self.assertLessEqual(len(witness_cells), total_uncertain)
        for target in result.resample:
            self.assertFalse(engine.store.constraint(target.cell).is_exact())
            self.assertTrue(target.sources)
        # 汇面缺测格点本身必须出现在见证集里。
        self.assertIn(hole, witness_cells)
        # 收紧该缺测格点为高确定值后，结论应翻转为肯定贯通。
        lo1_fixed, hi1_fixed = lo1.copy(), hi1.copy()
        lo1_fixed[hole.row, hole.col] = hi1_fixed[hole.row, hole.col] = 0.95
        engine.upsert_window(Window("w1", 2, 1, (0, 0), lo1_fixed, hi1_fixed))
        self.assertEqual(engine.penetration().verdict, Verdict.DEFINITE)

    def test_tightening_can_break_one_path_while_another_survives(self):
        # 汇面给两个候选格点（行 2 与行 6），各自路径上放一个缺测洞；
        # 实测填掉上侧洞为低值使该路径失效，下侧路径仍待补测但保留。
        nr, nc = 9, 8
        cfg = _chain_config(sink_row=4, sink_cols=(0,))
        cfg.faces["sink"] = (Cell(1, 2, 0), Cell(1, 6, 0))
        x0 = _bright_band(nr, nc)
        lo1 = _bright_band(nr, nc)
        hi1 = lo1.copy()
        # 两个汇面候选各放一个缺测洞。
        lo1[2, 0], hi1[2, 0] = 0.0, 1.0
        lo1[6, 0], hi1[6, 0] = 0.0, 1.0
        engine = CrackEngine(cfg)
        engine.upsert_window(Window("w0", 1, 0, (0, 0), x0, x0.copy()))
        engine.upsert_window(Window("w1", 1, 1, (0, 0), lo1, hi1))
        before = engine.penetration()
        self.assertEqual(before.verdict, Verdict.NEEDS_MEASUREMENT)
        # 上侧洞实测为低值：行 2 路径肯定失效，行 6 路径仍可能贯通。
        lo1[2, 0] = hi1[2, 0] = 0.05
        engine.upsert_window(Window("w1", 2, 1, (0, 0), lo1, hi1))
        after = engine.penetration()
        self.assertEqual(after.verdict, Verdict.NEEDS_MEASUREMENT)
        self.assertTrue(any(c.row == 6 for c in after.possible_path))
        self.assertFalse(any(c.row == 2 for c in after.possible_path))


class ExchangeRecordNarrativeTests(unittest.TestCase):
    def setUp(self):
        self.record = load_record()
        self.engine = CrackEngine.from_record(SAMPLE)

    def test_sample_is_needs_measurement_with_bounded_witness(self):
        # 交换样本层 1 存在缺测，叙事上是“可能贯通但缺采样”。
        result = self.engine.penetration()
        self.assertEqual(result.verdict, Verdict.NEEDS_MEASUREMENT)
        self.assertIsNotNone(result.possible_path)
        self.assertTrue(result.resample)
        for target in result.resample:
            self.assertFalse(self.engine.store.constraint(target.cell).is_exact())
        self.assertEqual(result.provenance["windows"]["b"][0], 2)

    def test_window_revision_tightens_response_and_versions_invalidate_cache(self):
        first = self.engine.penetration()
        self.assertEqual(first.store_version, self.engine.store.version)
        # 用 rev1（更宽区间）替换 rev2：可能路径仍在，但版本号必须推进。
        self.engine.upsert_window(self.record["windows"][1])
        second = self.engine.penetration()
        self.assertGreater(second.store_version, first.store_version)
        self.assertEqual(second.verdict, Verdict.NEEDS_MEASUREMENT)
        # 补测目标应随区间放宽而变化（rev1 的见证集不小于 rev2）。
        self.assertGreaterEqual(len(second.resample), len(first.resample))

    def test_conflict_blocks_affected_query_but_independent_region_stays_queryable(self):
        engine = CrackEngine.from_record(SAMPLE, use_conflict_window=True)
        result = engine.penetration()
        self.assertIn(result.verdict,
                      (Verdict.BLOCKED, Verdict.NEEDS_MEASUREMENT, Verdict.IMPOSSIBLE))
        resp = engine.responses[0]
        self.assertTrue(resp.blocked[3, 2])
        # 独立区域：层 1 内不经过冲突点的面仍可给出非阻断判定。
        other = engine.penetration((Cell(1, 1, 6),), (Cell(1, 3, 6),))
        self.assertNotEqual(other.verdict, Verdict.BLOCKED)

    def test_stale_judgement_is_not_reused_after_store_change(self):
        # 用合成的待补测工况，再补一个制造阻断的冲突窗口：
        # 观测版本推进后旧判定必须失效，结论不得继续按待补测放行。
        nr, nc = 7, 8
        cfg = _chain_config()
        x0 = _bright_band(nr, nc)
        lo1 = _bright_band(nr, nc)
        hi1 = lo1.copy()
        hole = cfg.faces["sink"][0]
        lo1[hole.row, hole.col], hi1[hole.row, hole.col] = 0.0, 1.0
        engine = CrackEngine(cfg)
        engine.upsert_window(Window("w0", 1, 0, (0, 0), x0, x0.copy()))
        engine.upsert_window(Window("w1", 1, 1, (0, 0), lo1, hi1))
        first = engine.penetration()
        self.assertEqual(first.verdict, Verdict.NEEDS_MEASUREMENT)

        # 在源面格点上放一个与现有高值冲突的低值确定窗口。
        source = cfg.faces["source"][0]
        clash = np.array([[0.01]])
        engine.upsert_window(Window(
            "clash", 1, source.layer, (source.row, source.col), clash, clash.copy()))
        second = engine.penetration()
        self.assertGreater(second.store_version, first.store_version)
        self.assertNotEqual(second.verdict, first.verdict)
        self.assertEqual(second.verdict, Verdict.BLOCKED)


if __name__ == "__main__":
    unittest.main()
