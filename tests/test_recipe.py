"""跨配方复用与溯源可追溯性。"""
import json
import unittest

from cracklib.io import config_from_record, probe_from_record, window_from_record
from cracklib.pipeline import CrackEngine
from crack_morphology_input import load_record, SAMPLE


class RecipeReuseTests(unittest.TestCase):
    def test_same_pipeline_serves_two_recipes(self):
        record = load_record()
        # 派生一份新配方：层阈值与跨层偏移都不同。
        alt = json.loads(json.dumps(record))
        alt["layers"][0]["threshold"] = 0.95
        alt["layers"][1]["threshold"] = 0.95
        alt["vertical_offsets"] = [[1, 0]]
        for window in alt["windows"]:
            window["id"] = window["id"] + "'"

        engine_a = CrackEngine.from_record(SAMPLE)
        config_b = config_from_record(alt)
        engine_b = CrackEngine(config_b)
        engine_b.load_windows(alt["windows"])

        self.assertNotEqual(engine_a.config.layer(0).threshold,
                            engine_b.config.layer(0).threshold)
        self.assertNotEqual(engine_a.config.vertical_offsets,
                            engine_b.config.vertical_offsets)
        # 同一套算子类与查询入口在两份配方上都能给出合法结论。
        for engine in (engine_a, engine_b):
            verdict = engine.penetration().verdict
            self.assertIn(verdict.value,
                          ("definite", "impossible", "needs_measurement", "blocked"))

    def test_provenance_links_results_to_observations(self):
        engine = CrackEngine.from_record(SAMPLE)
        result = engine.penetration()
        self.assertEqual(result.provenance["store_version"], engine.store.version)
        self.assertEqual(result.provenance["windows"]["c"][0], 1)
        plan = engine.training_plan(probe_from_record(load_record()))
        # 即便不可训练，溯源仍说明结论来自哪份观测、平局规则与边界常量。
        self.assertEqual(plan.provenance["tie_break"], "global_cell_c_order")
        self.assertTrue(plan.provenance["windows"])

        # 单条约束溯源可回到窗口 id、修订号与窗口内局部坐标。
        from cracklib.model import Cell
        sources = engine.store.constraint(Cell(0, 3, 3)).sources
        self.assertTrue(any(src[0] == "a" and src[1] == 1 for src in sources))


if __name__ == "__main__":
    unittest.main()
