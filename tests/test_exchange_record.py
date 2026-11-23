"""检查交换数据中用于回查的事实。"""
import unittest

import numpy as np

from crack_morphology_input import load_record


class ExchangeRecordTests(unittest.TestCase):
    def test_exchange_contract(self):
        record = load_record()
        self.assertTrue(all(np.all(np.asarray(w["lower"]) <= w["upper"]) for w in record["windows"]))
        self.assertTrue(record["windows"][1]["id"] == record["windows"][2]["id"] and record["windows"][1]["upper"][0][0] < record["windows"][2]["lower"][0][0])
        self.assertTrue(record["layers"][0]["footprint"] != record["layers"][1]["footprint"] and len(record["vertical_offsets"]) > 1)
        self.assertTrue(record["windows"][0]["upper"][1][2] > record["windows"][0]["lower"][1][2])
        self.assertTrue(record["alternate_conflict"]["upper"][0][0] < record["windows"][0]["lower"][2][2])
