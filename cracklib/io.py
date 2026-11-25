"""交换记录 ↔ 工程对象的装配。

读取 `data/record.json` 一类的合成/部署记录，构造 :class:`ProcessConfig`、
:class:`Window` 与训练 probe；不同配方只需换一份记录即可复用同一套算子。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import Cell, LayerSpec, ProcessConfig, Window


def load_json(path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def config_from_record(record: dict) -> ProcessConfig:
    shape = tuple(int(x) for x in record["shape"])
    layers = tuple(
        LayerSpec(
            threshold=float(item["threshold"]),
            footprint=tuple((int(dr), int(dc)) for dr, dc in item["footprint"]),
        )
        for item in record["layers"]
    )
    faces = {
        name: tuple(Cell(int(l), int(r), int(c)) for l, r, c in coords)
        for name, coords in record["faces"].items()
    }
    vertical = tuple((int(dr), int(dc)) for dr, dc in record["vertical_offsets"])
    scratch = record.get("scale", {}).get("scratch_bytes")
    return ProcessConfig(
        shape=shape,
        ranges=(float(record["ranges"][0]), float(record["ranges"][1])),
        outside=float(record["outside"]),
        layers=layers,
        vertical_offsets=vertical,
        faces=faces,
        scratch_bytes=int(scratch) if scratch is not None else None,
    )


def window_from_record(item: dict, provenance: str = "") -> Window:
    lower = np.asarray(item["lower"], dtype=np.float64)
    upper = np.asarray(item["upper"], dtype=np.float64)
    valid = None
    if "valid" in item:
        valid = np.asarray(item["valid"], dtype=bool)
    return Window(
        window_id=str(item["id"]),
        rev=int(item["rev"]),
        layer=int(item["layer"]),
        origin=(int(item["origin"][0]), int(item["origin"][1])),
        lower=lower,
        upper=upper,
        valid=valid,
        provenance=provenance or str(item.get("provenance", "")),
    )


def probe_from_record(record: dict) -> list:
    return [tuple(float(x) if i == 3 else int(x) for i, x in enumerate(row))
            for row in record["probe"]]


def build_store(config: ProcessConfig, windows) -> "tuple":
    """延迟导入避免循环依赖。返回 (store, 冲突格点列表)。"""
    from .store import ObservationStore

    store = ObservationStore(config)
    conflicts: list = []
    for item in windows:
        win = item if isinstance(item, Window) else window_from_record(item)
        conflicts.extend(store.upsert_window(win))
    return store, conflicts
