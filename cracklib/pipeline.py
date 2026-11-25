"""工程门面：把观测存储、分块闭运算、贯通查询与训练装配成一条可复用流水线。

同一配方下：
* 窗口替换会推进观测版本，旧的响应/贯通缓存自动失效并按需重算；
* 冲突只阻断受影响区域，独立区域的贯通判定继续可查；
* 贯通判定、补测坐标、训练梯度都带窗口级溯源（id + rev + 局部坐标）。
"""
from __future__ import annotations

from dataclasses import dataclass

from .graph import PenetrationEngine, PenetrationResult
from .io import config_from_record, load_json, probe_from_record
from .model import Cell, ProcessConfig, Window
from .morphology import AllocationTracker, ChunkedClosingScheduler, ClosingResult
from .store import ObservationStore
from .training import MorphologyTrainer, TrainingPlan


@dataclass
class EngineSnapshot:
    store_version: int
    responses: dict
    conflicts: tuple
    allocator_peak: int


class CrackEngine:
    def __init__(self, config: ProcessConfig, scratch_bytes: int | None = None):
        self.config = config
        self.store = ObservationStore(config)
        self.scratch_bytes = scratch_bytes
        self.allocator = AllocationTracker()
        self._responses: dict[int, ClosingResult] = {}
        self._response_version: int | None = None
        self._conflicts: tuple[Cell, ...] = ()

    # ------------------------------------------------------------ 装载与替换
    @classmethod
    def from_record(cls, path, use_conflict_window: bool = False,
                    scratch_bytes: int | None = None) -> "CrackEngine":
        record = load_json(path)
        config = config_from_record(record)
        engine = cls(config, scratch_bytes=scratch_bytes)
        engine.load_windows(record["windows"])
        if use_conflict_window:
            engine.upsert_window(record["alternate_conflict"])
        engine._probe = probe_from_record(record)
        return engine

    def load_windows(self, windows) -> tuple[Cell, ...]:
        fresh: list[Cell] = []
        for item in windows:
            fresh.extend(self.upsert_window(item))
        return tuple(fresh)

    def upsert_window(self, item) -> tuple[Cell, ...]:
        win = item if isinstance(item, Window) else None
        if win is None:
            from .io import window_from_record

            win = window_from_record(item)
        fresh = self.store.upsert_window(win)
        self._conflicts = self.store.conflict_cells()
        self._response_version = None  # 响应失效，强制重算
        return fresh

    # ------------------------------------------------------------ 响应计算
    def recompute(self) -> EngineSnapshot:
        scheduler = ChunkedClosingScheduler(
            self.config, self.store, scratch_bytes=self.scratch_bytes,
            allocator=self.allocator,
        )
        self._responses = scheduler.run_all()
        self._response_version = self.store.version
        return EngineSnapshot(
            self.store.version, self._responses, self._conflicts, self.allocator.peak
        )

    @property
    def responses(self) -> dict[int, ClosingResult]:
        if self._response_version != self.store.version:
            self.recompute()
        return self._responses

    # ------------------------------------------------------------ 贯通查询
    def penetration(self, source_face=None, sink_face=None) -> PenetrationResult:
        engine = PenetrationEngine(self.config, self.store, self.responses)
        return engine.query(source_face, sink_face)

    # ------------------------------------------------------------ 训练
    def training_plan(self, probe=None) -> TrainingPlan:
        probe = probe or getattr(self, "_probe", None)
        if probe is None:
            raise ValueError("缺少 probe：请传入 (layer, row, col, upstream_grad) 列表")
        trainer = MorphologyTrainer(self.config, self.store, self.responses)
        return trainer.gradients(probe)
