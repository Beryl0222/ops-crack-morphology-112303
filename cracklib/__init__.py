"""缺测像素上的跨层区间形态学与缺陷连通判定工程包。"""
from .graph import ConnectivityGraph, PenetrationEngine, PenetrationResult, Verdict
from .io import build_store, config_from_record, probe_from_record, window_from_record
from .model import Cell, Constraint, LayerSpec, ProcessConfig, ResampleTarget, Window
from .morphology import (
    AllocationTracker,
    ChunkedClosingScheduler,
    ClosingResult,
    interval_close_fused,
    interval_dilate,
    interval_erode,
)
from .pipeline import CrackEngine, EngineSnapshot
from .store import ObservationStore
from .training import MorphologyTrainer, NotTrainableError, TrainingPlan

__all__ = [
    "Cell", "Constraint", "LayerSpec", "ProcessConfig", "ResampleTarget", "Window",
    "ObservationStore",
    "interval_erode", "interval_dilate", "interval_close_fused",
    "ChunkedClosingScheduler", "ClosingResult", "AllocationTracker",
    "ConnectivityGraph", "PenetrationEngine", "PenetrationResult", "Verdict",
    "MorphologyTrainer", "TrainingPlan", "NotTrainableError",
    "CrackEngine", "EngineSnapshot",
    "config_from_record", "window_from_record", "probe_from_record", "build_store",
]
