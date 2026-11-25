"""固定工况下确定响应区域的上游梯度（可微形态学）。

设计要点：
* 只对所有参与判定的实际输入支撑均已实测确定（区间宽度为 0、无冲突）的
  响应格点开放训练；只要原始依赖里还存在区间不确定性，就返回不能训练的原因；
* 边界常量没有输入身份，因而没有输入梯度；
* 窗口里同一物理格点的贡献只交给唯一输入身份——梯度按全局物理格点累积，
  不会因为多窗口重叠而重复计数；
* 非唯一极值（完全相同的实测强度）按全局格点 C 序裁决选择，保证前向选择与
  反向路由严格一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Cell, ProcessConfig
from .morphology import ClosingResult
from .store import ObservationStore


class NotTrainableError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    probe_cells: tuple[Cell, ...]
    trainable: bool
    reasons: dict
    gradients: dict[Cell, float]
    provenance: dict

    @property
    def blocked_cells(self) -> tuple[Cell, ...]:
        return tuple(self.reasons.get("blocked", ()))

    def to_dict(self) -> dict:
        return {
            "trainable": self.trainable,
            "reasons": {k: [c.as_tuple() for c in v] if isinstance(v, (list, tuple)) else v
                        for k, v in self.reasons.items()},
            "gradients": {",".join(map(str, c.as_tuple())): g for c, g in self.gradients.items()},
            "provenance": self.provenance,
        }


class MorphologyTrainer:
    def __init__(self, config: ProcessConfig, store: ObservationStore,
                 responses: dict[int, ClosingResult]):
        self.config = config
        self.store = store
        self.responses = responses
        self._value_cache: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------- 可训练性检查
    def check_cell(self, cell: Cell) -> dict:
        """返回一个响应格点的可训练性诊断。"""
        _, nr, nc = self.config.shape
        layer = cell.layer
        resp = self.responses[layer]
        spec = self.config.layer(layer)
        reasons: dict[str, list] = {"blocked": [], "uncertain": [], "outside_support": []}
        if resp.blocked[cell.row, cell.col]:
            reasons["blocked"].append(cell)
            return {"trainable": False, "reasons": reasons}
        support = {(dr1 + dr2, dc1 + dc2)
                   for dr1, dc1 in spec.reflected_offsets
                   for dr2, dc2 in spec.footprint}
        for dr, dc in sorted(support):
            r, c = cell.row + dr, cell.col + dc
            if not (0 <= r < nr and 0 <= c < nc):
                reasons["outside_support"].append((dr, dc))
                continue
            inp = Cell(layer, r, c)
            cons = self.store.constraint(inp)
            if cons.conflict:
                reasons["blocked"].append(inp)
            elif not cons.is_exact():
                reasons["uncertain"].append(inp)
        trainable = not reasons["blocked"] and not reasons["uncertain"]
        return {"trainable": trainable, "reasons": reasons}

    # --------------------------------------------------------------- 前向路由
    def _deterministic_values(self, layer: int) -> np.ndarray:
        """该层可训练前向使用的确定性输入；未确定格点给 NaN（不应被选中）。"""
        if layer in self._value_cache:
            return self._value_cache[layer]
        region = self.store.materialize_layer(layer)
        vals = np.where(region["exact"], region["lower"], np.nan)
        vals[region["blocked"]] = np.nan
        self._value_cache[layer] = vals
        return vals

    def _pick(self, values: np.ndarray, candidates, mode: str):
        """按值选极值，平局按全局格点 C 序裁决。"""
        best, best_cell = None, None
        for cell, val in candidates:
            if np.isnan(val):
                continue
            if best is None:
                best, best_cell = val, cell
            elif mode == "min" and (val < best or (val == best and cell < best_cell)):
                best, best_cell = val, cell
            elif mode == "max" and (val > best or (val == best and cell < best_cell)):
                best, best_cell = val, cell
        return best_cell

    def _erode_route(self, layer: int, cell: Cell):
        spec = self.config.layer(layer)
        _, nr, nc = self.config.shape
        vals = self._deterministic_values(layer)
        candidates = []
        for dr, dc in spec.footprint:
            r, c = cell.row + dr, cell.col + dc
            if 0 <= r < nr and 0 <= c < nc:
                candidates.append((Cell(layer, r, c), vals[r, c]))
            else:
                candidates.append((None, self.config.outside))
        return self._pick(np.asarray(vals), candidates, "min")

    def _dilate_route(self, layer: int, cell: Cell, erode_values: dict):
        spec = self.config.layer(layer)
        _, nr, nc = self.config.shape
        candidates = []
        for dr, dc in spec.reflected_offsets:
            r, c = cell.row + dr, cell.col + dc
            if 0 <= r < nr and 0 <= c < nc:
                mid = Cell(layer, r, c)
                candidates.append((mid, erode_values[mid]))
            else:
                candidates.append((None, self.config.outside))
        return self._pick(np.asarray(erode_values), candidates, "max")

    def route_cell(self, cell: Cell) -> tuple[Cell, ...]:
        """返回一个响应格点的完整极值选择链（去重后的物理输入格点）。"""
        layer = cell.layer
        spec = self.config.layer(layer)
        _, nr, nc = self.config.shape
        vals = self._deterministic_values(layer)
        erode_values: dict[Cell, float] = {}
        erode_choice: dict[Cell, Cell | None] = {}
        erode_cells = set()
        # 闭运算在 cell 处的支撑内，所有可能成为膨胀候选的腐蚀中间格点。
        for dr, dc in spec.reflected_offsets:
            r, c = cell.row + dr, cell.col + dc
            if 0 <= r < nr and 0 <= c < nc:
                erode_cells.add(Cell(layer, r, c))
        for mid in erode_cells:
            choice = self._erode_route(layer, mid)
            erode_choice[mid] = choice
            erode_values[mid] = (self.config.outside if choice is None
                                 else vals[choice.row, choice.col])
        final_mid = self._dilate_route(layer, cell, erode_values)
        chain: list[Cell] = []
        if final_mid is not None:
            src = erode_choice[final_mid]
            if src is not None:
                chain.append(src)
        return tuple(dict.fromkeys(chain))

    # --------------------------------------------------------------- 反向梯度
    def gradients(self, probe: list) -> TrainingPlan:
        """``probe`` 为 ``[(cell, upstream_grad), ...]``（与交换记录 probe 对应）。"""
        probes: list[tuple[Cell, float]] = []
        for item in probe:
            layer, r, c, g = item
            probes.append((Cell(int(layer), int(r), int(c)), float(g)))

        all_reasons = {"blocked": [], "uncertain": [], "outside_support": []}
        trainable = True
        for cell, _ in probes:
            diag = self.check_cell(cell)
            if not diag["trainable"]:
                trainable = False
                for key, vals in diag["reasons"].items():
                    all_reasons[key].extend(vals)
        for key in all_reasons:
            all_reasons[key] = tuple(dict.fromkeys(all_reasons[key]))

        gradients: dict[Cell, float] = {}
        if trainable:
            for cell, upstream in probes:
                for src in self.route_cell(cell):
                    gradients[src] = gradients.get(src, 0.0) + upstream
        provenance = {
            "windows": {wid: (win.rev, win.provenance) for wid, win in self.store.windows().items()},
            "tie_break": "global_cell_c_order",
            "boundary_constant": self.config.outside,
        }
        return TrainingPlan(tuple(c for c, _ in probes), trainable, all_reasons,
                            gradients, provenance)
