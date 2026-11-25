"""区间观测存储。

职责：
* 接收带重叠的窗口，同一物理格点的多次观测取约束交集；
* 交集为空时把该格点标记为冲突阻断，而不是让整个工件报废；
* 支持同 id 窗口的整窗替换（替换后由上层重新判定响应与连通）；
* 按层/区域物化区间数组，未被任何窗口覆盖的格点退化为物理允许区间；
* 每条结果都可回溯到具体的窗口 id、修订号与窗口内局部坐标。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Cell, Constraint, ProcessConfig, Window


@dataclass(frozen=True, slots=True)
class _Contribution:
    window_id: str
    rev: int
    local_rc: tuple[int, int]
    lower: float
    upper: float


class ObservationStore:
    def __init__(self, config: ProcessConfig):
        self.config = config
        self._contrib: dict[Cell, list[_Contribution]] = {}
        self._constraint: dict[Cell, Constraint] = {}
        self._windows: dict[str, Window] = {}
        # 任何写操作都推进版本号，缓存（响应、连通判定）据此失效。
        self._version = 0

    # ------------------------------------------------------------------ 写入
    @property
    def version(self) -> int:
        return self._version

    def windows(self) -> dict[str, Window]:
        return dict(self._windows)

    def upsert_window(self, window: Window) -> tuple[Cell, ...]:
        """加入或整窗替换一份观测，返回本次变为冲突状态的格点。"""
        cfg = self.config
        nr, nc = window.shape
        if not (0 <= window.origin[0] and window.origin[0] + nr <= cfg.shape[1]
                and 0 <= window.origin[1] and window.origin[1] + nc <= cfg.shape[2]):
            raise ValueError(f"窗口 {window.window_id} 越出工件域")
        if np.any(np.asarray(window.lower) > np.asarray(window.upper)):
            raise ValueError(f"窗口 {window.window_id} 存在 lower>upper 的非法区间")

        affected: set[Cell] = set()
        old = self._windows.get(window.window_id)
        if old is not None:
            for r in range(old.shape[0]):
                for c in range(old.shape[1]):
                    affected.add(old.cell_at(r, c))
            for cell in affected:
                self._contrib[cell] = [
                    item for item in self._contrib.get(cell, ()) if item.window_id != window.window_id
                ]

        new_cells: set[Cell] = set()
        lower = np.asarray(window.lower, dtype=np.float64)
        upper = np.asarray(window.upper, dtype=np.float64)
        valid = np.ones_like(lower, dtype=bool) if window.valid is None else np.asarray(window.valid, dtype=bool)
        for r in range(nr):
            for c in range(nc):
                if not valid[r, c]:
                    continue
                cell = window.cell_at(r, c)
                new_cells.add(cell)
                self._contrib.setdefault(cell, []).append(
                    _Contribution(window.window_id, window.rev, (r, c), float(lower[r, c]), float(upper[r, c]))
                )
        affected |= new_cells

        fresh_conflicts: list[Cell] = []
        for cell in sorted(affected, key=lambda x: x.linear_index(cfg.shape)):
            items = self._contrib.get(cell, ())
            if not items:
                self._constraint.pop(cell, None)
                continue
            lo = max(it.lower for it in items)
            hi = min(it.upper for it in items)
            sources = tuple((it.window_id, it.rev, it.local_rc) for it in items)
            conflict = lo > hi
            prior = self._constraint.get(cell)
            if conflict and (prior is None or not prior.conflict):
                fresh_conflicts.append(cell)
            self._constraint[cell] = Constraint(cell, lo, hi, sources, conflict)

        self._windows[window.window_id] = window
        self._version += 1
        return tuple(fresh_conflicts)

    def remove_window(self, window_id: str) -> None:
        old = self._windows.pop(window_id, None)
        if old is None:
            return
        affected: set[Cell] = set()
        for r in range(old.shape[0]):
            for c in range(old.shape[1]):
                affected.add(old.cell_at(r, c))
        cfg = self.config
        lo_default, hi_default = cfg.ranges
        for cell in sorted(affected, key=lambda x: x.linear_index(cfg.shape)):
            items = [it for it in self._contrib.get(cell, ()) if it.window_id != window_id]
            if items:
                self._contrib[cell] = items
                lo = max(it.lower for it in items)
                hi = min(it.upper for it in items)
                sources = tuple((it.window_id, it.rev, it.local_rc) for it in items)
                self._constraint[cell] = Constraint(cell, lo, hi, sources, lo > hi)
            else:
                self._contrib.pop(cell, None)
                self._constraint.pop(cell, None)
        self._version += 1

    # ------------------------------------------------------------------ 查询
    def constraint(self, cell: Cell) -> Constraint:
        found = self._constraint.get(cell)
        if found is not None:
            return found
        lo, hi = self.config.ranges
        return Constraint(cell, lo, hi, (), False)

    def conflict_cells(self, layer: int | None = None) -> tuple[Cell, ...]:
        out = [cons.cell for cons in self._constraint.values()
               if cons.conflict and (layer is None or cons.cell.layer == layer)]
        return tuple(sorted(out, key=lambda x: x.linear_index(self.config.shape)))

    def materialize_layer(
        self, layer: int, rows: slice | None = None, cols: slice | None = None
    ) -> dict:
        """物化一层（或其矩形区域）的区间与阻断掩码。

        未观测格点填物理允许区间，冲突格点 ``blocked=True``。
        """
        _, nr_all, nc_all = self.config.shape
        rows = rows or slice(0, nr_all)
        cols = cols or slice(0, nc_all)
        r0, r1, _ = slice(rows.start or 0, rows.stop, rows.step).indices(nr_all)
        c0, c1, _ = slice(cols.start or 0, cols.stop, cols.step).indices(nc_all)
        h, w = r1 - r0, c1 - c0
        lo_default, hi_default = self.config.ranges
        lower = np.full((h, w), lo_default, dtype=np.float64)
        upper = np.full((h, w), hi_default, dtype=np.float64)
        blocked = np.zeros((h, w), dtype=bool)
        observed = np.zeros((h, w), dtype=bool)
        exact = np.zeros((h, w), dtype=bool)
        sources: dict[tuple[int, int], tuple] = {}
        for r in range(r0, r1):
            for c in range(c0, c1):
                cons = self._constraint.get(Cell(layer, r, c))
                if cons is None:
                    continue
                lr, lc = r - r0, c - c0
                observed[lr, lc] = True
                sources[(lr, lc)] = cons.sources
                if cons.conflict:
                    blocked[lr, lc] = True
                    continue
                lower[lr, lc] = cons.lower
                upper[lr, lc] = cons.upper
                exact[lr, lc] = cons.is_exact()
        return {
            "layer": layer,
            "origin": (r0, c0),
            "lower": lower,
            "upper": upper,
            "blocked": blocked,
            "observed": observed,
            "exact": exact,
            "sources": sources,
        }
