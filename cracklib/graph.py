"""肯定/可能双连通图与跨工艺面贯通判定。

* 肯定图：闭运算响应下界已达层阈值的格点；
* 可能图：上界仍达阈值的格点（包含肯定图）；
* 单层边取四邻接，跨层边取配方 ``vertical_offsets``，双向遍历；
* 查询结论为“肯定贯通 / 肯定不贯通 / 待补测”，待补测时只返回
  一条当前可能路径上、真正决定该路径能否成立的少数原始格点作为补采目标。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .model import Cell, ProcessConfig, ResampleTarget
from .morphology import ClosingResult
from .store import ObservationStore


class Verdict(str, Enum):
    DEFINITE = "definite"            # 肯定贯通，可放行
    IMPOSSIBLE = "impossible"        # 肯定不贯通
    NEEDS_MEASUREMENT = "needs_measurement"  # 可能贯通但缺采样
    BLOCKED = "blocked"              # 查询面本身受冲突阻断


@dataclass(frozen=True, slots=True)
class PenetrationResult:
    verdict: Verdict
    source_face: tuple[Cell, ...]
    sink_face: tuple[Cell, ...]
    definite_path: tuple[Cell, ...] | None
    possible_path: tuple[Cell, ...] | None
    resample: tuple[ResampleTarget, ...]
    store_version: int
    provenance: dict

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "definite_path": None if self.definite_path is None
            else [c.as_tuple() for c in self.definite_path],
            "possible_path": None if self.possible_path is None
            else [c.as_tuple() for c in self.possible_path],
            "resample": [
                {
                    "cell": t.cell.as_tuple(),
                    "interval": list(t.current_interval),
                    "role": t.role,
                    "sources": [list(s) for s in t.sources],
                }
                for t in self.resample
            ],
            "store_version": self.store_version,
            "provenance": self.provenance,
        }


class ConnectivityGraph:
    """单层四邻接 + 配置跨层偏移的双判定图。"""

    def __init__(self, config: ProcessConfig, responses: dict[int, ClosingResult]):
        self.config = config
        self.responses = responses
        self._definite = {layer: resp.definite_mask() for layer, resp in responses.items()}
        self._possible = {layer: resp.possible_mask() for layer, resp in responses.items()}

    @property
    def store_version(self) -> int:
        return next(iter(self.responses.values())).store_version

    def is_definite(self, cell: Cell) -> bool:
        m = self._definite.get(cell.layer)
        return m is not None and bool(m[cell.row, cell.col])

    def is_possible(self, cell: Cell) -> bool:
        m = self._possible.get(cell.layer)
        return m is not None and bool(m[cell.row, cell.col])

    def neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        _, nr, nc = self.config.shape
        out: list[Cell] = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r2, c2 = cell.row + dr, cell.col + dc
            if 0 <= r2 < nr and 0 <= c2 < nc:
                out.append(Cell(cell.layer, r2, c2))
        out.extend(self.config.neighbors_below(cell))
        out.extend(self.config.neighbors_above(cell))
        return tuple(out)

    def _bfs(self, start: Cell, predicate) -> tuple[set[Cell], dict[Cell, Cell]]:
        seen = {start}
        parent: dict[Cell, Cell] = {}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in self.neighbors(cur):
                if nxt in seen or not predicate(nxt):
                    continue
                seen.add(nxt)
                parent[nxt] = cur
                queue.append(nxt)
        return seen, parent

    def _face_bfs(self, face, predicate):
        seen: set[Cell] = set()
        parent: dict[Cell, Cell] = {}
        for src in face:
            resp = self.responses.get(src.layer)
            if resp is None or resp.blocked[src.row, src.col]:
                continue
            if not predicate(src) or src in seen:
                continue
            seen.add(src)
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nxt in self.neighbors(cur):
                    resp = self.responses.get(nxt.layer)
                    if resp is None or resp.blocked[nxt.row, nxt.col]:
                        continue
                    if nxt in seen or not predicate(nxt):
                        continue
                    seen.add(nxt)
                    parent[nxt] = cur
                    queue.append(nxt)
        return seen, parent

    @staticmethod
    def _reconstruct(parent: dict[Cell, Cell], target: Cell, face) -> tuple[Cell, ...]:
        path = [target]
        while path[-1] not in face:
            path.append(parent[path[-1]])
        return tuple(reversed(path))


class PenetrationEngine:
    def __init__(self, config: ProcessConfig, store: ObservationStore,
                 responses: dict[int, ClosingResult]):
        self.config = config
        self.store = store
        self.graph = ConnectivityGraph(config, responses)
        # 贯通判定按观测版本缓存；store 写入后版本号变化即失效，
        # 不会拿旧判定继续放行。
        self._cache: dict[tuple, PenetrationResult] = {}

    def query(self, source_face=None, sink_face=None) -> PenetrationResult:
        source_face = tuple(source_face or self.config.faces["source"])
        sink_face = tuple(sink_face or self.config.faces["sink"])
        key = (self.store.version, tuple(source_face), tuple(sink_face))
        if key in self._cache:
            return self._cache[key]
        result = self._query(source_face, sink_face)
        self._cache[key] = result
        return result

    def _query(self, source_face, sink_face) -> PenetrationResult:
        g = self.graph
        blocked_face = tuple(c for c in (*source_face, *sink_face)
                             if (resp := g.responses.get(c.layer)) is not None
                             and resp.blocked[c.row, c.col])
        provenance = {
            "windows": {wid: (win.rev, win.provenance) for wid, win in self.store.windows().items()},
            "store_version": self.store.version,
        }
        if blocked_face:
            result = PenetrationResult(
                Verdict.BLOCKED, source_face, sink_face, None, None, (),
                self.store.version, {"blocked_face": [c.as_tuple() for c in blocked_face],
                                     **provenance},
            )
            return result

        seen_d, parent_d = g._face_bfs(source_face, g.is_definite)
        hit_d = next((c for c in sink_face if c in seen_d), None)
        if hit_d is not None:
            path = g._reconstruct(parent_d, hit_d, set(source_face))
            return PenetrationResult(
                Verdict.DEFINITE, source_face, sink_face, path, path, (),
                self.store.version, provenance,
            )

        seen_p, parent_p = g._face_bfs(source_face, g.is_possible)
        hit_p = next((c for c in sink_face if c in seen_p), None)
        if hit_p is None:
            return PenetrationResult(
                Verdict.IMPOSSIBLE, source_face, sink_face, None, None, (),
                self.store.version, provenance,
            )

        possible_path = g._reconstruct(parent_p, hit_p, set(source_face))
        resample = self._witness(possible_path)
        return PenetrationResult(
            Verdict.NEEDS_MEASUREMENT, source_face, sink_face, None, possible_path,
            tuple(resample), self.store.version, provenance,
        )

    # ----------------------------------------------------------- 补测见证集
    def _witness(self, path: tuple[Cell, ...]) -> list[ResampleTarget]:
        """只从一条可能路径上提取真正决定其成立与否的原始格点。

        路径上的肯定格点无需补采；对每个不确定响应格点，取闭运算完整支撑内
        仍未实测确定（区间宽度>0）的原始输入格点：
        * 区间下界的实测值若抬高，可让响应转为“肯定达阈值”；
        * 区间上界的实测值若压低，可让响应转为“肯定不达标”。
        已实测确定的格点与冲突阻断格点不会出现在建议里，
        原图外常量既不可补采也没有输入梯度。
        """
        _, nr, nc = self.config.shape
        targets: dict[Cell, ResampleTarget] = {}
        for cell in path:
            if self.graph.is_definite(cell):
                continue
            spec = self.config.layer(cell.layer)
            support = {(dr1 + dr2, dc1 + dc2)
                       for dr1, dc1 in spec.reflected_offsets
                       for dr2, dc2 in spec.footprint}
            for dr, dc in sorted(support):
                r, c = cell.row + dr, cell.col + dc
                if not (0 <= r < nr and 0 <= c < nc):
                    continue
                inp = Cell(cell.layer, r, c)
                if inp in targets:
                    continue
                cons = self.store.constraint(inp)
                if cons.conflict or cons.is_exact():
                    continue
                targets[inp] = ResampleTarget(
                    inp, (cons.lower, cons.upper),
                    "close_support:raise_lower_or_pull_upper", cons.sources,
                )
        return list(targets.values())
