"""核心数据模型：物理格点、窗口观测、工艺配方。

坐标约定与交换记录一致：格点为 ``(layer, row, col)``，窗口只覆盖单层，
``origin`` 为窗口左上角在该层的 ``(row, col)``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Cell:
    """全局物理格点。"""

    layer: int
    row: int
    col: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.layer, self.row, self.col)

    def linear_index(self, shape: Sequence[int]) -> int:
        """全局 C 序下标；非唯一极值的平局按该顺序裁决。"""
        _, nr, nc = shape
        return (self.layer * nr + self.row) * nc + self.col

    def _key(self) -> tuple[int, int, int]:
        return self.as_tuple()

    def __lt__(self, other: "Cell") -> bool:
        if not isinstance(other, Cell):
            return NotImplemented
        return self._key() < other._key()

    def __le__(self, other: "Cell") -> bool:
        if not isinstance(other, Cell):
            return NotImplemented
        return self._key() <= other._key()


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """单层工艺参数：阈值与非矩形结构元素（腐蚀相对偏移）。"""

    threshold: float
    footprint: tuple[tuple[int, int], ...]

    @property
    def reflected_offsets(self) -> tuple[tuple[int, int], ...]:
        """膨胀使用结构元素的反射集合。"""
        return tuple((-dr, -dc) for dr, dc in self.footprint)

    @property
    def support_offsets(self) -> tuple[tuple[int, int], ...]:
        """闭运算（先腐蚀后膨胀）的完整相对支撑。"""
        out: set[tuple[int, int]] = set()
        for dr1, dc1 in self.reflected_offsets:
            for dr2, dc2 in self.footprint:
                out.add((dr1 + dr2, dc1 + dc2))
        return tuple(sorted(out))

    def support_radius(self) -> tuple[int, int]:
        offsets = self.support_offsets
        return (
            max(abs(dr) for dr, _ in offsets),
            max(abs(dc) for _, dc in offsets),
        )


@dataclass(frozen=True, slots=True)
class ProcessConfig:
    """一份完整工艺配方。"""

    shape: tuple[int, int, int]
    ranges: tuple[float, float]
    outside: float
    layers: tuple[LayerSpec, ...]
    vertical_offsets: tuple[tuple[int, int], ...]
    faces: dict[str, tuple[Cell, ...]]
    scratch_bytes: int | None = None

    @property
    def num_layers(self) -> int:
        return self.shape[0]

    def layer(self, layer: int) -> LayerSpec:
        return self.layers[layer]

    def neighbors_below(self, cell: Cell) -> tuple[Cell, ...]:
        """按配置偏移给出本格点到下一层的双向跨层邻居（落在域外的丢弃）。"""
        if cell.layer >= self.num_layers - 1:
            return ()
        _, nr, nc = self.shape
        out: list[Cell] = []
        for dr, dc in self.vertical_offsets:
            nrw, ncc = cell.row + dr, cell.col + dc
            if 0 <= nrw < nr and 0 <= ncc < nc:
                out.append(Cell(cell.layer + 1, nrw, ncc))
        return tuple(out)

    def neighbors_above(self, cell: Cell) -> tuple[Cell, ...]:
        if cell.layer <= 0:
            return ()
        _, nr, nc = self.shape
        out: list[Cell] = []
        for dr, dc in self.vertical_offsets:
            nrw, ncc = cell.row - dr, cell.col - dc
            if 0 <= nrw < nr and 0 <= ncc < nc:
                out.append(Cell(cell.layer - 1, nrw, ncc))
        return tuple(out)


@dataclass(frozen=True, slots=True)
class Window:
    """一个带重叠的窗口观测。

    ``lower``/``upper`` 同形且含端点；完全测到的格点上下界相同，
    缺测格点给物理允许区间。``valid`` 为有效掩码，掩码外的条目不构成约束。
    """

    window_id: str
    rev: int
    layer: int
    origin: tuple[int, int]
    lower: "object"
    upper: "object"
    valid: "object | None" = None
    provenance: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.lower.shape

    def cell_at(self, r: int, c: int) -> Cell:
        return Cell(self.layer, self.origin[0] + r, self.origin[1] + c)

    def cells(self) -> Iterable[Cell]:
        nr, nc = self.shape
        for r in range(nr):
            for c in range(nc):
                if self.valid is None or bool(self.valid[r, c]):
                    yield self.cell_at(r, c)


@dataclass(frozen=True, slots=True)
class Constraint:
    """同一物理格点当前的交集约束与来源溯源。"""

    cell: Cell
    lower: float
    upper: float
    sources: tuple[tuple[str, int, tuple[int, int]], ...] = field(default_factory=tuple)
    conflict: bool = False

    def is_exact(self) -> bool:
        return not self.conflict and self.lower == self.upper


@dataclass(frozen=True, slots=True)
class ResampleTarget:
    """补测建议：一个值得重新采集的原始格点及理由。"""

    cell: Cell
    current_interval: tuple[float, float]
    role: str
    sources: tuple[tuple[str, int, tuple[int, int]], ...]


class ConflictError(ValueError):
    """加入窗口后约束交集为空。受影响格点被阻断，但不影响独立区域。"""

    def __init__(self, cells: Sequence[Cell]):
        self.cells = tuple(cells)
        pretty = ", ".join(".".join(map(str, c.as_tuple())) for c in cells[:8])
        super().__init__(f"观测冲突（交集为空）的格点：{pretty}")
