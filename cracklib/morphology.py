"""可分块的区间张量形态学算子。

区间传播规则（设输入区间为 ``[lo, hi]``，结构元素支撑为 S，原图外取常量 c）：

* 腐蚀（输出取输入 r+offset 的 min）：
  ``erode_lo = min lo``，``erode_hi = min hi``
* 膨胀（反射偏移上取 max）：
  ``dil_lo = max lo``，``dil_hi = max hi``
* 闭运算 = 先腐蚀后膨胀（``footprint`` 是腐蚀偏移，膨胀用其反射）

对区间取 min/max 后仍保持 ``lo<=hi``，因此响应的上下界始终可查询。
分块只在原图外边界填充工艺常量，块与块的交界不引入任何边界：
闭运算的 halo 半径取完整支撑半径，裁掉 halo 后的结果与整域计算逐格点一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import LayerSpec, ProcessConfig
from .store import ObservationStore


def _gather(input_arr: np.ndarray, offset: tuple[int, int], origin: tuple[int, int],
            fill: float, global_shape: tuple[int, int]) -> np.ndarray:
    """输出 (r,c) 取输入画布的 (r+dr, c+dc)；按全局坐标判断原图外并填常量。"""
    dr, dc = offset
    h, w = input_arr.shape
    r0, c0 = origin
    gh, gw = global_shape
    out = np.full((h, w), fill, dtype=input_arr.dtype)
    # 输入画布局部 (sr,sc) 对应输出 (sr-dr, sc-dc)。
    sr_lo, sr_hi = max(0, dr), min(h, h + dr)
    sc_lo, sc_hi = max(0, dc), min(w, w + dc)
    if sr_hi <= sr_lo or sc_hi <= sc_lo:
        return out
    src = input_arr[sr_lo:sr_hi, sc_lo:sc_hi]
    dst_r_lo, dst_r_hi = sr_lo - dr, sr_hi - dr
    dst_c_lo, dst_c_hi = sc_lo - dc, sc_hi - dc
    # 该切片内每个输出格点引用的全局坐标。
    gr = np.arange(dst_r_lo, dst_r_hi)[:, None] + r0 + dr
    gc = np.arange(dst_c_lo, dst_c_hi)[None, :] + c0 + dc
    inside = (gr >= 0) & (gr < gh) & (gc >= 0) & (gc < gw)
    out[dst_r_lo:dst_r_hi, dst_c_lo:dst_c_hi] = np.where(inside, src, fill)
    return out


def _gather_blocked(blocked: np.ndarray, offset: tuple[int, int],
                    origin: tuple[int, int], global_shape: tuple[int, int]) -> np.ndarray:
    """阻断掩码的邻居：原图外视为未阻断（常量自身不阻断）。"""
    dr, dc = offset
    h, w = blocked.shape
    r0, c0 = origin
    gh, gw = global_shape
    out = np.zeros((h, w), dtype=bool)
    sr_lo, sr_hi = max(0, dr), min(h, h + dr)
    sc_lo, sc_hi = max(0, dc), min(w, w + dc)
    if sr_hi <= sr_lo or sc_hi <= sc_lo:
        return out
    dst_r_lo, dst_r_hi = sr_lo - dr, sr_hi - dr
    dst_c_lo, dst_c_hi = sc_lo - dc, sc_hi - dc
    gr = np.arange(dst_r_lo, dst_r_hi)[:, None] + r0 + dr
    gc = np.arange(dst_c_lo, dst_c_hi)[None, :] + c0 + dc
    inside = (gr >= 0) & (gr < gh) & (gc >= 0) & (gc < gw)
    out[dst_r_lo:dst_r_hi, dst_c_lo:dst_c_hi] = (
        blocked[sr_lo:sr_hi, sc_lo:sc_hi] & inside
    )
    return out


def interval_erode(lower, upper, blocked, spec: LayerSpec, outside: float,
                   origin: tuple[int, int] = (0, 0),
                   global_shape: tuple[int, int] | None = None):
    """区间腐蚀：对 footprint 偏移处的输入区间逐端点取 min。"""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    blocked = np.asarray(blocked, dtype=bool)
    global_shape = global_shape or lower.shape
    lo_acc = np.full(lower.shape, np.inf, dtype=np.float64)
    hi_acc = np.full(lower.shape, np.inf, dtype=np.float64)
    blk_acc = np.zeros(lower.shape, dtype=bool)
    for offset in spec.footprint:
        lo_acc = np.minimum(lo_acc, _gather(lower, offset, origin, outside, global_shape))
        hi_acc = np.minimum(hi_acc, _gather(upper, offset, origin, outside, global_shape))
        blk_acc |= _gather_blocked(blocked, offset, origin, global_shape)
    return lo_acc, hi_acc, blk_acc


def interval_dilate(lower, upper, blocked, spec: LayerSpec, outside: float,
                    origin: tuple[int, int] = (0, 0),
                    global_shape: tuple[int, int] | None = None):
    """区间膨胀：对反射偏移处的输入区间逐端点取 max。"""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    blocked = np.asarray(blocked, dtype=bool)
    global_shape = global_shape or lower.shape
    lo_acc = np.full(lower.shape, -np.inf, dtype=np.float64)
    hi_acc = np.full(lower.shape, -np.inf, dtype=np.float64)
    blk_acc = np.zeros(lower.shape, dtype=bool)
    for offset in spec.reflected_offsets:
        lo_acc = np.maximum(lo_acc, _gather(lower, offset, origin, outside, global_shape))
        hi_acc = np.maximum(hi_acc, _gather(upper, offset, origin, outside, global_shape))
        blk_acc |= _gather_blocked(blocked, offset, origin, global_shape)
    return lo_acc, hi_acc, blk_acc


def _crop_to_center(arr: np.ndarray, center_rows: tuple[int, int],
                    center_cols: tuple[int, int]) -> np.ndarray:
    """按中心区在画布中的实际起止位置裁切（首尾条带 halo 可能为 0）。"""
    r_lo, r_hi = center_rows
    c_lo, c_hi = center_cols
    return arr[r_lo:r_hi, c_lo:c_hi]


def interval_close_fused(lower, upper, blocked, spec: LayerSpec, outside: float,
                         origin: tuple[int, int] = (0, 0),
                         center_rows: tuple[int, int] | None = None,
                         center_cols: tuple[int, int] | None = None,
                         global_shape: tuple[int, int] | None = None):
    """整块区间闭运算。

    输入画布需自带 halo；给定 ``center_rows``/``center_cols``（画布局部下标）
    时，输出只保留中心区，分块交界因此不引入任何边界。
    """
    global_shape = global_shape or np.asarray(lower).shape
    e_lo, e_hi, e_blk = interval_erode(lower, upper, blocked, spec, outside,
                                       origin, global_shape)
    c_lo, c_hi, c_blk = interval_dilate(e_lo, e_hi, e_blk, spec, outside,
                                        origin, global_shape)
    if center_rows is not None:
        rows = center_rows
        cols = center_cols or (0, c_lo.shape[1])
        c_lo = _crop_to_center(c_lo, rows, cols)
        c_hi = _crop_to_center(c_hi, rows, cols)
        c_blk = _crop_to_center(c_blk, rows, cols)
    return c_lo, c_hi, c_blk


@dataclass
class ClosingResult:
    """一层的闭运算响应（上下界可查询）与溯源。"""

    layer: int
    lower: np.ndarray
    upper: np.ndarray
    blocked: np.ndarray
    threshold: float
    store_version: int
    provenance: dict

    def definite_mask(self) -> np.ndarray:
        """肯定达阈值：下界已达阈值。"""
        return (~self.blocked) & (self.lower >= self.threshold)

    def possible_mask(self) -> np.ndarray:
        """仍可能达阈值：上界尚未跌破阈值（含肯定达阈值）。"""
        return (~self.blocked) & (self.upper >= self.threshold)


class AllocationTracker:
    """记录分块过程中机台额外内存（临时数组）的峰值占用。"""

    def __init__(self):
        self.peak = 0
        self.current = 0
        self.events: list[tuple[str, int]] = []

    def full(self, shape, fill, dtype):
        return self._track(np.full(shape, fill, dtype=dtype))

    def zeros(self, shape, dtype):
        return self._track(np.zeros(shape, dtype=dtype))

    def _track(self, arr):
        nbytes = arr.nbytes
        self.current += nbytes
        self.peak = max(self.peak, self.current)
        self.events.append((f"alloc {arr.shape} {arr.dtype}", nbytes))
        return arr

    def release(self, arr):
        self.current -= arr.nbytes


class ChunkedClosingScheduler:
    """按水平条带分块的闭运算调度器，受机台 scratch 上限约束。

    条带之间在原图上重叠 halo，交界处不填任何边界；
    调用者无需先拼出整个张量——条带数据直接从 :class:`ObservationStore` 按需物化。
    """

    def __init__(self, config: ProcessConfig, store: ObservationStore,
                 scratch_bytes: int | None = None, allocator: AllocationTracker | None = None,
                 two_pass: bool = False):
        self.config = config
        self.store = store
        self.scratch_bytes = scratch_bytes if scratch_bytes is not None else config.scratch_bytes
        self.allocator = allocator or AllocationTracker()
        # 严格内存模式：腐蚀、膨胀分开跑，单遍只取 footprint 半径的 halo。
        self.two_pass = two_pass

    def _fused_band_rows(self, nc: int, rh: int, ch: int) -> int:
        """按 scratch 上限反算融合条带允许的中心行数（至少 1 行）。"""
        _, nr, _ = self.config.shape
        if self.scratch_bytes is None:
            return nr
        # 峰值临时画布：输入(lo/hi/blk)+腐蚀中间(lo/hi/blk)+闭运算工作(lo/hi/blk)
        per_row = (nc + 2 * ch) * (6 * 8 + 3)
        affordable = (self.scratch_bytes // per_row) - 2 * rh
        return max(1, min(nr, affordable))

    def run_layer(self, layer: int) -> ClosingResult:
        cfg = self.config
        spec = cfg.layer(layer)
        _, nr, nc = cfg.shape
        rh_full, ch_full = spec.support_radius()
        rh_half = max(abs(dr) for dr, _ in spec.footprint)
        ch_half = max(abs(dc) for _, dc in spec.footprint)
        track = self.allocator
        global_shape = (nr, nc)

        out_lo = np.empty((nr, nc), dtype=np.float64)
        out_hi = np.empty((nr, nc), dtype=np.float64)
        out_blk = np.zeros((nr, nc), dtype=bool)

        if not self.two_pass:
            band_h = self._fused_band_rows(nc, rh_full, ch_full)
        elif self.scratch_bytes is None:
            band_h = nr
        else:
            per_row = (nc + 2 * ch_half) * (3 * 8 + 2)
            band_h = max(1, min(nr, self.scratch_bytes // per_row - 2 * rh_half))

        r0 = 0
        while r0 < nr:
            r1 = min(nr, r0 + band_h)
            if not self.two_pass:
                halo_r0, halo_r1 = max(0, r0 - rh_full), min(nr, r1 + rh_full)
                region = self.store.materialize_layer(
                    layer, slice(halo_r0, halo_r1), slice(0, nc))
                work_lo = track.full(region["lower"].shape, np.nan, np.float64)
                work_hi = track.full(region["upper"].shape, np.nan, np.float64)
                work_blk = track.zeros(region["lower"].shape, bool)
                work_lo[:], work_hi[:], work_blk[:] = (
                    region["lower"], region["upper"], region["blocked"])
                c_lo, c_hi, c_blk = interval_close_fused(
                    work_lo, work_hi, work_blk, spec, cfg.outside,
                    origin=region["origin"],
                    center_rows=(r0 - halo_r0, r1 - halo_r0),
                    global_shape=global_shape,
                )
                for arr in (work_lo, work_hi, work_blk):
                    track.release(arr)
            else:
                halo_r0, halo_r1 = max(0, r0 - rh_half), min(nr, r1 + rh_half)
                region = self.store.materialize_layer(
                    layer, slice(halo_r0, halo_r1), slice(0, nc))
                work_lo = track.full(region["lower"].shape, np.nan, np.float64)
                work_hi = track.full(region["upper"].shape, np.nan, np.float64)
                work_blk = track.zeros(region["lower"].shape, bool)
                work_lo[:], work_hi[:], work_blk[:] = (
                    region["lower"], region["upper"], region["blocked"])
                # 腐蚀画布自带 footprint 半径，已覆盖膨胀反射支撑，整体膨胀后裁中心行。
                e_lo, e_hi, e_blk = interval_erode(
                    work_lo, work_hi, work_blk, spec, cfg.outside,
                    origin=region["origin"], global_shape=global_shape)
                for arr in (work_lo, work_hi, work_blk):
                    track.release(arr)
                c_lo, c_hi, c_blk = interval_dilate(
                    e_lo, e_hi, e_blk, spec, cfg.outside,
                    origin=region["origin"], global_shape=global_shape)
                pick_lo, pick_hi = r0 - halo_r0, r1 - halo_r0
                c_lo, c_hi, c_blk = (c_lo[pick_lo:pick_hi, :],
                                     c_hi[pick_lo:pick_hi, :],
                                     c_blk[pick_lo:pick_hi, :])
            out_lo[r0:r1, :] = c_lo
            out_hi[r0:r1, :] = c_hi
            out_blk[r0:r1, :] = c_blk
            r0 = r1

        return ClosingResult(
            layer=layer,
            lower=out_lo,
            upper=out_hi,
            blocked=out_blk,
            threshold=spec.threshold,
            store_version=self.store.version,
            provenance={"sources": "ObservationStore", "outside": cfg.outside,
                        "footprint": spec.footprint},
        )

    def run_all(self) -> dict[int, ClosingResult]:
        return {layer: self.run_layer(layer) for layer in range(self.config.num_layers)}
