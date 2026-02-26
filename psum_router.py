"""
PSUM Buffer Router & Cycle-Level Timing Model

Maps systolic array column outputs → 32 PSUM buffers (FIFOs).
Direct-mapped: column c → buffer c. No crossbar needed.

Timing per (K-tile, C_out-tile):
  weight_load:  sz cycles (preload weight tile via shared pass bus)
  fill:         sz cycles (activations fill the array diagonally before first valid output)
  compute:      M_tile cycles (one output row per cycle, all 32 cols simultaneously)
  drain:        sz cycles (last activations drain out, last valid outputs emerge)
  buffer_drain: ceil(M_tile / 1) cycles (drain accumulated PSUMs to VREGs, 1 entry/buf/cycle)

Total cycles for one (K-tile, C_out-tile, spatial-tile):
  weight_load + fill + M_tile + drain + buffer_drain

Raphael's RTL: value_ready asserted 2 cycles after start; all 32 cols output simultaneously.
We model fill=2 (pipeline depth) for simplicity; adjust if RTL changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import List, Optional, Dict

import numpy as np

ARRAY_SIZE = 32
PIPELINE_DEPTH = 2  # cycles from first activation entering to first valid output (from Raphael's code)


class Phase(Enum):
    WEIGHT_LOAD = "weight_load"
    FILL = "fill"
    COMPUTE = "compute"
    DRAIN = "drain"
    BUFFER_DRAIN = "buffer_drain"


@dataclass
class BufferEntry:
    spatial_pos: int  # m index
    n_b: int
    oh: int
    ow: int
    value: float


@dataclass
class PSUMBuffer:
    """One of 32 PSUM FIFOs. Accumulates across K-tiles, drains when all K-tiles done."""
    column: int
    depth: int = 64
    entries: Dict[int, float] = field(default_factory=dict)  # spatial_pos → accumulated value
    drain_order: List[int] = field(default_factory=list)  # order spatial positions arrived

    def accumulate(self, spatial_pos: int, value: float):
        if spatial_pos in self.entries:
            self.entries[spatial_pos] += value
        else:
            self.entries[spatial_pos] = value
            self.drain_order.append(spatial_pos)

    def drain_one(self) -> Optional[tuple]:
        """Pop oldest entry. Returns (spatial_pos, accumulated_value) or None."""
        if not self.drain_order:
            return None
        sp = self.drain_order.pop(0)
        val = self.entries.pop(sp)
        return sp, val

    @property
    def occupancy(self) -> int:
        return len(self.entries)

    def reset(self):
        self.entries.clear()
        self.drain_order.clear()


@dataclass
class TileTiming:
    """Cycle breakdown for one (K-tile, C_out-tile, spatial-tile)."""
    k_start: int
    n_start: int
    m_start: int
    m_count: int
    k_tile: int
    n_tile: int
    weight_load_cycles: int
    fill_cycles: int
    compute_cycles: int
    drain_cycles: int
    buffer_drain_cycles: int

    @property
    def total_cycles(self) -> int:
        return self.weight_load_cycles + self.fill_cycles + self.compute_cycles + self.drain_cycles + self.buffer_drain_cycles


@dataclass
class PSUMRouter:
    """
    Direct-mapped PSUM router for K-rows / C_out-cols mapping.
    column c → buffer c (no crossbar).

    Proof of conflict-freedom:
      Each cycle, columns 0..31 output simultaneously.
      buffer_index = column_index → all 32 targets are distinct.
    """
    sz: int = ARRAY_SIZE
    buffers: List[PSUMBuffer] = field(default_factory=list)
    tile_timings: List[TileTiming] = field(default_factory=list)
    route_log: List[dict] = field(default_factory=list)  # per-cycle routing events
    cycle: int = 0

    def __post_init__(self):
        if not self.buffers:
            self.buffers = [PSUMBuffer(column=c) for c in range(self.sz)]

    def reset(self):
        for buf in self.buffers:
            buf.reset()
        self.tile_timings.clear()
        self.route_log.clear()
        self.cycle = 0

    def route_output_row(self, column_values: np.ndarray, n_tile: int,
                         spatial_pos: int, phase: Phase):
        """Route one cycle's 32-column output to buffers. Direct-mapped: col c → buf c."""
        for c in range(min(n_tile, self.sz)):
            self.buffers[c].accumulate(spatial_pos, float(column_values[c]))
        self.route_log.append({
            "cycle": self.cycle,
            "phase": phase.value,
            "spatial_pos": spatial_pos,
            "n_active_cols": n_tile,
        })
        self.cycle += 1

    def drain_all(self, n_tile: int) -> List[List[tuple]]:
        """Drain all buffers. Returns list of (cycle_results), each a list of (buf, spatial_pos, value)."""
        all_results = []
        while any(self.buffers[c].occupancy > 0 for c in range(n_tile)):
            cycle_results = []
            for c in range(n_tile):
                result = self.buffers[c].drain_one()
                if result:
                    sp, val = result
                    cycle_results.append((c, sp, val))
            if not cycle_results:
                break
            all_results.append(cycle_results)
            self.route_log.append({
                "cycle": self.cycle,
                "phase": Phase.BUFFER_DRAIN.value,
                "drained": len(cycle_results),
            })
            self.cycle += 1
        return all_results


def compute_tile_timing(k_tile: int, n_tile: int, m_count: int,
                        k_start: int = 0, n_start: int = 0, m_start: int = 0,
                        sz: int = ARRAY_SIZE) -> TileTiming:
    """Cycle budget for one (K-tile, C_out-tile, spatial-tile)."""
    return TileTiming(
        k_start=k_start,
        n_start=n_start,
        m_start=m_start,
        m_count=m_count,
        k_tile=k_tile,
        n_tile=n_tile,
        weight_load_cycles=sz,
        fill_cycles=PIPELINE_DEPTH,
        compute_cycles=m_count,
        drain_cycles=PIPELINE_DEPTH,
        buffer_drain_cycles=m_count,
    )


def compute_full_timing(cfg, sz: int = ARRAY_SIZE, spatial_tile: int = 32) -> dict:
    """
    Full cycle-level timing for a convolution on the systolic array.
    Returns dict with per-tile breakdowns and totals.
    """
    K_flat = cfg.R * cfg.S * cfg.C
    Ho = (cfg.H + 2 * cfg.pad - cfg.R) // cfg.stride + 1
    Wo = (cfg.W + 2 * cfg.pad - cfg.S) // cfg.stride + 1
    M = cfg.N * Ho * Wo

    tiles = []
    total_cycles = 0

    for k_start in range(0, K_flat, sz):
        k_tile = min(sz, K_flat - k_start)
        for n_start in range(0, cfg.K, sz):
            n_tile = min(sz, cfg.K - n_start)
            for m_start in range(0, M, spatial_tile):
                m_count = min(spatial_tile, M - m_start)
                tt = compute_tile_timing(k_tile, n_tile, m_count,
                                         k_start, n_start, m_start, sz)
                tiles.append(tt)
                total_cycles += tt.total_cycles

    num_k_tiles = ceil(K_flat / sz)
    num_n_tiles = ceil(cfg.K / sz)
    num_m_tiles = ceil(M / spatial_tile)

    return {
        "tiles": tiles,
        "total_cycles": total_cycles,
        "num_k_tiles": num_k_tiles,
        "num_n_tiles": num_n_tiles,
        "num_m_tiles": num_m_tiles,
        "M": M,
        "K_flat": K_flat,
        "C_out": cfg.K,
        "weight_load_total": sum(t.weight_load_cycles for t in tiles),
        "fill_total": sum(t.fill_cycles for t in tiles),
        "compute_total": sum(t.compute_cycles for t in tiles),
        "drain_total": sum(t.drain_cycles for t in tiles),
        "buffer_drain_total": sum(t.buffer_drain_cycles for t in tiles),
    }


def simulate_with_routing(sim, mapping="k_rows_n_cols") -> dict:
    """
    Run the K-rows/C_out-cols sim with PSUM buffer routing.
    Returns {ofmap, router, timing, verified}.
    """
    from im2col import SPATIAL_TILE_SIZE, direct_conv_hwc

    cfg = sim.cfg
    sim.ofmap = np.zeros((cfg.N, sim.Ho, sim.Wo, cfg.K), dtype=float)
    padded = sim._pad_ifmap()
    K_flat = cfg.R * cfg.S * cfg.C
    M = cfg.N * sim.Ho * sim.Wo
    sz = cfg.array_size

    router = PSUMRouter(sz=sz)
    timing = compute_full_timing(cfg, sz, SPATIAL_TILE_SIZE)

    # Outer: C_out tiles → inner: K tiles, so buffers accumulate across all K-tiles
    # before draining for a given C_out tile.
    for n_start in range(0, cfg.K, sz):
        n_end = min(n_start + sz, cfg.K)
        n_tile = n_end - n_start

        for k_start in range(0, K_flat, sz):
            k_end = min(k_start + sz, K_flat)
            k_tile = k_end - k_start
            W_tile = np.zeros((sz, sz), dtype=float)
            W_block = sim.weights[:, :, :, n_start:n_end].reshape(K_flat, -1)
            W_tile[:k_tile, :n_tile] = W_block[k_start:k_end, :]

            router.cycle += sz  # weight_load phase

            for m_tile_start in range(0, M, SPATIAL_TILE_SIZE):
                m_tile_end = min(m_tile_start + SPATIAL_TILE_SIZE, M)
                router.cycle += PIPELINE_DEPTH  # fill phase

                for m in range(m_tile_start, m_tile_end):
                    n_b = m // (sim.Ho * sim.Wo)
                    oh = (m // sim.Wo) % sim.Ho
                    ow = m % sim.Wo
                    rf = sim._receptive_flat(padded, n_b, oh, ow)
                    in_row = np.zeros((sz,), dtype=float)
                    in_row[:k_tile] = rf[k_start:k_end]
                    out_row = in_row[:k_tile] @ W_tile[:k_tile, :n_tile]
                    router.route_output_row(out_row, n_tile, m, Phase.COMPUTE)

                router.cycle += PIPELINE_DEPTH  # drain phase

        # All K-tiles accumulated for this C_out tile; drain buffers → ofmap
        drained_cycles = router.drain_all(n_tile)
        for cycle_results in drained_cycles:
            for col, sp, val in cycle_results:
                n_b = sp // (sim.Ho * sim.Wo)
                oh = (sp // sim.Wo) % sim.Ho
                ow = sp % sim.Wo
                sim.ofmap[n_b, oh, ow, n_start + col] = val

        for buf in router.buffers:
            buf.reset()

    ref = direct_conv_hwc(sim.ifmap, sim.weights, stride=cfg.stride, pad=cfg.pad)
    diff = float(np.max(np.abs(ref - sim.ofmap)))

    return {
        "ofmap": sim.ofmap,
        "router": router,
        "timing": timing,
        "max_abs_diff": diff,
        "verified": diff < 1e-6,
    }
