"""
Sweep timing & routing metrics for the PSUM buffer router (K-rows / C_out-cols).
Generates figures:
  1. Phase breakdown stacked-bar for representative ResNet-50 layers
  2. Compute efficiency vs network depth (layers ordered shallow → deep)
  3. Weight-load overhead % vs M (spatial positions) — shows amortization
  4. Sequential vs pipelined cycle comparison
  5. Tile count heatmap (K_flat vs C_out)
  6. Overhead % vs K_flat for different C_out

Run:  python sweep_timing.py
"""
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from im2col import SimConfig, utilization_k_rows_n_cols
from psum_router import compute_full_timing, ARRAY_SIZE, PIPELINE_DEPTH

SZ = ARRAY_SIZE

# ── representative layers ─────────────────────────────────────────────────
#  (label, N, H, W, Cin, Cout, R, S, stride, pad)
RESNET50_INFERENCE = [
    ("conv1\n7×7/2",      1, 224, 224,   3,   64, 7, 7, 2, 3),
    ("res2a\n1×1",        1,  56,  56,  64,   64, 1, 1, 1, 0),
    ("res2a\n3×3",        1,  56,  56,  64,   64, 3, 3, 1, 1),
    ("res3a\n3×3",        1,  28,  28, 128,  128, 3, 3, 1, 1),
    ("res4a\n3×3",        1,  14,  14, 256,  256, 3, 3, 1, 1),
    ("res5a\n3×3",        1,   7,   7, 512,  512, 3, 3, 1, 1),
    ("1×1 btl\n256→1024", 1,  14,  14, 256, 1024, 1, 1, 1, 0),
]

PHASE_COLORS = {
    "weight_load": "#4e79a7",
    "fill":        "#f28e2b",
    "compute":     "#59a14f",
    "drain":       "#e15759",
    "buffer_drain":"#76b7b2",
}
PHASE_LABELS = {
    "weight_load": "Weight load",
    "fill":        "Fill",
    "compute":     "Compute (MAC)",
    "drain":       "Drain",
    "buffer_drain":"Buffer drain",
}


def _timing(N, H, W, Cin, Cout, R, S, stride, pad, spatial_tile=32):
    cfg = SimConfig(N=N, H=H, W=W, C=Cin, K=Cout, R=R, S=S, stride=stride, pad=pad)
    return compute_full_timing(cfg, SZ, spatial_tile)


def _pipelined_total(tm):
    """Estimate pipelined total: overlap weight_load of tile i+1 with buffer_drain of tile i."""
    tiles = tm["tiles"]
    if not tiles:
        return 0
    total = tiles[0].total_cycles
    for tt in tiles[1:]:
        total += max(tt.weight_load_cycles,
                     tt.compute_cycles + tt.fill_cycles + tt.drain_cycles) + tt.buffer_drain_cycles
    return total


# ── Panel drawing functions (accept an ax so they work standalone or combined) ─

def draw_phase_breakdown(ax):
    labels, wl, fi, co, dr, bd = [], [], [], [], [], []
    for name, N, H, W, Cin, Cout, R, S, stride, pad in RESNET50_INFERENCE:
        tm = _timing(N, H, W, Cin, Cout, R, S, stride, pad)
        tot = tm["total_cycles"]
        labels.append(name)
        wl.append(tm["weight_load_total"] / tot * 100)
        fi.append(tm["fill_total"] / tot * 100)
        co.append(tm["compute_total"] / tot * 100)
        dr.append(tm["drain_total"] / tot * 100)
        bd.append(tm["buffer_drain_total"] / tot * 100)

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for vals, key in [(wl, "weight_load"), (fi, "fill"), (co, "compute"),
                      (dr, "drain"), (bd, "buffer_drain")]:
        ax.bar(x, vals, bottom=bottom, color=PHASE_COLORS[key],
               label=PHASE_LABELS[key], edgecolor="white", linewidth=0.5)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("% of total cycles", fontsize=7)
    ax.set_title("(a) Cycle Phase Breakdown — ResNet-50 (N=1, sequential)", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)


def draw_compute_efficiency(ax):
    labels, seq_eff, pipe_eff, utils = [], [], [], []
    for name, N, H, W, Cin, Cout, R, S, stride, pad in RESNET50_INFERENCE:
        cfg = SimConfig(N=N, H=H, W=W, C=Cin, K=Cout, R=R, S=S, stride=stride, pad=pad)
        tm = compute_full_timing(cfg, SZ, 32)
        K_flat = R * S * Cin
        Ho = (H + 2 * pad - R) // stride + 1
        Wo = (W + 2 * pad - S) // stride + 1
        M = N * Ho * Wo
        useful_macs = M * K_flat * Cout
        peak = SZ * SZ
        labels.append(name)
        seq_eff.append(useful_macs / (tm["total_cycles"] * peak) * 100)
        pipe_eff.append(useful_macs / (_pipelined_total(tm) * peak) * 100)
        u, _, _ = utilization_k_rows_n_cols(cfg)
        utils.append(u * 100)

    x = np.arange(len(labels))
    w = 0.28
    ax.bar(x - w, utils, w, label="PE utilization", color="#59a14f", edgecolor="black", linewidth=0.5)
    ax.bar(x, seq_eff, w, label="Compute eff. (seq)", color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, pipe_eff, w, label="Compute eff. (pipe)", color="#f28e2b", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("%", fontsize=7)
    ax.set_title("(b) PE Utilization vs Compute Efficiency", fontsize=8)
    ax.legend(fontsize=6)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)


def draw_overhead_vs_m(ax):
    hw_vals = [4, 7, 8, 14, 28, 56, 112]
    wl_pct, fill_drain_pct, bd_pct, comp_pct = [], [], [], []
    m_vals = []
    for hw in hw_vals:
        tm = _timing(1, hw, hw, 64, 64, 3, 3, 1, 1)
        Ho = hw + 2 - 3 + 1
        m_vals.append(Ho * Ho)
        tot = tm["total_cycles"]
        wl_pct.append(tm["weight_load_total"] / tot * 100)
        fill_drain_pct.append((tm["fill_total"] + tm["drain_total"]) / tot * 100)
        bd_pct.append(tm["buffer_drain_total"] / tot * 100)
        comp_pct.append(tm["compute_total"] / tot * 100)

    ax.stackplot(m_vals, wl_pct, fill_drain_pct, comp_pct, bd_pct,
                 labels=["Wt load", "Fill+Drain", "Compute", "Buf drain"],
                 colors=["#4e79a7", "#e15759", "#59a14f", "#76b7b2"], alpha=0.85)
    ax.set_xlabel("M (Ho × Wo)", fontsize=7)
    ax.set_ylabel("% of total cycles", fontsize=7)
    ax.set_title("(c) Phase Share vs Spatial Size M\n(Cin=64, Cout=64, 3×3)", fontsize=8)
    ax.legend(fontsize=5, loc="right")
    ax.set_ylim(0, 100)
    ax.set_xscale("log")
    ax.grid(axis="y", alpha=0.3)


def draw_seq_vs_pipe(ax):
    labels, seq, pipe = [], [], []
    for name, N, H, W, Cin, Cout, R, S, stride, pad in RESNET50_INFERENCE:
        tm = _timing(N, H, W, Cin, Cout, R, S, stride, pad)
        labels.append(name)
        seq.append(tm["total_cycles"])
        pipe.append(_pipelined_total(tm))

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, seq, w, label="Sequential", color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, pipe, w, label="Pipelined", color="#59a14f", edgecolor="black", linewidth=0.5)
    for i in range(len(labels)):
        speedup = seq[i] / pipe[i]
        ax.text(x[i] + w / 2, pipe[i], f"{speedup:.2f}×", va="bottom", ha="center", fontsize=5, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Total cycles", fontsize=7)
    ax.set_title("(d) Sequential vs Pipelined — ResNet-50 (N=1)", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(axis="y", alpha=0.3)


def draw_tile_heatmap(ax):
    k_flat_vals = [9, 16, 27, 32, 64, 128, 256, 576, 1152, 2304]
    cout_vals = [16, 32, 64, 128, 256, 512, 1024]
    grid = np.zeros((len(k_flat_vals), len(cout_vals)))
    for i, kf in enumerate(k_flat_vals):
        for j, co in enumerate(cout_vals):
            grid[i, j] = ceil(kf / SZ) * ceil(co / SZ)

    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", norm=plt.matplotlib.colors.LogNorm())
    ax.set_xticks(range(len(cout_vals)))
    ax.set_yticks(range(len(k_flat_vals)))
    ax.set_xticklabels(cout_vals, fontsize=6)
    ax.set_yticklabels(k_flat_vals, fontsize=6)
    ax.set_xlabel("C_out", fontsize=7)
    ax.set_ylabel("K_flat", fontsize=7)
    ax.set_title("(e) Weight Reloads (K_tiles × Cout_tiles)", fontsize=8)
    for i in range(len(k_flat_vals)):
        for j in range(len(cout_vals)):
            ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center", fontsize=5,
                    color="white" if grid[i, j] > grid.max() * 0.4 else "black")
    plt.colorbar(im, ax=ax, label="# tiles", fraction=0.046, pad=0.04)


def draw_eff_vs_spatial_tile(ax):
    tile_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    layers = [
        ("res2a (M=3136)", 1, 56, 56, 64, 64, 3, 3, 1, 1),
        ("res5a (M=49)",   1,  7,  7, 512, 512, 3, 3, 1, 1),
    ]
    colors_seq = ["#4e79a7", "#e15759"]
    colors_pipe = ["#59a14f", "#f28e2b"]

    for idx, (name, N, H, W, Cin, Cout, R, S, stride, pad) in enumerate(layers):
        cfg = SimConfig(N=N, H=H, W=W, C=Cin, K=Cout, R=R, S=S, stride=stride, pad=pad)
        K_flat = R * S * Cin
        Ho = (H + 2 * pad - R) // stride + 1
        Wo = (W + 2 * pad - S) // stride + 1
        M = N * Ho * Wo
        useful_macs = M * K_flat * Cout
        peak = SZ * SZ
        seq_eff, pipe_eff = [], []
        for ts in tile_sizes:
            tm = compute_full_timing(cfg, SZ, ts)
            seq_eff.append(useful_macs / (tm["total_cycles"] * peak) * 100)
            pipe_eff.append(useful_macs / (_pipelined_total(tm) * peak) * 100)
        ax.plot(tile_sizes, seq_eff, "o--", color=colors_seq[idx], label=f"{name} seq", markersize=4)
        ax.plot(tile_sizes, pipe_eff, "s-", color=colors_pipe[idx], label=f"{name} pipe", markersize=4)

    ax.set_xlabel("Spatial tile size", fontsize=7)
    ax.set_ylabel("Compute efficiency (%)", fontsize=7)
    ax.set_title("(f) Efficiency vs Spatial Tile Size", fontsize=8)
    ax.legend(fontsize=5)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 55)
    ax.grid(True, alpha=0.3)


# ── Standalone figure wrappers ────────────────────────────────────────────

def _standalone(draw_fn, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    plt.tight_layout()
    return fig

def fig_phase_breakdown():
    return _standalone(draw_phase_breakdown)

def fig_compute_efficiency():
    return _standalone(draw_compute_efficiency)

def fig_overhead_vs_m():
    return _standalone(draw_overhead_vs_m, figsize=(8, 5))

def fig_seq_vs_pipe():
    return _standalone(draw_seq_vs_pipe)

def fig_tile_heatmap():
    return _standalone(draw_tile_heatmap, figsize=(8, 6))

def fig_eff_vs_spatial_tile():
    return _standalone(draw_eff_vs_spatial_tile, figsize=(8, 5))


# ── Combined single-page figure ──────────────────────────────────────────

def fig_combined():
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    draw_phase_breakdown(fig.add_subplot(gs[0, 0]))
    draw_compute_efficiency(fig.add_subplot(gs[0, 1]))
    draw_overhead_vs_m(fig.add_subplot(gs[1, 0]))
    draw_seq_vs_pipe(fig.add_subplot(gs[1, 1]))
    draw_tile_heatmap(fig.add_subplot(gs[2, 0]))
    draw_eff_vs_spatial_tile(fig.add_subplot(gs[2, 1]))

    fig.suptitle(
        "PSUM Buffer Routing & Timing — Atalla 32×32 Systolic Array\n"
        "K-rows / C_out-cols mapping  ·  Direct-mapped routing (col c → buf c, no crossbar)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    return fig


def main():
    figs = [
        ("timing_phase_breakdown.png",   fig_phase_breakdown()),
        ("timing_compute_efficiency.png", fig_compute_efficiency()),
        ("timing_overhead_vs_m.png",      fig_overhead_vs_m()),
        ("timing_seq_vs_pipe.png",        fig_seq_vs_pipe()),
        ("timing_tile_heatmap.png",       fig_tile_heatmap()),
        ("timing_eff_vs_spatial_tile.png", fig_eff_vs_spatial_tile()),
    ]
    for path, fig in figs:
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")

    combined = fig_combined()
    combined.savefig("timing_combined.png", dpi=150, bbox_inches="tight")
    combined.savefig("timing_combined.pdf", bbox_inches="tight")
    plt.close(combined)
    print("Saved timing_combined.png")
    print("Saved timing_combined.pdf")


if __name__ == "__main__":
    main()
