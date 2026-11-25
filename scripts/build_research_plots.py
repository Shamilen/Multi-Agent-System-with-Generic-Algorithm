"""Generate publication-grade WaterWorld comparison charts for docs/ from saved experiment runs.

Reads JSON outputs from:
  - runs/waterworld/results.json       (MLP + SO(2) GA)
  - runs/waterworld_tx/results.json    (Ray-Transformer + SO(2) GA)
  - runs/waterworld_coop/results.json  (Hybrid QMIX + PPO-Coop Comm+Attn-GRU + SO(2) GA)

Usage:
  .\\venv\\Scripts\\python.exe scripts/build_research_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from waterworld_marl.scenarios import SCENARIOS


def _moving_avg(vals: list[float] | np.ndarray, win: int = 30) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    csum = np.cumsum(arr)
    for i in range(arr.size):
        start = max(0, i - win + 1)
        s = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        out[i] = s / (i - start + 1)
    return out


def build_waterworld_comparison(runs_root: Path, out_path: Path) -> None:
    ww_mlp = json.loads((runs_root / "waterworld" / "results.json").read_text(encoding="utf-8"))
    ww_tx = json.loads((runs_root / "waterworld_tx" / "results.json").read_text(encoding="utf-8"))
    ww_coop = json.loads((runs_root / "waterworld_coop" / "results.json").read_text(encoding="utf-8"))

    sc_keys = ["base", "single", "no_speed", "few_sensors", "more_poison"]
    sc_labels = [
        "base\n(5 ag, 30s)",
        "single\n(1 ag, 30s)",
        "no_speed\n(5 ag, 152d)",
        "few_sensors\n(5 ag, 10s)",
        "more_poison\n(3f / 25p)",
    ]

    plt.rcParams.update({"font.size": 9.0, "axes.titlesize": 10.6, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.8), dpi=170)
    fig.suptitle(
        "PettingZoo WaterWorld: MLP vs Ray-Transformer vs Cooperative QMIX / Attn-GRU MAPPO (20 pymunk Held-Out)",
        fontsize=13.2,
        fontweight="bold",
        y=0.985,
    )

    x = np.arange(len(sc_keys))
    palette = {
        "GA (SO(2))": "#2563EB",
        "DQN MLP": "#FBBF24",
        "DQN Ray-Tx": "#D97706",
        "QMIX Coop-Tx": "#92400E",
        "PPO MLP": "#34D399",
        "PPO Ray-Tx": "#059669",
        "PPO-Coop Tx": "#064E3B",
    }

    # (0, 0) Team Return across all 7 models
    ax = axes[0, 0]
    series_vals = {
        "GA (SO(2))": [ww_mlp["scenarios"][k]["heldout_20"]["GA"]["mean_team_return"] for k in sc_keys],
        "DQN MLP": [ww_mlp["scenarios"][k]["heldout_20"]["DQN"]["mean_team_return"] for k in sc_keys],
        "DQN Ray-Tx": [ww_tx["scenarios"][k]["heldout_20"]["DQN"]["mean_team_return"] for k in sc_keys],
        "QMIX Coop-Tx": [ww_coop["scenarios"][k]["heldout_20"]["DQN"]["mean_team_return"] for k in sc_keys],
        "PPO MLP": [ww_mlp["scenarios"][k]["heldout_20"]["PPO"]["mean_team_return"] for k in sc_keys],
        "PPO Ray-Tx": [ww_tx["scenarios"][k]["heldout_20"]["PPO"]["mean_team_return"] for k in sc_keys],
        "PPO-Coop Tx": [ww_coop["scenarios"][k]["heldout_20"]["PPO"]["mean_team_return"] for k in sc_keys],
    }
    w = 0.122
    offsets = np.linspace(-3.0, 3.0, len(series_vals)) * w
    for s_idx, (off, (name, vals)) in enumerate(zip(offsets, series_vals.items())):
        bars = ax.bar(x + off, vals, width=w, label=name, color=palette[name], edgecolor="#1F2937", linewidth=0.45)
        for k_idx, (b, v) in enumerate(zip(bars, vals)):
            if abs(v) >= 20:
                extra_y = 9.5 if (k_idx == 1 and name == "QMIX Coop-Tx") else 3.2
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + extra_y,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    fontweight="bold" if "Coop" in name else "normal",
                )
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x, sc_labels)
    ax.set_ylim(-25, 270)
    ax.set_ylabel("Mean Team Return (20 held-out eps)")
    ax.set_title("A. Held-Out Team Return Comparison (All Architectures)")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=7.5)

    # (0, 1) Cooperative Food Capture Boost (Before vs After Coop Upgrade)
    ax = axes[0, 1]
    f_series = {
        "DQN Ray-Tx (before)": ([ww_tx["scenarios"][k]["heldout_20"]["DQN"]["mean_food_events"] for k in sc_keys], palette["DQN Ray-Tx"]),
        "QMIX Coop-Tx (after)": ([ww_coop["scenarios"][k]["heldout_20"]["DQN"]["mean_food_events"] for k in sc_keys], palette["QMIX Coop-Tx"]),
        "PPO Ray-Tx (before)": ([ww_tx["scenarios"][k]["heldout_20"]["PPO"]["mean_food_events"] for k in sc_keys], palette["PPO Ray-Tx"]),
        "PPO-Coop Tx (after)": ([ww_coop["scenarios"][k]["heldout_20"]["PPO"]["mean_food_events"] for k in sc_keys], palette["PPO-Coop Tx"]),
    }
    w4 = 0.19
    offs4 = np.linspace(-1.5, 1.5, 4) * w4
    for off, (name, (vals, col)) in zip(offs4, f_series.items()):
        bars = ax.bar(x + off, vals, width=w4, label=name, color=col, edgecolor="#1F2937", linewidth=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.45, f"{v:.1f}", ha="center", va="bottom", fontsize=7.3)
    ax.set_xticks(x, sc_labels)
    ax.set_ylim(0, 27.5)
    ax.set_ylabel("Mean Food Captures / Episode")
    ax.set_title("B. Cooperative Food Captures: Baseline Ray-Tx vs Coop Upgrade")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right", frameon=True, fontsize=7.9)

    # (1, 0) Direct A/B Gain Delta (Coop Ray-Tx minus Baseline Ray-Tx)
    ax = axes[1, 0]
    dqn_delta = [
        ww_coop["scenarios"][k]["heldout_20"]["DQN"]["mean_team_return"]
        - ww_tx["scenarios"][k]["heldout_20"]["DQN"]["mean_team_return"]
        for k in sc_keys
    ]
    ppo_delta = [
        ww_coop["scenarios"][k]["heldout_20"]["PPO"]["mean_team_return"]
        - ww_tx["scenarios"][k]["heldout_20"]["PPO"]["mean_team_return"]
        for k in sc_keys
    ]
    dqn_pos = sum(1 for v in dqn_delta if v > 0)
    ppo_pos = sum(1 for v in ppo_delta if v > 0)
    w2 = 0.34
    b1 = ax.bar(x - w2 / 2, dqn_delta, width=w2, color="#B45309", edgecolor="#1F2937", linewidth=0.6, label=f"DQN -> QMIX Delta ({dqn_pos}/5 improved)")
    b2 = ax.bar(x + w2 / 2, ppo_delta, width=w2, color="#047857", edgecolor="#1F2937", linewidth=0.6, label=f"PPO -> PPO-Coop Delta ({ppo_pos}/5 improved)")
    for bars_g in (b1, b2):
        for b in bars_g:
            v = b.get_height()
            va = "bottom" if v >= 0 else "top"
            pad = 0.9 if v >= 0 else -0.9
            ax.text(b.get_x() + b.get_width() / 2.0, v + pad, f"{v:+.1f}", ha="center", va=va, fontsize=8.0, fontweight="bold")
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xticks(x, sc_labels)
    ax.set_ylim(-22, 68)
    ax.set_ylabel("Team Return Delta (Coop - Baseline Ray-Tx)")
    ax.set_title("C. Direct A/B Cooperation Improvement Over Baseline Ray-Transformer")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right", frameon=True, fontsize=8.0)

    # (1, 1) Wall-clock training time per 600-ep run
    ax = axes[1, 1]
    t_series = {
        "GA (SO(2))": ([ww_coop["scenarios"][k]["runs"]["GA"]["elapsed_sec"] for k in sc_keys], palette["GA (SO(2))"]),
        "DQN Ray-Tx": ([ww_tx["scenarios"][k]["runs"]["DQN"]["elapsed_sec"] for k in sc_keys], palette["DQN Ray-Tx"]),
        "QMIX Coop-Tx": ([ww_coop["scenarios"][k]["runs"]["DQN"]["elapsed_sec"] for k in sc_keys], palette["QMIX Coop-Tx"]),
        "PPO Ray-Tx": ([ww_tx["scenarios"][k]["runs"]["PPO"]["elapsed_sec"] for k in sc_keys], palette["PPO Ray-Tx"]),
        "PPO-Coop Tx": ([ww_coop["scenarios"][k]["runs"]["PPO"]["elapsed_sec"] for k in sc_keys], palette["PPO-Coop Tx"]),
    }
    w5 = 0.16
    offs5 = np.linspace(-2.0, 2.0, 5) * w5
    for off, (name, (vals, col)) in zip(offs5, t_series.items()):
        bars = ax.bar(x + off, vals, width=w5, label=name, color=col, edgecolor="#1F2937", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 2.2, f"{v:.0f}s", ha="center", va="bottom", fontsize=6.8)
    tx_sum = sum(ww_tx["scenarios"][k]["runs"][a]["elapsed_sec"] for k in sc_keys for a in ("GA", "DQN", "PPO"))
    cp_sum = sum(ww_coop["scenarios"][k]["runs"][a]["elapsed_sec"] for k in sc_keys for a in ("GA", "DQN", "PPO"))
    ax.set_xticks(x, sc_labels)
    ax.set_ylim(0, 215)
    ax.set_ylabel("Training Time per 600-ep Run (seconds)")
    ax.set_title(f"D. Training Wall-Clock Time (Total 5x3: Ray-Tx {tx_sum:.1f}s vs Coop {cp_sum:.1f}s)")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=7.6)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def build_waterworld_learning_curves(runs_root: Path, out_path: Path) -> None:
    ww_mlp = json.loads((runs_root / "waterworld" / "results.json").read_text(encoding="utf-8"))
    ww_tx = json.loads((runs_root / "waterworld_tx" / "results.json").read_text(encoding="utf-8"))
    ww_coop = json.loads((runs_root / "waterworld_coop" / "results.json").read_text(encoding="utf-8"))

    sc_keys = ["base", "single", "no_speed", "few_sensors", "more_poison"]
    plt.rcParams.update({"font.size": 9.1, "axes.titlesize": 10.3, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 10.0), dpi=165)
    fig.suptitle(
        "WaterWorld Learning Dynamics (600 GPU Episodes): Baseline Ray-Tx vs Cooperative QMIX & Attn-GRU MAPPO",
        fontsize=13.2,
        fontweight="bold",
        y=0.985,
    )

    flat_axes = axes.ravel()
    for idx, sc_key in enumerate(sc_keys):
        ax = flat_axes[idx]
        sc = SCENARIOS[sc_key]
        sc_mlp = ww_mlp["scenarios"][sc_key]
        sc_tx = ww_tx["scenarios"][sc_key]
        sc_cp = ww_coop["scenarios"][sc_key]

        zero_ret = sc_tx["baselines"]["Zero"]["mean_team_return"]
        ax.axhline(zero_ret, color="#6B7280", ls=":", lw=1.1, label=f"Zero ({zero_ret:+.1f})")
        if sc.n_coop > 1:
            ax.axvline(96, color="#9333EA", ls=":", lw=1.0, alpha=0.7, label="PPO n_coop 1->2" if idx == 0 else None)

        curves = [
            ("GA (SO(2))", sc_mlp["runs"]["GA"], "#2563EB", "-", "o", 1.6),
            ("DQN (Ray-Tx)", sc_tx["runs"]["DQN"], "#F59E0B", "--", "s", 1.5),
            ("QMIX (Coop-Tx)", sc_cp["runs"]["DQN"], "#92400E", "-", "^", 1.9),
            ("PPO (Ray-Tx)", sc_tx["runs"]["PPO"], "#10B981", "--", "s", 1.5),
            ("PPO-Coop (Attn-GRU)", sc_cp["runs"]["PPO"], "#064E3B", "-", "D", 2.0),
        ]
        for label, src, col, ls, mk, lw in curves:
            ep_x = [e["episode"] for e in src["episode_curve"]]
            ep_y = _moving_avg([e["team_return"] for e in src["episode_curve"]], 30)
            ev_x = [e["episode"] for e in src["eval_curve"]]
            ev_y = [e["mean_team_return"] for e in src["eval_curve"]]
            ax.plot(ep_x, ep_y, color=col, ls=ls, alpha=0.14, lw=0.9)
            ax.plot(ev_x, ev_y, color=col, ls=ls, marker=mk, ms=3.5, lw=lw, label=label)

        ax.set_title(f"{sc_key}: {sc.title_ru}\n(A={sc.n_pursuers}, n_coop={sc.n_coop}, obs={sc.obs_dim})")
        ax.set_xlabel("Training Episodes")
        ax.set_ylabel("Team Return")
        ax.grid(True, alpha=0.28)
        if idx == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=7.3)

    ax = flat_axes[5]
    ax.axis("off")
    lines = ["Held-out 20-ep Team Return (Ray-Tx -> Coop):\n"]
    for k in sc_keys:
        ht = ww_tx["scenarios"][k]["heldout_20"]
        hc = ww_coop["scenarios"][k]["heldout_20"]
        lines.append(
            f"{k:11s} | DQN {ht['DQN']['mean_team_return']:+6.1f} -> {hc['DQN']['mean_team_return']:+6.1f} | "
            f"PPO {ht['PPO']['mean_team_return']:+6.1f} -> {hc['PPO']['mean_team_return']:+6.1f}"
        )
    lines.append("\nHighlights:")
    lines.append("• QMIX improves over Ray-Tx DQN on 5/5 scenarios")
    lines.append("• PPO-Coop (Salience+Comm+Dodge) beats Ray-Tx on 4/5")
    lines.append("  & hits #1 all-time base (+108.3), no_speed (+78.8),")
    lines.append("  5-sc avg (+94.4 vs +78.7 MLP / +76.9 Ray-Tx)")
    lines.append("• Poison hits cut 3.8x avg (more_poison: 18.1 -> 1.6)")
    ax.text(
        0.02,
        0.92,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def build_acceleration_summary(runs_root: Path, out_path: Path) -> None:
    ww_mlp = json.loads((runs_root / "waterworld" / "results.json").read_text(encoding="utf-8"))
    ww_tx = json.loads((runs_root / "waterworld_tx" / "results.json").read_text(encoding="utf-8"))
    ww_coop = json.loads((runs_root / "waterworld_coop" / "results.json").read_text(encoding="utf-8"))

    mlp_ga = sum(v["runs"]["GA"]["elapsed_sec"] for v in ww_mlp["scenarios"].values())
    mlp_dqn = sum(v["runs"]["DQN"]["elapsed_sec"] for v in ww_mlp["scenarios"].values())
    mlp_ppo = sum(v["runs"]["PPO"]["elapsed_sec"] for v in ww_mlp["scenarios"].values())

    tx_ga = sum(v["runs"]["GA"]["elapsed_sec"] for v in ww_tx["scenarios"].values())
    tx_dqn = sum(v["runs"]["DQN"]["elapsed_sec"] for v in ww_tx["scenarios"].values())
    tx_ppo = sum(v["runs"]["PPO"]["elapsed_sec"] for v in ww_tx["scenarios"].values())

    cp_ga = sum(v["runs"]["GA"]["elapsed_sec"] for v in ww_coop["scenarios"].values())
    cp_dqn = sum(v["runs"]["DQN"]["elapsed_sec"] for v in ww_coop["scenarios"].values())
    cp_ppo = sum(v["runs"]["PPO"]["elapsed_sec"] for v in ww_coop["scenarios"].values())

    plt.rcParams.update({"font.size": 9.3, "axes.titlesize": 10.8, "axes.titleweight": "bold"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.6, 5.9), dpi=170)
    fig.suptitle(
        "WaterWorld GPU & CUDA Graph Simulation Engineering Speedups (NVIDIA GeForce RTX 4080)",
        fontsize=13.3,
        fontweight="bold",
        y=0.98,
    )

    envs = [
        "CPU pymunk\n(NumPy sensors)\nB=1",
        "GPU Eager PyTorch\nVectorizedWaterworld\nB=1024",
        "GPU CUDA Graph\nVectorizedWaterworld\nB=256",
        "GPU CUDA Graph\nVectorizedWaterworld\nB=1024",
    ]
    sps = [1858.0, 297_793.0, 568_996.0, 1_389_224.0]
    cols = ["#9CA3AF", "#F59E0B", "#2563EB", "#059669"]
    xb = np.arange(len(envs))
    bars = ax1.bar(xb, sps, width=0.58, color=cols, edgecolor="#1F2937", linewidth=0.7)
    ax1.set_yscale("log")
    ax1.set_ylim(100, 8_000_000)
    ax1.set_xticks(xb, envs, fontsize=8.4)
    ax1.set_ylabel("Environment Throughput (env-steps / s, log scale)")
    ax1.set_title("A. WaterWorld Simulator Throughput Scaling (base, 500 cycles)")
    ax1.grid(True, axis="y", alpha=0.28)
    notes = [
        "269.1 ms/ep\n1.0x baseline",
        "1.679 ms/ep\n160x vs CPU",
        "0.879 ms/ep\n306x vs CPU",
        "0.360 ms/ep\n748x vs CPU (4.7x vs Eager)",
    ]
    for b, val, note in zip(bars, sps, notes):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            val * 1.18,
            f"{val:,.0f} step/s\n{note}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )

    configs = [
        "CPU Baseline\n(pymunk ref.)",
        "GPU CUDA Graph\nMLP + GA",
        "GPU CUDA Graph\nRay-Tx + GA",
        "GPU CUDA Graph\nCoop QMIX/PPO + GA",
    ]
    ax2.bar([0], [3250.8], width=0.52, color="#9CA3AF", edgecolor="#1F2937", linewidth=0.7, label="CPU Baseline (~54.2 min)")
    ax2.bar([1], [mlp_ga], width=0.52, color="#2563EB", edgecolor="#1F2937", linewidth=0.7, label="GA (5 scenarios)")
    ax2.bar([1], [mlp_dqn], bottom=[mlp_ga], width=0.52, color="#F59E0B", edgecolor="#1F2937", linewidth=0.7, label="DQN / QMIX")
    ax2.bar([1], [mlp_ppo], bottom=[mlp_ga + mlp_dqn], width=0.52, color="#10B981", edgecolor="#1F2937", linewidth=0.7, label="PPO / PPO-Coop")

    ax2.bar([2], [tx_ga], width=0.52, color="#2563EB", edgecolor="#1F2937", linewidth=0.7)
    ax2.bar([2], [tx_dqn], bottom=[tx_ga], width=0.52, color="#D97706", edgecolor="#1F2937", linewidth=0.7)
    ax2.bar([2], [tx_ppo], bottom=[tx_ga + tx_dqn], width=0.52, color="#059669", edgecolor="#1F2937", linewidth=0.7)

    ax2.bar([3], [cp_ga], width=0.52, color="#2563EB", edgecolor="#1F2937", linewidth=0.7)
    ax2.bar([3], [cp_dqn], bottom=[cp_ga], width=0.52, color="#92400E", edgecolor="#1F2937", linewidth=0.7)
    ax2.bar([3], [cp_ppo], bottom=[cp_ga + cp_dqn], width=0.52, color="#064E3B", edgecolor="#1F2937", linewidth=0.7)

    ax2.set_xticks([0, 1, 2, 3], configs, fontsize=8.3)
    ax2.set_ylim(0, 3950)
    ax2.set_ylabel("Total Wall-Clock Time Across 15 Runs (seconds)")
    ax2.set_title("B. Complete 5x3 WaterWorld Training Suite Time")
    ax2.grid(True, axis="y", alpha=0.28)
    mlp_tot = mlp_ga + mlp_dqn + mlp_ppo
    tx_tot = tx_ga + tx_dqn + tx_ppo
    cp_tot = cp_ga + cp_dqn + cp_ppo
    ax2.text(0, 3250.8 + 70, "3,250.8s\n(54.2 min)", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax2.text(1, mlp_tot + 70, f"{mlp_tot:.1f}s\n(8.2m, 6.6x)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#065F46")
    ax2.text(2, tx_tot + 70, f"{tx_tot:.1f}s\n(12.0m, 4.5x)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#92400E")
    ax2.text(3, cp_tot + 70, f"{cp_tot:.1f}s\n({cp_tot/60:.1f}m, {3250.8/cp_tot:.1f}x)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#064E3B")
    ax2.legend(loc="upper right", ncol=2, frameon=True, fontsize=7.8)

    plt.tight_layout(rect=[0, 0, 1, 0.935])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    runs_root = Path("runs")
    docs_root = Path("docs")
    build_waterworld_comparison(runs_root, docs_root / "waterworld_mlp_vs_transformer.png")
    build_waterworld_learning_curves(runs_root, docs_root / "waterworld_learning_curves.png")
    build_acceleration_summary(runs_root, docs_root / "acceleration_summary.png")
    print("Generated all 3 WaterWorld research comparison charts in docs/")


if __name__ == "__main__":
    main()
