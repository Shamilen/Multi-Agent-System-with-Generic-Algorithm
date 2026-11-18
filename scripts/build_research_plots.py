"""Generate publication-grade comparison charts for docs/ from saved experiment runs.

Reads JSON outputs from:
  - runs/pursuit_ga_eq/{config,history,eval_summary}.json
  - runs/pursuit_ga_transformer/{config,history,eval_summary}.json
  - runs/pursuit_behavioral_470k.json (optional behavioral breakdown; falls back safely)
  - runs/waterworld/results.json
  - runs/waterworld_tx/results.json

Usage:
  .\\venv\\Scripts\\python.exe scripts/build_research_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pursuit_ga.config import EnvConfig, ModelConfig
from pursuit_ga.networks import PursuitPolicyNet
from waterworld_marl.algos.dqn import DuelingQNet
from waterworld_marl.algos.ga import WaterworldGAPolicy
from waterworld_marl.algos.ppo import ActorCriticPPO
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


def build_pursuit_comparison(runs_root: Path, out_path: Path) -> None:
    eq_dir = runs_root / "pursuit_ga_eq"
    tx_dir = runs_root / "pursuit_ga_transformer"
    hist_eq = json.loads((eq_dir / "history.json").read_text(encoding="utf-8"))
    hist_tx = json.loads((tx_dir / "history.json").read_text(encoding="utf-8"))
    eval_eq = json.loads((eq_dir / "eval_summary.json").read_text(encoding="utf-8"))
    eval_tx = json.loads((tx_dir / "eval_summary.json").read_text(encoding="utf-8"))

    beh_path = runs_root / "pursuit_behavioral_470k.json"
    beh = json.loads(beh_path.read_text(encoding="utf-8")) if beh_path.exists() else {}

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=170)
    fig.suptitle(
        "Pursuit (sisl/pursuit-v5): C4-Equivariant CNN-GRU vs C4-Equivariant Vision-Transformer",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    gens_eq = [g["generation"] for g in hist_eq]
    gens_tx = [g["generation"] for g in hist_tx]
    c_eq = "#1F6FEB"
    c_tx = "#D97706"

    # (0, 0) Mean Agent Return over 70 generations
    ax = axes[0, 0]
    ax.plot(gens_eq, [g["best_return"] for g in hist_eq], color=c_eq, lw=2.0, label="eq_cnn_gru (Gen Best)")
    ax.plot(gens_eq, [g["mean_return"] for g in hist_eq], color=c_eq, lw=1.4, ls="--", alpha=0.75, label="eq_cnn_gru (Pop Mean)")
    ax.plot(gens_tx, [g["best_return"] for g in hist_tx], color=c_tx, lw=2.0, label="eq_transformer (Gen Best)")
    ax.plot(gens_tx, [g["mean_return"] for g in hist_tx], color=c_tx, lw=1.4, ls="--", alpha=0.75, label="eq_transformer (Pop Mean)")
    ax.set_title("Training Convergence: Mean Agent Return per Episode")
    ax.set_xlabel("Generation (pop=96, 3 ep/genome, 500 cycles)")
    ax.set_ylabel("Mean Agent Return")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)

    # (0, 1) Evaders Caught (/30) over 70 generations
    ax = axes[0, 1]
    ax.plot(gens_eq, [g["best_evaders_caught"] for g in hist_eq], color=c_eq, lw=2.0, label="eq_cnn_gru (8,954 params) Best")
    ax.plot(gens_eq, [g["mean_evaders_caught"] for g in hist_eq], color=c_eq, lw=1.4, ls="--", alpha=0.75, label="eq_cnn_gru Pop Mean")
    ax.plot(gens_tx, [g["best_evaders_caught"] for g in hist_tx], color=c_tx, lw=2.0, label="eq_transformer (7,414 params) Best")
    ax.plot(gens_tx, [g["mean_evaders_caught"] for g in hist_tx], color=c_tx, lw=1.4, ls="--", alpha=0.75, label="eq_transformer Pop Mean")
    ax.axhline(30, color="#6B7280", ls=":", lw=1.2, label="Max Evaders (30)")
    ax.set_ylim(0, 32)
    ax.set_title("Training Convergence: Evaders Captured (/ 30)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Evaders Caught")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)

    # (1, 0) Held-out 20-seed Benchmark (470000..470019)
    ax = axes[1, 0]
    b_eq = eval_eq["benchmarks"]
    b_tx = eval_tx["benchmarks"]
    labels = [
        "Random\nBaseline",
        "Greedy\nHeuristic",
        "eq_cnn_gru\nArgmax",
        "eq_cnn_gru\nStoch T=0.15",
        "eq_transformer\nArgmax",
        "eq_transformer\nStoch T=0.15",
    ]
    caught_vals = [
        b_eq["random"]["mean_evaders_caught"],
        b_eq["greedy_heuristic"]["mean_evaders_caught"],
        b_eq["trained_ga_argmax"]["mean_evaders_caught"],
        b_eq["trained_ga_stochastic"]["mean_evaders_caught"],
        b_tx["trained_ga_argmax"]["mean_evaders_caught"],
        b_tx["trained_ga_stochastic"]["mean_evaders_caught"],
    ]
    caught_stds = [
        b_eq["random"]["std_evaders_caught"],
        b_eq["greedy_heuristic"]["std_evaders_caught"],
        b_eq["trained_ga_argmax"]["std_evaders_caught"],
        b_eq["trained_ga_stochastic"]["std_evaders_caught"],
        b_tx["trained_ga_argmax"]["std_evaders_caught"],
        b_tx["trained_ga_stochastic"]["std_evaders_caught"],
    ]
    ret_vals = [
        b_eq["random"]["mean_agent_return"],
        b_eq["greedy_heuristic"]["mean_agent_return"],
        b_eq["trained_ga_argmax"]["mean_agent_return"],
        b_eq["trained_ga_stochastic"]["mean_agent_return"],
        b_tx["trained_ga_argmax"]["mean_agent_return"],
        b_tx["trained_ga_stochastic"]["mean_agent_return"],
    ]
    clear_vals = [
        b_eq["random"]["full_clear_rate_pct"],
        b_eq["greedy_heuristic"]["full_clear_rate_pct"],
        b_eq["trained_ga_argmax"]["full_clear_rate_pct"],
        b_eq["trained_ga_stochastic"]["full_clear_rate_pct"],
        b_tx["trained_ga_argmax"]["full_clear_rate_pct"],
        b_tx["trained_ga_stochastic"]["full_clear_rate_pct"],
    ]
    bar_colors = ["#9CA3AF", "#6B7280", "#60A5FA", c_eq, "#F59E0B", c_tx]
    x = np.arange(len(labels))
    bars = ax.bar(x, caught_vals, yerr=caught_stds, capsize=4, color=bar_colors, edgecolor="#1F2937", linewidth=0.7)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, 34)
    ax.set_ylabel("Mean Evaders Caught (/ 30)")
    ax.set_title("Held-Out Benchmark (20 unseen seeds: 470000..470019)")
    ax.grid(True, axis="y", alpha=0.28)
    for bar, std, c_val, r_val, fc_val in zip(bars, caught_stds, caught_vals, ret_vals, clear_vals):
        pct = 100.0 * c_val / 30.0
        extra = f"\nclear {fc_val:.0f}%" if fc_val > 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std + 0.5,
            f"{c_val:.2f}/30 ({pct:.1f}%)\nret {r_val:+.1f}{extra}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )

    # (1, 1) Action Balance & Overlaps/Step
    ax = axes[1, 1]
    act_names = ["LEFT", "RIGHT", "DOWN", "UP", "STAY"]
    if beh:
        series = [
            ("eq_cnn_gru (Stoch)", beh["eq_cnn_gru"]["modes"]["stochastic"], c_eq),
            ("eq_cnn_gru (Argmax)", beh["eq_cnn_gru"]["modes"]["greedy_argmax"], "#93C5FD"),
            ("eq_transformer (Argmax)", beh["eq_transformer"]["modes"]["greedy_argmax"], c_tx),
            ("eq_transformer (Stoch)", beh["eq_transformer"]["modes"]["stochastic"], "#FCD34D"),
        ]
        width = 0.19
        ax_x = np.arange(len(act_names))
        for idx, (lbl, m, col) in enumerate(series):
            cnts = np.asarray(m["action_counts"], dtype=np.float64)
            pcts = 100.0 * cnts / max(cnts.sum(), 1.0)
            ov = m["overlaps_per_step"]
            offset = (idx - 1.5) * width
            ax.bar(
                ax_x + offset,
                pcts,
                width=width,
                color=col,
                edgecolor="#1F2937",
                linewidth=0.5,
                label=f"{lbl} | ov/step={ov:.3f}",
            )
        ax.axhline(25.0, color="#6B7280", ls="--", lw=1.0, alpha=0.7, label="Ideal 4-dir Uniform (25%)")
        ax.set_xticks(ax_x, act_names)
        ax.set_ylim(0, 34)
        ax.set_ylabel("Action Share (%)")
        ax.set_title("C4 Action Symmetry & Agent Overlaps per Step (470000..470019)")
        ax.grid(True, axis="y", alpha=0.28)
        ax.legend(loc="upper right", frameon=True, fontsize=8.0)
    else:
        ax.text(0.5, 0.5, "Behavioral JSON not found", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def build_waterworld_comparison(runs_root: Path, out_path: Path) -> None:
    ww_mlp = json.loads((runs_root / "waterworld" / "results.json").read_text(encoding="utf-8"))
    ww_tx = json.loads((runs_root / "waterworld_tx" / "results.json").read_text(encoding="utf-8"))

    sc_keys = ["base", "single", "no_speed", "few_sensors", "more_poison"]
    sc_labels = ["base\n(5 ag, 30s)", "single\n(1 ag, 30s)", "no_speed\n(5 ag, 152d)", "few_sensors\n(5 ag, 10s)", "more_poison\n(3f / 25p)"]

    plt.rcParams.update({"font.size": 9.2, "axes.titlesize": 10.8, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6), dpi=170)
    fig.suptitle(
        "PettingZoo WaterWorld: MLP vs Ray-Transformer vs SO(2) Equivariant GA (600 GPU eps -> 20 pymunk Held-Out)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    x = np.arange(len(sc_keys))
    w = 0.145
    palette = {
        "Zero": "#9CA3AF",
        "GA": "#2563EB",
        "DQN MLP": "#F59E0B",
        "DQN Ray-Tx": "#B45309",
        "PPO MLP": "#10B981",
        "PPO Ray-Tx": "#065F46",
    }

    # (0, 0) Team Return
    ax = axes[0, 0]
    series_vals = {
        "Zero": [ww_tx["scenarios"][k]["baselines"]["Zero"]["mean_team_return"] for k in sc_keys],
        "GA": [ww_mlp["scenarios"][k]["heldout_20"]["GA"]["mean_team_return"] for k in sc_keys],
        "DQN MLP": [ww_mlp["scenarios"][k]["heldout_20"]["DQN"]["mean_team_return"] for k in sc_keys],
        "DQN Ray-Tx": [ww_tx["scenarios"][k]["heldout_20"]["DQN"]["mean_team_return"] for k in sc_keys],
        "PPO MLP": [ww_mlp["scenarios"][k]["heldout_20"]["PPO"]["mean_team_return"] for k in sc_keys],
        "PPO Ray-Tx": [ww_tx["scenarios"][k]["heldout_20"]["PPO"]["mean_team_return"] for k in sc_keys],
    }
    offsets = np.linspace(-2.5, 2.5, len(series_vals)) * w
    for off, (name, vals) in zip(offsets, series_vals.items()):
        bars = ax.bar(x + off, vals, width=w, label=name, color=palette[name], edgecolor="#1F2937", linewidth=0.5)
        for b, v in zip(bars, vals):
            if abs(v) >= 15:
                y_pos = v + (4 if v >= 0 else -12)
                ax.text(b.get_x() + b.get_width() / 2.0, y_pos, f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6.8)
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x, sc_labels)
    ax.set_ylim(-30, 265)
    ax.set_ylabel("Mean Team Return (20 held-out eps)")
    ax.set_title("A. Held-Out Team Return Comparison")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=8.0)

    # (0, 1) Food captures vs Poison hits (MLP vs Ray-Tx vs GA)
    ax = axes[0, 1]
    food_groups = {
        "GA Food": ([ww_mlp["scenarios"][k]["heldout_20"]["GA"]["mean_food_events"] for k in sc_keys], palette["GA"]),
        "DQN MLP Food": ([ww_mlp["scenarios"][k]["heldout_20"]["DQN"]["mean_food_events"] for k in sc_keys], palette["DQN MLP"]),
        "DQN Ray-Tx Food": ([ww_tx["scenarios"][k]["heldout_20"]["DQN"]["mean_food_events"] for k in sc_keys], palette["DQN Ray-Tx"]),
        "PPO MLP Food": ([ww_mlp["scenarios"][k]["heldout_20"]["PPO"]["mean_food_events"] for k in sc_keys], palette["PPO MLP"]),
        "PPO Ray-Tx Food": ([ww_tx["scenarios"][k]["heldout_20"]["PPO"]["mean_food_events"] for k in sc_keys], palette["PPO Ray-Tx"]),
    }
    poison_vals = {
        "GA Food": [ww_mlp["scenarios"][k]["heldout_20"]["GA"]["mean_poison_hits"] for k in sc_keys],
        "DQN MLP Food": [ww_mlp["scenarios"][k]["heldout_20"]["DQN"]["mean_poison_hits"] for k in sc_keys],
        "DQN Ray-Tx Food": [ww_tx["scenarios"][k]["heldout_20"]["DQN"]["mean_poison_hits"] for k in sc_keys],
        "PPO MLP Food": [ww_mlp["scenarios"][k]["heldout_20"]["PPO"]["mean_poison_hits"] for k in sc_keys],
        "PPO Ray-Tx Food": [ww_tx["scenarios"][k]["heldout_20"]["PPO"]["mean_poison_hits"] for k in sc_keys],
    }
    w5 = 0.165
    offs5 = np.linspace(-2.0, 2.0, 5) * w5
    for off, (name, (f_vals, col)) in zip(offs5, food_groups.items()):
        p_vals = poison_vals[name]
        ax.bar(x + off, f_vals, width=w5, color=col, edgecolor="#1F2937", linewidth=0.5, label=name)
        ax.bar(x + off, [-pv for pv in p_vals], width=w5, color=col, alpha=0.40, hatch="//", edgecolor="#DC2626", linewidth=0.5)
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xticks(x, sc_labels)
    ax.set_ylim(-18, 31)
    ax.set_ylabel("Food Captures (+) / Poison Hits (-)")
    ax.set_title("B. Behavioral Trade-Off: Food Captures (top) vs Poison Hits (bottom, hatched)")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=7.8)

    # (1, 0) Wall-clock training time per 600-episode run
    ax = axes[1, 0]
    time_series = {
        "GA (SO(2) GPU)": [ww_mlp["scenarios"][k]["runs"]["GA"]["elapsed_sec"] for k in sc_keys],
        "DQN MLP": [ww_mlp["scenarios"][k]["runs"]["DQN"]["elapsed_sec"] for k in sc_keys],
        "DQN Ray-Tx": [ww_tx["scenarios"][k]["runs"]["DQN"]["elapsed_sec"] for k in sc_keys],
        "PPO MLP": [ww_mlp["scenarios"][k]["runs"]["PPO"]["elapsed_sec"] for k in sc_keys],
        "PPO Ray-Tx": [ww_tx["scenarios"][k]["runs"]["PPO"]["elapsed_sec"] for k in sc_keys],
    }
    time_cols = [palette["GA"], palette["DQN MLP"], palette["DQN Ray-Tx"], palette["PPO MLP"], palette["PPO Ray-Tx"]]
    for off, (name, vals), col in zip(offs5, time_series.items(), time_cols):
        bars = ax.bar(x + off, vals, width=w5, label=name, color=col, edgecolor="#1F2937", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 1.2, f"{v:.0f}s", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x, sc_labels)
    ax.set_ylim(0, 102)
    ax.set_ylabel("Training Time per 600-ep Run (seconds)")
    ax.set_title("C. Training Time on RTX 4080 (Total 5x3: MLP 491.9s vs Ray-Tx 717.1s)")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=7.8)

    # (1, 1) Parameter count efficiency
    ax = axes[1, 1]
    p_configs = ["base/single/more_poison\n(30 rays, speed=True)", "no_speed\n(30 rays, speed=False)", "few_sensors\n(10 rays, speed=True)"]
    p_scenarios = [SCENARIOS["base"], SCENARIOS["no_speed"], SCENARIOS["few_sensors"]]
    p_data = {
        "GA (SO(2))": [sum(p.numel() for p in WaterworldGAPolicy(s.n_sensors, s.speed_features).parameters()) for s in p_scenarios],
        "DQN MLP": [sum(p.numel() for p in DuelingQNet(s.obs_dim + 2, 9, arch="mlp", n_sensors=s.n_sensors, speed_features=s.speed_features).parameters()) for s in p_scenarios],
        "DQN Ray-Tx": [sum(p.numel() for p in DuelingQNet(s.obs_dim + 2, 9, arch="transformer", n_sensors=s.n_sensors, speed_features=s.speed_features).parameters()) for s in p_scenarios],
        "PPO MLP": [sum(p.numel() for p in ActorCriticPPO(s.obs_dim, arch="mlp", n_sensors=s.n_sensors, speed_features=s.speed_features).parameters()) for s in p_scenarios],
        "PPO Ray-Tx": [sum(p.numel() for p in ActorCriticPPO(s.obs_dim, arch="transformer", n_sensors=s.n_sensors, speed_features=s.speed_features).parameters()) for s in p_scenarios],
    }
    xp = np.arange(len(p_configs))
    for off, (name, vals), col in zip(offs5, p_data.items(), time_cols):
        bars = ax.bar(xp + off, vals, width=w5, label=name, color=col, edgecolor="#1F2937", linewidth=0.5)
        for b, v in zip(bars, vals):
            lbl = f"{v/1000:.1f}k" if v >= 1000 else str(v)
            ax.text(b.get_x() + b.get_width() / 2.0, v * 1.12, lbl, ha="center", va="bottom", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(8, 650_000)
    ax.set_xticks(xp, p_configs)
    ax.set_ylabel("Trainable Parameters (log scale)")
    ax.set_title("D. Parameter Efficiency: Ray-Transformer Is Ray-Count Agnostic (4.8x-6.8x Smaller)")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=7.8)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def build_waterworld_learning_curves(runs_root: Path, out_path: Path) -> None:
    ww_mlp = json.loads((runs_root / "waterworld" / "results.json").read_text(encoding="utf-8"))
    ww_tx = json.loads((runs_root / "waterworld_tx" / "results.json").read_text(encoding="utf-8"))

    sc_keys = ["base", "single", "no_speed", "few_sensors", "more_poison"]
    plt.rcParams.update({"font.size": 9.2, "axes.titlesize": 10.5, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8), dpi=165)
    fig.suptitle(
        "WaterWorld Learning Dynamics (600 GPU Episodes): Held-Out Eval Curves & Moving Average (30-ep)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    flat_axes = axes.ravel()
    for idx, sc_key in enumerate(sc_keys):
        ax = flat_axes[idx]
        sc = SCENARIOS[sc_key]
        sc_mlp = ww_mlp["scenarios"][sc_key]
        sc_tx = ww_tx["scenarios"][sc_key]

        zero_ret = sc_tx["baselines"]["Zero"]["mean_team_return"]
        ax.axhline(zero_ret, color="#6B7280", ls="--", lw=1.2, label=f"Zero ({zero_ret:+.1f})")

        # GA curve (eval_curve points + faint moving avg)
        ga_run = sc_mlp["runs"]["GA"]
        ga_ep_x = [e["episode"] for e in ga_run["episode_curve"]]
        ga_ep_y = _moving_avg([e["team_return"] for e in ga_run["episode_curve"]], 30)
        ga_ev_x = [e["episode"] for e in ga_run["eval_curve"]]
        ga_ev_y = [e["mean_team_return"] for e in ga_run["eval_curve"]]
        ax.plot(ga_ep_x, ga_ep_y, color="#2563EB", alpha=0.22, lw=1.0)
        ax.plot(ga_ev_x, ga_ev_y, color="#2563EB", marker="o", ms=3.5, lw=1.9, label="GA (SO(2))")

        # DQN MLP vs Ray-Tx
        for label, src, col, ls, mk in [
            ("DQN (MLP)", sc_mlp["runs"]["DQN"], "#F59E0B", "--", "s"),
            ("DQN (Ray-Tx)", sc_tx["runs"]["DQN"], "#B45309", "-", "^"),
            ("PPO (MLP)", sc_mlp["runs"]["PPO"], "#10B981", "--", "s"),
            ("PPO (Ray-Tx)", sc_tx["runs"]["PPO"], "#065F46", "-", "D"),
        ]:
            ep_x = [e["episode"] for e in src["episode_curve"]]
            ep_y = _moving_avg([e["team_return"] for e in src["episode_curve"]], 30)
            ev_x = [e["episode"] for e in src["eval_curve"]]
            ev_y = [e["mean_team_return"] for e in src["eval_curve"]]
            ax.plot(ep_x, ep_y, color=col, ls=ls, alpha=0.18, lw=1.0)
            ax.plot(ev_x, ev_y, color=col, ls=ls, marker=mk, ms=3.6, lw=1.8, label=label)

        ax.set_title(f"{sc_key}: {sc.title_ru}\n(A={sc.n_pursuers}, obs={sc.obs_dim})")
        ax.set_xlabel("Training Episodes")
        ax.set_ylabel("Team Return")
        ax.grid(True, alpha=0.28)
        if idx == 0:
            ax.legend(loc="lower right", frameon=True, fontsize=7.6)

    # Bottom-right summary panel
    ax = flat_axes[5]
    ax.axis("off")
    lines = ["Held-out 20-ep Team Return (MLP -> Ray-Tx):\n"]
    for k in sc_keys:
        hm = ww_mlp["scenarios"][k]["heldout_20"]
        ht = ww_tx["scenarios"][k]["heldout_20"]
        lines.append(
            f"{k:11s} | GA {hm['GA']['mean_team_return']:+6.1f} | "
            f"DQN {hm['DQN']['mean_team_return']:+6.1f} -> {ht['DQN']['mean_team_return']:+6.1f} | "
            f"PPO {hm['PPO']['mean_team_return']:+6.1f} -> {ht['PPO']['mean_team_return']:+6.1f}"
        )
    lines.append("\nLegend:")
    lines.append("• Solid lines + markers: Periodic 8-ep eval curves (Ray-Tx / GA)")
    lines.append("• Dashed lines + square markers: Periodic 8-ep eval curves (MLP)")
    lines.append("• Faded background curves: 30-episode moving average training returns")
    ax.text(
        0.02,
        0.92,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def build_acceleration_summary(runs_root: Path, out_path: Path) -> None:
    ww_mlp = json.loads((runs_root / "waterworld" / "results.json").read_text(encoding="utf-8"))
    ww_tx = json.loads((runs_root / "waterworld_tx" / "results.json").read_text(encoding="utf-8"))

    mlp_ga = sum(v["runs"]["GA"]["elapsed_sec"] for v in ww_mlp["scenarios"].values())
    mlp_dqn = sum(v["runs"]["DQN"]["elapsed_sec"] for v in ww_mlp["scenarios"].values())
    mlp_ppo = sum(v["runs"]["PPO"]["elapsed_sec"] for v in ww_mlp["scenarios"].values())

    tx_ga = sum(v["runs"]["GA"]["elapsed_sec"] for v in ww_tx["scenarios"].values())
    tx_dqn = sum(v["runs"]["DQN"]["elapsed_sec"] for v in ww_tx["scenarios"].values())
    tx_ppo = sum(v["runs"]["PPO"]["elapsed_sec"] for v in ww_tx["scenarios"].values())

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=170)
    fig.suptitle(
        "GPU & Vectorized Simulation Engineering Speedups (NVIDIA GeForce RTX 4080)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Environment stepping throughput (measured live in current workspace)
    envs = ["WaterWorld\nCPU pymunk\n(NumPy sensors)", "WaterWorld\nGPU CUDA Graph\n(B=1024)", "Pursuit\nCPU PettingZoo\n(sisl/pursuit-v5)", "Pursuit\nGPU Vectorized\n(B=288)"]
    sps = [1858.0, 1_293_996.0, 322.0, 49_469.0]
    cols = ["#9CA3AF", "#059669", "#9CA3AF", "#2563EB"]
    xb = np.arange(len(envs))
    bars = ax1.bar(xb, sps, width=0.58, color=cols, edgecolor="#1F2937", linewidth=0.7)
    ax1.set_yscale("log")
    ax1.set_ylim(100, 6_000_000)
    ax1.set_xticks(xb, envs, fontsize=8.8)
    ax1.set_ylabel("Environment Throughput (env-steps / s, log scale)")
    ax1.set_title("A. Raw Simulator Stepping Throughput (500-cycle episodes)")
    ax1.grid(True, axis="y", alpha=0.28)
    notes = ["269.1 ms/ep", "0.386 ms/ep\n696x speedup", "1,552.4 ms/ep", "10.11 ms/ep\n154x speedup"]
    for b, val, note in zip(bars, sps, notes):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            val * 1.18,
            f"{val:,.0f} step/s\n{note}",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    # Panel B: End-to-End 5x3 WaterWorld Matrix Time breakdown
    configs = [
        "Pre-GPU CPU Baseline\n(pymunk Multi-Proc, ref.)",
        "GPU CUDA Graph\nMLP + SO(2) GA",
        "GPU CUDA Graph\nRay-Transformer + SO(2) GA",
    ]
    # Note: 3250.8s is the documented pre-GPU CPU baseline from waterworld_marl/README.md
    ax2.bar([0], [3250.8], width=0.52, color="#9CA3AF", edgecolor="#1F2937", linewidth=0.7, label="CPU Baseline (~54.2 min)")
    ax2.bar([1], [mlp_ga], width=0.52, color="#2563EB", edgecolor="#1F2937", linewidth=0.7, label="GA (5 scenarios)")
    ax2.bar([1], [mlp_dqn], bottom=[mlp_ga], width=0.52, color="#F59E0B", edgecolor="#1F2937", linewidth=0.7, label="DQN (5 scenarios)")
    ax2.bar([1], [mlp_ppo], bottom=[mlp_ga + mlp_dqn], width=0.52, color="#10B981", edgecolor="#1F2937", linewidth=0.7, label="PPO (5 scenarios)")

    ax2.bar([2], [tx_ga], width=0.52, color="#2563EB", edgecolor="#1F2937", linewidth=0.7)
    ax2.bar([2], [tx_dqn], bottom=[tx_ga], width=0.52, color="#B45309", edgecolor="#1F2937", linewidth=0.7, label="DQN Ray-Tx")
    ax2.bar([2], [tx_ppo], bottom=[tx_ga + tx_dqn], width=0.52, color="#065F46", edgecolor="#1F2937", linewidth=0.7, label="PPO Ray-Tx")

    ax2.set_xticks([0, 1, 2], configs, fontsize=8.8)
    ax2.set_ylim(0, 3850)
    ax2.set_ylabel("Total Wall-Clock Time Across 15 Runs (seconds)")
    ax2.set_title("B. Complete 5x3 WaterWorld Training Suite Time")
    ax2.grid(True, axis="y", alpha=0.28)
    mlp_tot = mlp_ga + mlp_dqn + mlp_ppo
    tx_tot = tx_ga + tx_dqn + tx_ppo
    ax2.text(0, 3250.8 + 70, "3,250.8s\n(54.2 min)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.text(1, mlp_tot + 70, f"{mlp_tot:.1f}s (8.2 min)\n6.6x faster", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#065F46")
    ax2.text(2, tx_tot + 70, f"{tx_tot:.1f}s (12.0 min)\n4.5x faster", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#92400E")
    ax2.legend(loc="upper right", ncol=2, frameon=True, fontsize=7.8)

    plt.tight_layout(rect=[0, 0, 1, 0.935])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    runs_root = Path("runs")
    docs_root = Path("docs")
    build_pursuit_comparison(runs_root, docs_root / "pursuit_comparison.png")
    build_waterworld_comparison(runs_root, docs_root / "waterworld_mlp_vs_transformer.png")
    build_waterworld_learning_curves(runs_root, docs_root / "waterworld_learning_curves.png")
    build_acceleration_summary(runs_root, docs_root / "acceleration_summary.png")
    print("Generated all 4 research comparison charts in docs/")


if __name__ == "__main__":
    main()
