"""Run the 5x3 WaterWorld MARL benchmark (GA, Double DQN, MAPPO) and generate plots/GIF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from waterworld_marl.scenarios import (
    SCENARIOS,
    SubprocWaterworldVecEnv,
    WaterworldScenario,
    evaluate_policy_parallel,
    rollout_single_episode,
)


def _moving_average(values: list[float], window: int = 25) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    w = min(window, len(arr))
    if w <= 1:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    # Variable-width trailing window at early indices so no artificial padding bias occurs.
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - w + 1)
        out[i] = (cumsum[i + 1] - cumsum[start]) / float(i - start + 1)
    return out


def evaluate_baselines(
    scenario: WaterworldScenario,
    eval_seeds: list[int],
    num_envs: int = 10,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate Random Uniform and Zero-Action baselines on fixed held-out seeds."""
    rng = np.random.default_rng(seed + 99)

    def zero_fn(obs_batch: np.ndarray, ph: np.ndarray) -> np.ndarray:
        return np.zeros((*obs_batch.shape[:-1], 2), dtype=np.float32)

    def rand_fn(obs_batch: np.ndarray, ph: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(*obs_batch.shape[:-1], 2)).astype(np.float32)

    with SubprocWaterworldVecEnv(
        scenario,
        num_envs=min(num_envs, len(eval_seeds), os.cpu_count() or 16),
        base_seed=940_000,
    ) as eval_env:
        zero_res = evaluate_policy_parallel(eval_env, zero_fn, eval_seeds)
        rand_res = evaluate_policy_parallel(eval_env, rand_fn, eval_seeds)
    return {"Zero": zero_res, "Random": rand_res}


def build_action_fn_from_result(
    scenario: WaterworldScenario,
    algo: str,
    ckpt_payload: dict[str, Any],
    device: str = "auto",
) -> Any:
    if algo == "DQN":
        from waterworld_marl.algos.dqn import build_dqn_action_fn

        return build_dqn_action_fn(
            scenario,
            ckpt_payload["state_dict"],
            obs_rms_state=ckpt_payload.get("obs_rms_state"),
            device=device,
        )
    if algo == "PPO":
        from waterworld_marl.algos.ppo import build_ppo_action_fn

        return build_ppo_action_fn(
            scenario, ckpt_payload["state_dict"], ckpt_payload["obs_rms_state"], device=device
        )
    if algo == "GA":
        from waterworld_marl.algos.ga import build_ga_action_fn

        return build_ga_action_fn(scenario, ckpt_payload["state_dict"], device=device)
    raise ValueError(f"Unknown algo: {algo}")


def plot_curves_grid(
    results: dict[str, Any],
    metric_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot 5-scenario learning curves (GA, DQN, PPO) with Random/Zero references."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_keys = list(SCENARIOS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), dpi=150)
    axes_flat = axes.flatten()

    arch_cfg = results.get("config", {}).get("arch", "transformer")
    colors = {"GA": "#1f77b4", "DQN": "#ff7f0e", "PPO": "#2ca02c"}
    labels = (
        {
            "GA": "GA (SO(2) Equiv. + CRN)",
            "DQN": "Double DQN (Ray-Transformer)",
            "PPO": "MAPPO (CTDE Ray-Transformer)",
        }
        if arch_cfg == "transformer"
        else {
            "GA": "GA (SO(2) Equiv. + CRN)",
            "DQN": "Double DQN (Dueling)",
            "PPO": "MAPPO (CTDE)",
        }
    )

    for idx, sc_key in enumerate(scenario_keys):
        ax = axes_flat[idx]
        sc_data = results["scenarios"].get(sc_key, {})
        sc_obj = SCENARIOS[sc_key]
        eval_metric = "mean_team_return" if metric_key == "team_return" else "mean_agent_return"
        std_metric = "std_team_return" if metric_key == "team_return" else "std_agent_return"

        # Reference lines
        zero_val = sc_data.get("baselines", {}).get("Zero", {}).get(eval_metric)
        rand_val = sc_data.get("baselines", {}).get("Random", {}).get(eval_metric)
        if zero_val is not None:
            ax.axhline(zero_val, color="#7f7f7f", linestyle="--", linewidth=1.2, label="Zero Action")
        if rand_val is not None and rand_val > -120:
            ax.axhline(rand_val, color="#d62728", linestyle=":", linewidth=1.2, label="Random Action")

        for algo in ("GA", "DQN", "PPO"):
            run = sc_data.get("runs", {}).get(algo)
            if not run:
                continue
            ep_curve = run.get("episode_curve", [])
            if ep_curve:
                xs_ep = [p["episode"] for p in ep_curve]
                ys_ep = _moving_average([p[metric_key] for p in ep_curve], window=30)
                ax.plot(xs_ep, ys_ep, color=colors[algo], alpha=0.28, linewidth=1.0)

            ev_curve = run.get("eval_curve", [])
            if ev_curve:
                xs_ev = np.asarray([p["episode"] for p in ev_curve], dtype=np.float64)
                ys_ev = np.asarray([p[eval_metric] for p in ev_curve], dtype=np.float64)
                stds = np.asarray([p.get(std_metric, 0.0) for p in ev_curve], dtype=np.float64)
                ax.plot(
                    xs_ev,
                    ys_ev,
                    color=colors[algo],
                    marker="o",
                    markersize=3.5,
                    linewidth=2.0,
                    label=labels[algo],
                )
                ax.fill_between(
                    xs_ev, ys_ev - 0.5 * stds, ys_ev + 0.5 * stds, color=colors[algo], alpha=0.10
                )

        ax.set_title(f"{sc_obj.title_ru}\n({sc_key}: A={sc_obj.n_pursuers}, obs={sc_obj.obs_dim})", fontsize=10)
        ax.set_xlabel("Training episodes")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    # 6th subplot: Legend & Summary table
    ax_sum = axes_flat[5]
    ax_sum.axis("off")
    eval_metric = "mean_team_return" if metric_key == "team_return" else "mean_agent_return"
    held_n = int(results.get("config", {}).get("heldout_episodes", 20))
    ev_n = int(results.get("config", {}).get("eval_episodes", 8))
    lines = [f"Held-out {held_n}-ep {eval_metric} summary:\n"]
    for sc_key in scenario_keys:
        sc_runs = results["scenarios"].get(sc_key, {}).get("heldout_20", {})
        zero_v = results["scenarios"].get(sc_key, {}).get("baselines", {}).get("Zero", {}).get(eval_metric, 0.0)
        parts = [f"{a}: {sc_runs.get(a, {}).get(eval_metric, float('nan')):+6.2f}" for a in ("GA", "DQN", "PPO")]
        lines.append(f"{sc_key:11s} | Zero={zero_v:+5.2f} | " + " | ".join(parts))
    ax_sum.text(
        0.02,
        0.92,
        "\n".join(lines),
        transform=ax_sum.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.5,
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )
    ax_sum.text(
        0.02,
        0.22,
        f"Solid lines + markers: periodic held-out eval ({ev_n} seeds)\n"
        "Faded lines: 30-ep moving average of real training returns\n"
        "Shaded band: ±0.5 std across eval seeds",
        transform=ax_sum.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#444444",
    )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def plot_bar_comparison(results: dict[str, Any], out_path: Path) -> None:
    """Bar chart comparing Zero, GA, Double DQN, MAPPO on held-out 20 episodes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_keys = list(SCENARIOS.keys())
    algos = ["Zero", "GA", "DQN", "PPO"]
    colors = {"Zero": "#8c8c8c", "GA": "#1f77b4", "DQN": "#ff7f0e", "PPO": "#2ca02c"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.5), dpi=150)
    x = np.arange(len(scenario_keys))
    width = 0.20

    for j, algo in enumerate(algos):
        vals_team = []
        vals_food = []
        for sc_key in scenario_keys:
            sc_data = results["scenarios"].get(sc_key, {})
            if algo == "Zero":
                m = sc_data.get("baselines", {}).get("Zero", {})
            else:
                m = sc_data.get("heldout_20", {}).get(algo, {})
            vals_team.append(m.get("mean_team_return", 0.0))
            vals_food.append(m.get("mean_food_events", 0.0))

        offset = (j - 1.5) * width
        b1 = ax1.bar(x + offset, vals_team, width=width, label=algo, color=colors[algo])
        ax2.bar(x + offset, vals_food, width=width, label=algo, color=colors[algo])
        for rect in b1:
            h = rect.get_height()
            ax1.annotate(
                f"{h:+.0f}",
                xy=(rect.get_x() + rect.get_width() / 2.0, h),
                xytext=(0, 2 if h >= 0 else -10),
                textcoords="offset points",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=7.5,
            )

    short_labels = {
        "base": "Базовая",
        "single": "Упрощённый\nвариант",
        "no_speed": "Отключение\nspeed_features",
        "few_sensors": "Уменьшение\nкол-ва сенсоров",
        "more_poison": "Увеличение яда и\nуменьшение еды",
    }
    tick_labels = [short_labels.get(k, SCENARIOS[k].title_ru) for k in scenario_keys]

    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels, fontsize=9)
    ax1.set_ylabel("Mean Team Return (20 held-out eps)")
    ax1.set_title("WaterWorld: Held-out Team Return Comparison (20 episodes)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()

    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels, fontsize=9)
    ax2.set_ylabel("Mean Food Captures / Episode")
    ax2.set_title("WaterWorld: Food Captures per Episode (20 held-out episodes)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_best_gif(
    scenario: WaterworldScenario,
    action_fn: Any,
    out_gif: Path,
    seed: int = 950_000,
    stride: int = 2,
) -> dict[str, float]:
    from PIL import Image

    stats, frames = rollout_single_episode(scenario, action_fn, seed=seed, capture_frames=True)
    if frames:
        imgs = [Image.fromarray(frames[i]) for i in range(0, len(frames), stride)]
        imgs[0].save(
            out_gif,
            save_all=True,
            append_images=imgs[1:],
            duration=40,
            loop=0,
        )
    return stats


def _serialize_results(results: dict[str, Any]) -> dict[str, Any]:
    return results


def main() -> None:
    import torch

    from waterworld_marl.algos.dqn import train_double_dqn
    from waterworld_marl.algos.ga import train_ga
    from waterworld_marl.algos.ppo import train_ppo

    parser = argparse.ArgumentParser(description="Run 5x3 PettingZoo WaterWorld MARL Benchmark")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--heldout-episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=24 if torch.cuda.is_available() else 12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs/waterworld")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()))
    parser.add_argument("--algos", nargs="+", default=["GA", "DQN", "PPO"])
    parser.add_argument("--arch", type=str, default="transformer", choices=["transformer", "mlp"])
    parser.add_argument(
        "--coop",
        action="store_true",
        help="Enable cooperative MARL upgrades (QMIX + Cross-Agent Attention Critic + GRU + n_coop curriculum)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    if args.resume and results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        results = {
            "config": vars(args),
            "scenarios": {},
        }

    heldout_seeds = [950_000 + i for i in range(args.heldout_episodes)]
    t_all = time.perf_counter()

    for sc_key in args.scenarios:
        scenario = SCENARIOS[sc_key]
        print(f"\n==================== Scenario: {sc_key} ({scenario.title_ru}) ====================", flush=True)
        sc_entry = results["scenarios"].setdefault(
            sc_key, {"runs": {}, "heldout_20": {}, "baselines": {}}
        )

        if not sc_entry["baselines"]:
            baselines = evaluate_baselines(scenario, heldout_seeds, num_envs=args.num_envs, seed=args.seed)
            sc_entry["baselines"] = baselines
            print(
                f"  [Baselines] Random team={baselines['Random']['mean_team_return']:+6.2f} "
                f"(agent={baselines['Random']['mean_agent_return']:+5.2f}) | "
                f"Zero team={baselines['Zero']['mean_team_return']:+6.2f} "
                f"(agent={baselines['Zero']['mean_agent_return']:+5.2f})",
                flush=True,
            )

        for algo in args.algos:
            ckpt_file = ckpt_dir / f"{sc_key}_{algo.lower()}.pt"
            if args.resume and algo in sc_entry["runs"] and algo in sc_entry["heldout_20"] and ckpt_file.exists():
                print(f"  [Resume] Skipping {sc_key}/{algo} (already completed).", flush=True)
                continue

            if algo == "GA":
                res = train_ga(
                    scenario=scenario,
                    total_episodes=args.episodes,
                    pop_size=12,
                    num_workers=args.num_envs,
                    eval_interval_episodes=args.eval_interval,
                    eval_episodes=args.eval_episodes,
                    seed=args.seed,
                    device=args.device,
                    verbose=True,
                )
            elif algo == "DQN":
                res = train_double_dqn(
                    scenario=scenario,
                    total_episodes=args.episodes,
                    num_envs=args.num_envs,
                    eval_interval_episodes=args.eval_interval,
                    eval_episodes=args.eval_episodes,
                    seed=args.seed,
                    device=args.device,
                    verbose=True,
                    arch=args.arch,
                    coop_mode=args.coop,
                )
            else:
                res = train_ppo(
                    scenario=scenario,
                    total_episodes=args.episodes,
                    num_envs=args.num_envs,
                    eval_interval_episodes=args.eval_interval,
                    eval_episodes=args.eval_episodes,
                    seed=args.seed,
                    device=args.device,
                    verbose=True,
                    arch=args.arch,
                    coop_mode=args.coop,
                )

            ckpt_payload = {"state_dict": res["state_dict"]}
            if "obs_rms_state" in res:
                ckpt_payload["obs_rms_state"] = res["obs_rms_state"]
            torch.save(ckpt_payload, ckpt_file)

            # Evaluate best checkpoint on 20 fixed held-out seeds
            act_fn = build_action_fn_from_result(scenario, algo, ckpt_payload, device=args.device)
            with SubprocWaterworldVecEnv(
                scenario,
                num_envs=min(args.num_envs, len(heldout_seeds), os.cpu_count() or 16),
                base_seed=960_000,
            ) as h_env:
                h_eval = evaluate_policy_parallel(h_env, act_fn, heldout_seeds)

            sc_entry["heldout_20"][algo] = h_eval
            sc_entry["runs"][algo] = {
                "algo": res["algo"],
                "scenario": res["scenario"],
                "elapsed_sec": res["elapsed_sec"],
                "total_episodes": res["total_episodes"],
                "train_updates": res["train_updates"],
                "best_eval": res["best_eval"],
                "episode_curve": res["episode_curve"],
                "eval_curve": res["eval_curve"],
            }
            print(
                f"  ==> {sc_key}/{algo} heldout-20: team={h_eval['mean_team_return']:+6.2f}±{h_eval['std_team_return']:.2f} "
                f"| agent={h_eval['mean_agent_return']:+5.2f} "
                f"| food={h_eval['mean_food_events']:.1f} | poison={h_eval['mean_poison_hits']:.1f} "
                f"({res['elapsed_sec']:.1f}s)",
                flush=True,
            )
            results_path.write_text(json.dumps(_serialize_results(results), indent=2), encoding="utf-8")

    results["total_elapsed_sec"] = float(time.perf_counter() - t_all)
    results_path.write_text(json.dumps(_serialize_results(results), indent=2), encoding="utf-8")

    # Plots
    plot_curves_grid(
        results,
        metric_key="mean_agent_return",
        ylabel="Mean Agent Return",
        title="PettingZoo WaterWorld: Mean Agent Return vs Training Episodes",
        out_path=out_dir / "figure4_reproduction_agent.png",
    )
    plot_curves_grid(
        results,
        metric_key="team_return",
        ylabel="Team Return (Sum over agents)",
        title="PettingZoo WaterWorld: Team Return vs Training Episodes",
        out_path=out_dir / "figure4_reproduction_team.png",
    )
    plot_bar_comparison(results, out_dir / "bar_comparison.png")

    # Render GIF of best-performing policy on base scenario
    if "base" in results["scenarios"]:
        held_base = results["scenarios"]["base"].get("heldout_20", {})
        if held_base:
            best_algo = max(held_base.keys(), key=lambda a: held_base[a]["mean_team_return"])
            ckpt = torch.load(ckpt_dir / f"base_{best_algo.lower()}.pt", map_location="cpu", weights_only=False)
            fn = build_action_fn_from_result(SCENARIOS["base"], best_algo, ckpt, device=args.device)
            gif_stats = render_best_gif(SCENARIOS["base"], fn, out_dir / f"demo_base_{best_algo.lower()}.gif")
            print(f"\nSaved best base demo GIF ({best_algo}): {gif_stats}", flush=True)


if __name__ == "__main__":
    main()
