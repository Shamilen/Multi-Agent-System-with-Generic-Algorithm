"""Evaluation, baseline comparison, and GIF/chart visualization for trained Pursuit GA agents."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from pursuit_ga.config import EnvConfig, ModelConfig
from pursuit_ga.env_utils import get_evaders_remaining, make_pursuit_env
from pursuit_ga.evaluator import EpisodeMetrics, run_episode
from pursuit_ga.networks import PursuitPolicyNet, build_agent_phase_features


@dataclass
class BenchmarkSummary:
    """Aggregate evaluation statistics across multiple unseen episodes."""

    name: str
    episodes: int
    mean_agent_return: float
    std_agent_return: float
    mean_team_return: float
    mean_evaders_caught: float
    std_evaders_caught: float
    catch_rate_pct: float
    mean_cycles: float
    full_clear_rate_pct: float


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[PursuitPolicyNet, EnvConfig, ModelConfig, dict[str, Any]]:
    """Load saved checkpoint (.pt) and reconstruct policy network and configs."""
    dev = device or torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    env_cfg = EnvConfig(**ckpt["env_config"])
    model_raw = dict(ckpt["model_config"])
    if "conv_channels" in model_raw:
        model_raw["conv_channels"] = tuple(model_raw["conv_channels"])
    model_cfg = ModelConfig(**model_raw)

    policy = PursuitPolicyNet(
        model_cfg=model_cfg,
        obs_range=env_cfg.obs_range,
        n_actions=5,
    ).to(dev)
    state_dict = dict(ckpt["state_dict"])
    # Backfill deterministic coordinate buffer for legacy pre-transformer CNN checkpoints.
    if "rel_coords" not in state_dict:
        state_dict["rel_coords"] = policy.rel_coords.clone()
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, env_cfg, model_cfg, ckpt


def _summarize_metrics(name: str, items: list[EpisodeMetrics]) -> BenchmarkSummary:
    agent_rets = np.array([m.mean_agent_return for m in items], dtype=np.float64)
    team_rets = np.array([m.team_return for m in items], dtype=np.float64)
    caught = np.array([m.evaders_caught for m in items], dtype=np.float64)
    totals = np.array([m.evaders_total for m in items], dtype=np.float64)
    cycles = np.array([m.cycles for m in items], dtype=np.float64)
    clears = np.array([1.0 if m.cleared_all else 0.0 for m in items], dtype=np.float64)
    return BenchmarkSummary(
        name=name,
        episodes=len(items),
        mean_agent_return=float(agent_rets.mean()),
        std_agent_return=float(agent_rets.std()),
        mean_team_return=float(team_rets.mean()),
        mean_evaders_caught=float(caught.mean()),
        std_evaders_caught=float(caught.std()),
        catch_rate_pct=float(100.0 * caught.sum() / max(totals.sum(), 1.0)),
        mean_cycles=float(cycles.mean()),
        full_clear_rate_pct=float(100.0 * clears.mean()),
    )


def evaluate_policy_suite(
    policy: PursuitPolicyNet,
    env_cfg: EnvConfig,
    seeds: list[int],
    stochastic: bool | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[BenchmarkSummary, list[EpisodeMetrics]]:
    """Evaluate a trained PursuitPolicyNet across a list of episode seeds."""
    dev = device or next(policy.parameters()).device
    env = make_pursuit_env(env_cfg)
    phase_t = torch.from_numpy(build_agent_phase_features(env_cfg.n_pursuers)).to(dev)
    results: list[EpisodeMetrics] = []
    try:
        for idx, s in enumerate(seeds, start=1):
            m, _ = run_episode(
                env=env,
                policy=policy,
                seed=s,
                n_evaders=env_cfg.n_evaders,
                agent_phase_t=phase_t,
                stochastic=stochastic,
                device=dev,
            )
            results.append(m)
            if verbose and (idx % 5 == 0 or idx == len(seeds)):
                print(f"    ... {idx}/{len(seeds)} episodes done", flush=True)
    finally:
        env.close()
    label = "Trained GA Policy"
    return _summarize_metrics(label, results), results


def evaluate_random_baseline(
    env_cfg: EnvConfig,
    seeds: list[int],
    verbose: bool = False,
) -> tuple[BenchmarkSummary, list[EpisodeMetrics]]:
    """Evaluate a uniform random baseline on the exact same seeds."""
    env = make_pursuit_env(env_cfg)
    results: list[EpisodeMetrics] = []
    try:
        for idx, s in enumerate(seeds, start=1):
            rng = np.random.default_rng(s + 99991)
            obs, _ = env.reset(seed=int(s))
            team_ret = 0.0
            cycles = 0
            while env.agents:
                acts = {ag: int(rng.integers(0, 5)) for ag in env.agents}
                obs, rew, term, trunc, _ = env.step(acts)
                team_ret += float(sum(rew.values()))
                cycles += 1
                if not env.agents or any(term.values()) or any(trunc.values()):
                    break
            rem = get_evaders_remaining(env)
            caught = env_cfg.n_evaders - rem
            n_ag = max(env_cfg.n_pursuers, 1)
            results.append(
                EpisodeMetrics(
                    fitness=team_ret / n_ag,
                    mean_agent_return=team_ret / n_ag,
                    team_return=team_ret,
                    evaders_caught=caught,
                    evaders_total=env_cfg.n_evaders,
                    cycles=cycles,
                    cleared_all=(rem == 0),
                )
            )
            if verbose and (idx % 5 == 0 or idx == len(seeds)):
                print(f"    ... {idx}/{len(seeds)} episodes done", flush=True)
    finally:
        env.close()
    return _summarize_metrics("Random Baseline", results), results


def _greedy_chase_action(obs: np.ndarray, rng: np.random.Generator) -> int:
    """Simple non-cooperative baseline: step toward nearest visible evader, else random.

    In Pursuit raw obs (H, W, 3), axis 0 indexes dy (y-axis) and axis 1 indexes dx (x-axis)
    because pursuit.py calls np.swapaxes(o, 2, 0).
    Action deltas from discrete_agent.py:
      0: dx=-1, dy=0
      1: dx=+1, dy=0
      2: dx=0,  dy=+1
      3: dx=0,  dy=-1
      4: stay
    """
    evader_layer = obs[:, :, 2]
    wall_layer = obs[:, :, 0]
    coords = np.argwhere(evader_layer > 0)
    c = obs.shape[0] // 2
    if len(coords) == 0:
        return int(rng.integers(0, 4))

    dists = np.abs(coords[:, 0] - c) + np.abs(coords[:, 1] - c)
    target_dy, target_dx = coords[int(np.argmin(dists))] - c
    if target_dy == 0 and target_dx == 0:
        return 4

    candidates: list[int] = []
    if abs(target_dx) >= abs(target_dy):
        if target_dx < 0 and wall_layer[c, c - 1] == 0:
            candidates.append(0)
        elif target_dx > 0 and wall_layer[c, c + 1] == 0:
            candidates.append(1)
        if target_dy > 0 and wall_layer[c + 1, c] == 0:
            candidates.append(2)
        elif target_dy < 0 and wall_layer[c - 1, c] == 0:
            candidates.append(3)
    else:
        if target_dy > 0 and wall_layer[c + 1, c] == 0:
            candidates.append(2)
        elif target_dy < 0 and wall_layer[c - 1, c] == 0:
            candidates.append(3)
        if target_dx < 0 and wall_layer[c, c - 1] == 0:
            candidates.append(0)
        elif target_dx > 0 and wall_layer[c, c + 1] == 0:
            candidates.append(1)

    if candidates:
        return candidates[0]
    return int(rng.integers(0, 4))


def evaluate_greedy_baseline(
    env_cfg: EnvConfig,
    seeds: list[int],
    verbose: bool = False,
) -> tuple[BenchmarkSummary, list[EpisodeMetrics]]:
    """Evaluate a non-cooperative greedy nearest-evader heuristic baseline."""
    env = make_pursuit_env(env_cfg)
    results: list[EpisodeMetrics] = []
    try:
        for idx, s in enumerate(seeds, start=1):
            rng = np.random.default_rng(s + 77773)
            obs, _ = env.reset(seed=int(s))
            team_ret = 0.0
            cycles = 0
            while env.agents:
                acts = {ag: _greedy_chase_action(obs[ag], rng) for ag in env.agents}
                obs, rew, term, trunc, _ = env.step(acts)
                team_ret += float(sum(rew.values()))
                cycles += 1
                if not env.agents or any(term.values()) or any(trunc.values()):
                    break
            rem = get_evaders_remaining(env)
            caught = env_cfg.n_evaders - rem
            n_ag = max(env_cfg.n_pursuers, 1)
            results.append(
                EpisodeMetrics(
                    fitness=team_ret / n_ag,
                    mean_agent_return=team_ret / n_ag,
                    team_return=team_ret,
                    evaders_caught=caught,
                    evaders_total=env_cfg.n_evaders,
                    cycles=cycles,
                    cleared_all=(rem == 0),
                )
            )
            if verbose and (idx % 5 == 0 or idx == len(seeds)):
                print(f"    ... {idx}/{len(seeds)} episodes done", flush=True)
    finally:
        env.close()
    return _summarize_metrics("Greedy Heuristic Baseline", results), results


def render_episode_gif(
    policy: PursuitPolicyNet,
    env_cfg: EnvConfig,
    seed: int,
    gif_path: str | Path,
    fps: int = 8,
    max_frames: int = 250,
    stochastic: bool | None = None,
    device: torch.device | None = None,
) -> EpisodeMetrics:
    """Record an animated GIF of a single Pursuit episode."""
    dev = device or next(policy.parameters()).device
    env = make_pursuit_env(env_cfg, render_mode="rgb_array")
    phase_t = torch.from_numpy(build_agent_phase_features(env_cfg.n_pursuers)).to(dev)
    try:
        metrics, frames = run_episode(
            env=env,
            policy=policy,
            seed=seed,
            n_evaders=env_cfg.n_evaders,
            agent_phase_t=phase_t,
            stochastic=stochastic,
            device=dev,
            capture_frames=True,
        )
    finally:
        env.close()

    if frames:
        if len(frames) > max_frames:
            stride = max(1, len(frames) // max_frames)
            frames = frames[::stride]
        pil_frames = [Image.fromarray(f) for f in frames]
        out_path = Path(gif_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / max(fps, 1)),
            loop=0,
        )
    return metrics


def plot_training_curves(
    history: list[dict[str, Any]],
    benchmarks: list[BenchmarkSummary],
    out_path: str | Path,
) -> None:
    """Save a multi-panel matplotlib figure showing GA learning curves and benchmarks."""
    gens = [h["generation"] for h in history]
    best_ret = [h["best_return"] for h in history]
    mean_ret = [h["mean_return"] for h in history]
    val_fit = [h["elite_validated_fitness"] for h in history]
    best_caught = [h["best_evaders_caught"] for h in history]
    mean_caught = [h["mean_evaders_caught"] for h in history]
    mean_cycles = [h["mean_cycles"] for h in history]
    sigmas = [h["mutation_sigma"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
    fig.suptitle(
        "Deep MARL Neuroevolution (Genetic Algorithm) on PettingZoo sisl/pursuit-v5",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(gens, mean_ret, label="Population Mean Return", color="#4c72b0", linewidth=1.8)
    ax.plot(gens, best_ret, label="Generation Best Return", color="#dd8452", linewidth=1.8)
    ax.plot(gens, val_fit, label="Elite Validated Fitness", color="#55a868", linewidth=2.0, linestyle="--")
    ax.set_title("Mean Agent Return per Episode")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Agent Return")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(gens, mean_caught, label="Population Mean Caught", color="#4c72b0", linewidth=1.8)
    ax.plot(gens, best_caught, label="Generation Best Caught", color="#c44e52", linewidth=2.0)
    ax.axhline(30, color="#333333", linestyle=":", linewidth=1.2, label="Max Evaders (30)")
    ax.set_title("Evaders Captured per Episode")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Evaders Caught")
    ax.set_ylim(0, 32)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(gens, mean_cycles, label="Mean Episode Cycles", color="#8172b3", linewidth=1.8)
    l2 = ax2.plot(gens, sigmas, label="Mutation Sigma", color="#ccb974", linewidth=1.8, linestyle="--")
    ax.set_title("Episode Length & Adaptive Mutation Sigma")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cycles (max 500)", color="#8172b3")
    ax2.set_ylabel("Mutation Sigma", color="#8c7a3b")
    ax.grid(True, alpha=0.3)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc="upper right")

    ax = axes[1, 1]
    if benchmarks:
        names = [b.name for b in benchmarks]
        caught_vals = [b.mean_evaders_caught for b in benchmarks]
        caught_errs = [b.std_evaders_caught for b in benchmarks]
        colors = ["#999999", "#dd8452", "#55a868"][: len(names)]
        bars = ax.bar(names, caught_vals, yerr=caught_errs, capsize=5, color=colors, alpha=0.9)
        ax.set_ylim(0, 36)
        ax.set_title("Held-Out Evaluation: Evaders Caught (/ 30)")
        ax.set_ylabel("Mean Evaders Caught")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, err, b in zip(bars, caught_errs, benchmarks):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + err + 0.6,
                f"{b.mean_evaders_caught:.1f}/30\n({b.mean_agent_return:+.1f})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Pursuit GA model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--seed-start", type=int, default=50000, help="Starting evaluation seed")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions instead of sampling")
    parser.add_argument("--save-gif", type=str, default=None, help="Optional output path for episode GIF")
    parser.add_argument("--gif-seed", type=int, default=None, help="Seed for rendered GIF episode")
    parser.add_argument("--save-json", type=str, default=None, help="Optional JSON summary output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, env_cfg, _, _ = load_checkpoint(args.checkpoint, device=device)
    seeds = [args.seed_start + i for i in range(args.episodes)]

    stochastic = False if args.deterministic else None
    trained_summary, _ = evaluate_policy_suite(
        policy, env_cfg, seeds, stochastic=stochastic, device=device
    )
    random_summary, _ = evaluate_random_baseline(env_cfg, seeds)
    greedy_summary, _ = evaluate_greedy_baseline(env_cfg, seeds)

    for s in (random_summary, greedy_summary, trained_summary):
        print(
            f"[{s.name}] return={s.mean_agent_return:+.2f}±{s.std_agent_return:.2f} | "
            f"caught={s.mean_evaders_caught:.2f}/30 ({s.catch_rate_pct:.1f}%) | "
            f"cycles={s.mean_cycles:.1f} | full_clear={s.full_clear_rate_pct:.1f}%"
        )

    if args.save_gif:
        g_seed = args.seed_start if args.gif_seed is None else args.gif_seed
        m = render_episode_gif(
            policy, env_cfg, seed=g_seed, gif_path=args.save_gif, stochastic=stochastic, device=device
        )
        print(f"Saved GIF to {args.save_gif}: caught={m.evaders_caught}/30, cycles={m.cycles}, return={m.mean_agent_return:+.2f}")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(s) for s in (random_summary, greedy_summary, trained_summary)],
                f,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
