"""Training entrypoint for Deep MARL Genetic Algorithm on PettingZoo SISL Pursuit."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pursuit_ga.config import EnvConfig, GAConfig, ModelConfig, TrainConfig
from pursuit_ga.evaluate import (
    evaluate_greedy_baseline,
    evaluate_policy_suite,
    evaluate_random_baseline,
    plot_training_curves,
    render_episode_gif,
)
from pursuit_ga.evaluator import ParallelGenomeEvaluator
from pursuit_ga.genetic_algorithm import GenerationStats, GeneticAlgorithm
from pursuit_ga.gpu_sim import GPUPopulationEvaluator
from pursuit_ga.networks import PursuitPolicyNet


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (--device cuda), but no CUDA GPU is available.")
    return dev


def _save_checkpoint(
    path: str | Path,
    policy: PursuitPolicyNet,
    genome: np.ndarray,
    cfg: TrainConfig,
    generation: int,
    best_val_fitness: float,
) -> None:
    policy.set_genome(genome)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "genome": genome,
            "generation": generation,
            "best_val_fitness": best_val_fitness,
            "env_config": asdict(cfg.env),
            "model_config": asdict(cfg.model),
            "ga_config": asdict(cfg.ga),
        },
        path,
    )


def train(cfg: TrainConfig) -> dict[str, Any]:
    """Run evolutionary MARL training and return summary metrics."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out_dir / "config.json")

    device = _resolve_device(cfg.device)

    def policy_factory() -> PursuitPolicyNet:
        return PursuitPolicyNet(
            model_cfg=cfg.model,
            obs_range=cfg.env.obs_range,
            n_actions=5,
        )

    ga = GeneticAlgorithm(policy_factory=policy_factory, ga_cfg=cfg.ga, seed=cfg.seed)
    template_policy = policy_factory().to(device)
    template_policy.eval()

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(
        f"[Init] env={cfg.env.env_id} | arch={cfg.model.arch} | "
        f"params={ga.num_params:,} | pop={cfg.ga.population_size} | "
        f"gens={cfg.ga.generations} | device={device} ({gpu_name})",
        flush=True,
    )

    val_seeds = [cfg.seed * 10000 + 1000 + j for j in range(cfg.ga.elite_eval_episodes)]
    test_seeds = [cfg.seed * 10000 + 50000 + j for j in range(cfg.eval_episodes)]

    best_val_fitness = -float("inf")
    best_val_genome: np.ndarray | None = None
    history: list[dict[str, Any]] = []

    evaluator: GPUPopulationEvaluator | ParallelGenomeEvaluator
    if device.type == "cuda":
        evaluator = GPUPopulationEvaluator(
            env_cfg=cfg.env,
            model_cfg=cfg.model,
            catch_bonus_weight=cfg.ga.catch_bonus_weight,
            device=device,
        )
    else:
        evaluator = ParallelGenomeEvaluator(
            env_cfg=cfg.env,
            model_cfg=cfg.model,
            catch_bonus_weight=cfg.ga.catch_bonus_weight,
            num_workers=cfg.num_workers,
        )

    with evaluator:
        for gen in range(cfg.ga.generations):
            t0 = time.perf_counter()
            train_seeds = [
                cfg.seed * 1000 + gen * 97 + k
                for k in range(cfg.ga.episodes_per_genome)
            ]
            eval_res = evaluator.evaluate_population(ga.population, train_seeds)

            # Re-evaluate top candidates on held-out validation seeds
            raw_order = np.argsort(eval_res.fitnesses)[::-1]
            top_k = min(max(2, cfg.ga.elite_count), len(ga.population))
            top_indices = raw_order[:top_k]
            top_genomes = ga.population[top_indices]
            val_res = evaluator.evaluate_population(top_genomes, val_seeds)

            val_best_local = int(np.argmax(val_res.fitnesses))
            gen_champion_idx = int(top_indices[val_best_local])
            elite_val_fit = float(val_res.fitnesses[val_best_local])
            elite_val_caught = float(val_res.evaders_caught[val_best_local])

            if elite_val_fit > best_val_fitness:
                best_val_fitness = elite_val_fit
                best_val_genome = ga.population[gen_champion_idx].copy()
                _save_checkpoint(
                    out_dir / "best_model.pt",
                    template_policy,
                    best_val_genome,
                    cfg,
                    generation=gen,
                    best_val_fitness=best_val_fitness,
                )

            combined_top = 0.4 * eval_res.fitnesses[top_indices] + 0.6 * val_res.fitnesses
            reranked_top = top_indices[np.argsort(combined_top)[::-1]]
            elite_override_order = np.concatenate([reranked_top, raw_order[top_k:]])

            best_train_idx = int(raw_order[0])
            elapsed = time.perf_counter() - t0

            stats = GenerationStats(
                generation=gen,
                best_fitness=float(eval_res.fitnesses[best_train_idx]),
                elite_validated_fitness=elite_val_fit,
                mean_fitness=float(np.mean(eval_res.fitnesses)),
                median_fitness=float(np.median(eval_res.fitnesses)),
                std_fitness=float(np.std(eval_res.fitnesses)),
                best_return=float(eval_res.mean_agent_returns[best_train_idx]),
                mean_return=float(np.mean(eval_res.mean_agent_returns)),
                best_evaders_caught=float(eval_res.evaders_caught[best_train_idx]),
                mean_evaders_caught=float(np.mean(eval_res.evaders_caught)),
                mean_cycles=float(np.mean(eval_res.cycles)),
                mutation_sigma=float(ga.sigma),
                stagnation_counter=int(ga.stagnation_counter),
                elapsed_sec=float(elapsed),
            )
            history.append(asdict(stats))

            print(
                f"Gen {gen:03d}/{cfg.ga.generations - 1:03d} | "
                f"train_ret best={stats.best_return:+6.2f} mean={stats.mean_return:+6.2f} | "
                f"caught best={stats.best_evaders_caught:4.1f} mean={stats.mean_evaders_caught:4.1f}/30 | "
                f"val_fit={elite_val_fit:+6.2f} (val_caught={elite_val_caught:4.1f}) | "
                f"cyc={stats.mean_cycles:5.1f} | sigma={stats.mutation_sigma:.4f} | {elapsed:.2f}s",
                flush=True,
            )

            if gen < cfg.ga.generations - 1:
                ga.step(
                    eval_res.fitnesses,
                    elite_override_order=elite_override_order,
                    tracked_val_fitness=elite_val_fit,
                    validated_best_genome=best_val_genome,
                )

            if (gen + 1) % 10 == 0 or gen == cfg.ga.generations - 1:
                last_genome = best_val_genome if best_val_genome is not None else ga.population[gen_champion_idx]
                _save_checkpoint(
                    out_dir / "last_checkpoint.pt",
                    template_policy,
                    last_genome,
                    cfg,
                    generation=gen,
                    best_val_fitness=best_val_fitness,
                )
                with open(out_dir / "history.json", "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

    # Final held-out evaluation against baselines on the canonical PettingZoo environment
    assert best_val_genome is not None
    template_policy.set_genome(best_val_genome)
    template_policy.eval()

    print(
        f"\n[Final Evaluation on Official PettingZoo {cfg.env.env_id}, {len(test_seeds)} test seeds]",
        flush=True,
    )
    print("  [1/4] Evaluating Random Baseline...", flush=True)
    random_summary, _ = evaluate_random_baseline(cfg.env, test_seeds, verbose=True)
    print("  [2/4] Evaluating Greedy Heuristic Baseline...", flush=True)
    greedy_summary, _ = evaluate_greedy_baseline(cfg.env, test_seeds, verbose=True)
    print("  [3/4] Evaluating Trained GA Policy (Stochastic)...", flush=True)
    trained_summary, _ = evaluate_policy_suite(
        template_policy, cfg.env, test_seeds, stochastic=cfg.model.stochastic_eval, device=device, verbose=True
    )
    print("  [4/4] Evaluating Trained GA Policy (Greedy Argmax)...", flush=True)
    trained_det_summary, _ = evaluate_policy_suite(
        template_policy, cfg.env, test_seeds, stochastic=False, device=device, verbose=True
    )
    trained_det_summary.name = "Trained GA (Greedy Argmax)"

    for s in (random_summary, greedy_summary, trained_summary, trained_det_summary):
        print(
            f"  {s.name:28s} | return={s.mean_agent_return:+6.2f}±{s.std_agent_return:5.2f} | "
            f"caught={s.mean_evaders_caught:5.2f}/30 ({s.catch_rate_pct:5.1f}%) | "
            f"cycles={s.mean_cycles:5.1f} | full_clear={s.full_clear_rate_pct:5.1f}%",
            flush=True,
        )

    benchmarks_for_plot = [random_summary, greedy_summary, trained_summary]
    plot_training_curves(history, benchmarks_for_plot, out_dir / "training_curves.png")

    gif_metrics = None
    if cfg.save_gif:
        print(f"[GIF] Rendering episode GIF to {out_dir / 'best_pursuit.gif'}...", flush=True)
        gif_metrics = render_episode_gif(
            template_policy,
            cfg.env,
            seed=test_seeds[0],
            gif_path=out_dir / "best_pursuit.gif",
            stochastic=cfg.model.stochastic_eval,
            device=device,
        )
        print(
            f"[GIF] Saved {out_dir / 'best_pursuit.gif'} "
            f"(caught={gif_metrics.evaders_caught}/30, cycles={gif_metrics.cycles}, "
            f"return={gif_metrics.mean_agent_return:+.2f})",
            flush=True,
        )

    summary_payload = {
        "best_val_fitness": best_val_fitness,
        "num_params": ga.num_params,
        "device": str(device),
        "benchmarks": {
            "random": asdict(random_summary),
            "greedy_heuristic": asdict(greedy_summary),
            "trained_ga_stochastic": asdict(trained_summary),
            "trained_ga_argmax": asdict(trained_det_summary),
        },
        "gif_episode": asdict(gif_metrics) if gif_metrics is not None else None,
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    return summary_payload


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train Deep MARL agents on PettingZoo sisl/pursuit-v5 using a Genetic Algorithm."
    )
    parser.add_argument("--generations", type=int, default=70)
    parser.add_argument("--pop-size", type=int, default=96)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=3, help="Training episodes per genome per generation")
    parser.add_argument("--elite-eval-episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--workers", type=int, default=12, help="CPU worker processes when --device cpu")
    parser.add_argument(
        "--arch",
        type=str,
        default="eq_transformer",
        choices=["cnn", "cnn_gru", "eq_cnn_gru", "transformer", "eq_transformer"],
    )
    parser.add_argument("--conv-channels", type=int, nargs=3, default=[8, 12, 8])
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--gru-dim", type=int, default=20)
    parser.add_argument("--ga-mode", type=str, default="standard", choices=["standard", "guided"])
    parser.add_argument("--consensus-lr", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--deterministic-train", action="store_true")
    parser.add_argument("--crossover-type", type=str, default="neuron", choices=["hybrid", "neuron", "arithmetic", "none"])
    parser.add_argument("--crossover-rate", type=float, default=0.30)
    parser.add_argument("--sigma", type=float, default=0.055)
    parser.add_argument("--sigma-min", type=float, default=0.007)
    parser.add_argument("--mutation-decay", type=float, default=0.978)
    parser.add_argument("--catch-bonus", type=float, default=0.8)
    parser.add_argument("--max-cycles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs/pursuit_ga")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        env=EnvConfig(max_cycles=args.max_cycles),
        model=ModelConfig(
            arch=args.arch,
            conv_channels=tuple(args.conv_channels),
            hidden_dim=args.hidden_dim,
            gru_dim=args.gru_dim,
            stochastic_eval=not args.deterministic_train,
            temperature=args.temperature,
        ),
        ga=GAConfig(
            ga_mode=args.ga_mode,
            population_size=args.pop_size,
            generations=args.generations,
            elite_count=args.elite_count,
            episodes_per_genome=args.episodes,
            elite_eval_episodes=args.elite_eval_episodes,
            crossover_type=args.crossover_type,
            crossover_rate=args.crossover_rate,
            mutation_sigma=args.sigma,
            mutation_sigma_min=args.sigma_min,
            mutation_decay=args.mutation_decay,
            consensus_lr=args.consensus_lr,
            catch_bonus_weight=args.catch_bonus,
        ),
        seed=args.seed,
        device=args.device,
        num_workers=args.workers,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        save_gif=not args.no_gif,
    )


if __name__ == "__main__":
    train(parse_args())
