"""Episode rollout and multi-process parallel fitness evaluation for Pursuit."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pursuit_ga.config import EnvConfig, ModelConfig
from pursuit_ga.env_utils import (
    get_evaders_remaining,
    make_pursuit_env,
    preprocess_observations,
)
from pursuit_ga.networks import PursuitPolicyNet, build_agent_phase_features


@dataclass
class EpisodeMetrics:
    """Detailed metrics from one Pursuit episode."""

    fitness: float
    mean_agent_return: float
    team_return: float
    evaders_caught: int
    evaders_total: int
    cycles: int
    cleared_all: bool


@dataclass
class PopulationEvaluation:
    """Aggregated evaluation results across an entire population."""

    fitnesses: np.ndarray
    mean_agent_returns: np.ndarray
    team_returns: np.ndarray
    evaders_caught: np.ndarray
    cycles: np.ndarray
    clear_rates: np.ndarray


@torch.inference_mode()
def run_episode(
    env: Any,
    policy: PursuitPolicyNet,
    seed: int,
    n_evaders: int,
    agent_phase_t: torch.Tensor,
    catch_bonus_weight: float = 0.0,
    stochastic: bool | None = None,
    temperature: float | None = None,
    action_seed: int | None = None,
    device: torch.device | None = None,
    capture_frames: bool = False,
) -> tuple[EpisodeMetrics, list[np.ndarray]]:
    """Run a single Pursuit episode using the given policy network."""
    dev = device or next(policy.parameters()).device
    use_stochastic = policy.cfg.stochastic_eval if stochastic is None else stochastic
    temp = max(policy.cfg.temperature if temperature is None else temperature, 1e-4)

    obs_dict, _ = env.reset(seed=int(seed))
    agent_ids = list(env.possible_agents)
    n_agents = len(agent_ids)
    n_actions = policy.n_actions

    hidden = policy.init_hidden(n_agents, device=dev)
    prev_actions = torch.zeros(n_agents, n_actions, dtype=torch.float32, device=dev)
    phase_t = agent_phase_t.to(dev)

    gen: torch.Generator | None = None
    if use_stochastic:
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed if action_seed is None else action_seed) + 100003)

    team_return = 0.0
    cycles = 0
    frames: list[np.ndarray] = []

    if capture_frames:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    while env.agents:
        obs_chw = preprocess_observations(obs_dict, agent_ids)
        obs_t = torch.from_numpy(obs_chw).to(dev)

        logits, hidden = policy(obs_t, prev_actions, phase_t, hidden)

        if use_stochastic:
            probs = torch.softmax(logits / temp, dim=-1)
            actions_t = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(1)
        else:
            actions_t = torch.argmax(logits, dim=-1)

        actions_np = actions_t.cpu().numpy()
        action_dict = {ag: int(actions_np[i]) for i, ag in enumerate(agent_ids)}

        obs_dict, rewards, terminations, truncations, _ = env.step(action_dict)
        team_return += float(sum(rewards.values()))
        cycles += 1

        if capture_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        prev_actions.zero_()
        prev_actions.scatter_(1, actions_t.unsqueeze(1), 1.0)

        if not env.agents or any(terminations.values()) or any(truncations.values()):
            break

    remaining = get_evaders_remaining(env)
    evaders_caught = int(n_evaders - remaining)
    mean_agent_return = float(team_return / max(n_agents, 1))
    fitness = float(mean_agent_return + catch_bonus_weight * evaders_caught)

    metrics = EpisodeMetrics(
        fitness=fitness,
        mean_agent_return=mean_agent_return,
        team_return=team_return,
        evaders_caught=evaders_caught,
        evaders_total=n_evaders,
        cycles=cycles,
        cleared_all=(remaining == 0),
    )
    return metrics, frames


@torch.inference_mode()
def evaluate_genome_episodes(
    env: Any,
    policy: PursuitPolicyNet,
    genome: np.ndarray,
    seeds: list[int],
    n_evaders: int,
    agent_phase_t: torch.Tensor,
    catch_bonus_weight: float = 0.0,
    stochastic: bool | None = None,
    device: torch.device | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Load genome into policy and evaluate across multiple seeds."""
    policy.set_genome(genome)
    fits, rets, team_rets, caught, cycs, clears = [], [], [], [], [], []
    for s in seeds:
        m, _ = run_episode(
            env=env,
            policy=policy,
            seed=s,
            n_evaders=n_evaders,
            agent_phase_t=agent_phase_t,
            catch_bonus_weight=catch_bonus_weight,
            stochastic=stochastic,
            device=device,
        )
        fits.append(m.fitness)
        rets.append(m.mean_agent_return)
        team_rets.append(m.team_return)
        caught.append(float(m.evaders_caught))
        cycs.append(float(m.cycles))
        clears.append(1.0 if m.cleared_all else 0.0)

    return (
        float(np.mean(fits)),
        float(np.mean(rets)),
        float(np.mean(team_rets)),
        float(np.mean(caught)),
        float(np.mean(cycs)),
        float(np.mean(clears)),
    )


# Worker process state (one persistent env + policy per worker process)
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(
    env_cfg: EnvConfig,
    model_cfg: ModelConfig,
    catch_bonus_weight: float,
) -> None:
    torch.set_num_threads(1)
    env = make_pursuit_env(env_cfg)
    policy = PursuitPolicyNet(
        model_cfg=model_cfg,
        obs_range=env_cfg.obs_range,
        n_actions=5,
    )
    policy.eval()
    phase_np = build_agent_phase_features(env_cfg.n_pursuers)
    _WORKER_STATE["env"] = env
    _WORKER_STATE["policy"] = policy
    _WORKER_STATE["n_evaders"] = env_cfg.n_evaders
    _WORKER_STATE["phase_t"] = torch.from_numpy(phase_np)
    _WORKER_STATE["catch_bonus_weight"] = catch_bonus_weight


def _worker_eval_single_episode(
    task: tuple[int, np.ndarray, int],
) -> tuple[int, float, float, float, float, float, float]:
    genome_idx, genome, seed = task
    policy: PursuitPolicyNet = _WORKER_STATE["policy"]
    policy.set_genome(genome)
    m, _ = run_episode(
        env=_WORKER_STATE["env"],
        policy=policy,
        seed=seed,
        n_evaders=_WORKER_STATE["n_evaders"],
        agent_phase_t=_WORKER_STATE["phase_t"],
        catch_bonus_weight=_WORKER_STATE["catch_bonus_weight"],
    )
    return (
        genome_idx,
        m.fitness,
        m.mean_agent_return,
        m.team_return,
        float(m.evaders_caught),
        float(m.cycles),
        1.0 if m.cleared_all else 0.0,
    )


class ParallelGenomeEvaluator:
    """Evaluates populations of genomes across CPU workers with persistent envs."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        model_cfg: ModelConfig,
        catch_bonus_weight: float = 0.0,
        num_workers: int = 12,
    ) -> None:
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.catch_bonus_weight = catch_bonus_weight
        self.num_workers = max(1, int(num_workers))

        self._pool: mp.pool.Pool | None = None
        self._local_env: Any | None = None
        self._local_policy: PursuitPolicyNet | None = None
        self._local_phase_t: torch.Tensor | None = None

        if self.num_workers > 1:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(env_cfg, model_cfg, catch_bonus_weight),
            )
        else:
            self._local_env = make_pursuit_env(env_cfg)
            self._local_policy = PursuitPolicyNet(
                model_cfg=model_cfg,
                obs_range=env_cfg.obs_range,
                n_actions=5,
            )
            self._local_policy.eval()
            self._local_phase_t = torch.from_numpy(
                build_agent_phase_features(env_cfg.n_pursuers)
            )

    def evaluate_population(
        self,
        genomes: np.ndarray,
        episode_seeds: list[int],
    ) -> PopulationEvaluation:
        """Evaluate every genome in `genomes` on `episode_seeds`."""
        n = len(genomes)
        m_eps = max(len(episode_seeds), 1)
        if self._pool is not None:
            tasks = [
                (i, genomes[i], int(seed))
                for i in range(n)
                for seed in episode_seeds
            ]
            raw = self._pool.map(_worker_eval_single_episode, tasks, chunksize=1)
            sums = np.zeros((n, 6), dtype=np.float64)
            for idx, fit, ret, tret, caught, cyc, clr in raw:
                sums[idx, 0] += fit
                sums[idx, 1] += ret
                sums[idx, 2] += tret
                sums[idx, 3] += caught
                sums[idx, 4] += cyc
                sums[idx, 5] += clr
            arr = (sums / float(m_eps)).astype(np.float32)
        else:
            assert self._local_env is not None and self._local_policy is not None
            assert self._local_phase_t is not None
            results = [
                evaluate_genome_episodes(
                    env=self._local_env,
                    policy=self._local_policy,
                    genome=genomes[i],
                    seeds=episode_seeds,
                    n_evaders=self.env_cfg.n_evaders,
                    agent_phase_t=self._local_phase_t,
                    catch_bonus_weight=self.catch_bonus_weight,
                )
                for i in range(n)
            ]
            arr = np.array(results, dtype=np.float32)

        return PopulationEvaluation(
            fitnesses=arr[:, 0],
            mean_agent_returns=arr[:, 1],
            team_returns=arr[:, 2],
            evaders_caught=arr[:, 3],
            cycles=arr[:, 4],
            clear_rates=arr[:, 5],
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        if self._local_env is not None:
            self._local_env.close()
            self._local_env = None

    def __enter__(self) -> "ParallelGenomeEvaluator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
