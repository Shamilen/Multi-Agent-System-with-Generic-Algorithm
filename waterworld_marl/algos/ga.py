"""Compact 17-Parameter SO(2) Equivariant Neuroevolution (GA with CRN) for WaterWorld."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import math
import multiprocessing as mp
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from waterworld_marl.algos.common import resolve_device
from waterworld_marl.algos.ga_worker import (
    _proximity_dist_indices,
    build_sensor_basis,
    ga_worker_eval_one,
    ga_worker_init,
)
from waterworld_marl.env.gpu_waterworld import VectorizedWaterworldGPU
from waterworld_marl.scenarios import WaterworldScenario


class WaterworldGAPolicy(nn.Module):
    """Minimal SO(2)-equivariant proximity steering + food-contact wait brake (~17 params).

    - Each ray computes radial & tangential weights from local proximity/velocity features.
    - Symmetric unit ray vectors sum to zero on empty observations -> automatic coasting [0, 0].
    - `wait_logit` is gated strictly by `food_obs`, preventing degenerate unconditional braking.
    """

    basis: torch.Tensor

    def __init__(
        self, n_sensors: int, speed_features: bool, has_teammates: bool = True
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.speed_features = speed_features
        self.has_teammates = has_teammates
        self.feats_per_sensor = 8 if speed_features else 5
        self.dist_indices = _proximity_dist_indices(speed_features, has_teammates)
        self.register_buffer("basis", torch.from_numpy(build_sensor_basis(n_sensors)))

        self.ray_vec = nn.Linear(self.feats_per_sensor, 2, bias=False)
        self.coll_wait = nn.Linear(1, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.ray_vec.weight)
        nn.init.constant_(self.coll_wait.weight, 1.5)

    def forward(self, obs: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        del phase
        batch = obs.shape[:-1]
        sensors = (
            obs[..., :-2]
            .reshape(*batch, self.feats_per_sensor, self.n_sensors)
            .swapaxes(-1, -2)
            .clone()
        )
        for idx in self.dist_indices:
            sensors[..., idx] = 1.0 - sensors[..., idx]

        w_rt = self.ray_vec(sensors)
        steer = torch.einsum("...sk,skd->...d", w_rt, self.basis)

        food_obs = obs[..., -2 : -1]
        wait_sig = torch.sigmoid(torch.clamp(self.coll_wait(food_obs), -15.0, 15.0))
        coll_brake = 1.0 - food_obs * wait_sig

        act = 0.25 * torch.tanh(steer) * coll_brake
        norms = torch.linalg.norm(act, dim=-1, keepdim=True)
        return torch.where(norms < 0.08, torch.zeros_like(act), act)


def _eval_genome_on_pool(
    pool: ProcessPoolExecutor,
    layers: tuple[np.ndarray, ...],
    eval_seeds: list[int],
) -> dict[str, float]:
    ep_res = list(pool.map(ga_worker_eval_one, [(layers, s) for s in eval_seeds]))
    team_arr = np.asarray([r["team_return"] for r in ep_res], dtype=np.float64)
    ag_arr = np.asarray([r["mean_agent_return"] for r in ep_res], dtype=np.float64)
    food_arr = np.asarray([r["food_events"] for r in ep_res], dtype=np.float64)
    pois_arr = np.asarray([r["poison_hits"] for r in ep_res], dtype=np.float64)
    return {
        "episodes": float(len(ep_res)),
        "mean_team_return": float(team_arr.mean()),
        "std_team_return": float(team_arr.std()),
        "mean_agent_return": float(ag_arr.mean()),
        "std_agent_return": float(ag_arr.std()),
        "mean_food_events": float(food_arr.mean()),
        "mean_poison_hits": float(pois_arr.mean()),
    }


def _pack_genome(model: WaterworldGAPolicy) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()]).numpy().copy()


def _unpack_to_layers(
    genome: np.ndarray, template: WaterworldGAPolicy
) -> tuple[np.ndarray, ...]:
    idx = 0
    arrs = []
    for p in template.parameters():
        n = p.numel()
        arrs.append(genome[idx : idx + n].reshape(tuple(p.shape)).astype(np.float32))
        idx += n
    return tuple(arrs)


def _genome_to_state_dict(
    genome: np.ndarray, template: WaterworldGAPolicy
) -> dict[str, torch.Tensor]:
    idx = 0
    sd: dict[str, torch.Tensor] = {}
    for name, p in template.named_parameters():
        n = p.numel()
        sd[name] = torch.from_numpy(genome[idx : idx + n].reshape(tuple(p.shape)).copy()).float()
        idx += n
    sd["basis"] = template.basis.detach().cpu().clone()
    return sd


def _centered_rank_utilities(fitness: np.ndarray) -> np.ndarray:
    """Map perturbation fitness values onto zero-mean normalized centered ranks."""
    n = fitness.shape[0]
    if n < 2:
        return np.zeros_like(fitness, dtype=np.float64)
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(fitness)] = np.arange(n, dtype=np.float64)
    u = ranks / float(n - 1) - 0.5
    std = float(u.std())
    return (u / std) if std > 1e-8 else u


def build_ga_action_fn(
    scenario: WaterworldScenario,
    state_dict: dict[str, torch.Tensor],
    device: str = "auto",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Reconstruct a deterministic evaluation callback from a saved GA state dict."""
    dev = resolve_device(device)
    model = WaterworldGAPolicy(
        n_sensors=scenario.n_sensors,
        speed_features=scenario.speed_features,
        has_teammates=scenario.n_pursuers > 1,
    ).to(dev)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    def _act(obs_batch: np.ndarray, ph: np.ndarray) -> np.ndarray:
        del ph
        obs_t = torch.from_numpy(obs_batch.reshape(-1, scenario.obs_dim)).to(dev)
        with torch.inference_mode():
            acts = model(obs_t).cpu().numpy()
        return np.clip(acts, -1.0, 1.0).reshape(*obs_batch.shape[:-1], 2)

    return _act


@torch.no_grad()
def _rollout_genomes_gpu(
    sim: VectorizedWaterworldGPU,
    genomes: np.ndarray,
    basis: torch.Tensor,
    n_sensors: int,
    feats_per_sensor: int,
    dist_sign: torch.Tensor,
    dist_bias: torch.Tensor,
    crn_seed: int | None,
    eval_seed: int = 900_000,
) -> list[dict[str, float]]:
    b_cnt = genomes.shape[0]
    a_cnt = sim.n_pursuers
    g_t = torch.from_numpy(np.ascontiguousarray(genomes, dtype=np.float32)).to(sim.device)
    vw_t = g_t[:, : 2 * feats_per_sensor].view(b_cnt, 2, feats_per_sensor).transpose(-1, -2).contiguous()
    wait_w = g_t[:, 2 * feats_per_sensor : 2 * feats_per_sensor + 1].view(b_cnt, 1, 1)

    obs = sim.reset_shared_crn(crn_seed) if crn_seed is not None else sim.reset(seed=eval_seed)
    last_infos: list[dict[str, float]] | None = None
    for _ in range(sim.max_cycles):
        sensors = obs[..., :-2].view(b_cnt, a_cnt, feats_per_sensor, n_sensors).transpose(-1, -2)
        prox = sensors * dist_sign + dist_bias
        w_rt = torch.matmul(prox, vw_t.unsqueeze(1))
        steer = torch.einsum("bask,skd->bad", w_rt, basis)
        food_obs = obs[..., -2:-1]
        wait_sig = torch.sigmoid(torch.clamp(food_obs * wait_w, -15.0, 15.0))
        act = 0.25 * torch.tanh(steer) * (1.0 - food_obs * wait_sig)
        norms = torch.linalg.norm(act, dim=-1, keepdim=True)
        act = torch.where(norms < 0.08, 0.0, act)
        obs, _, _, infos = sim.step(act)
        if infos is not None:
            last_infos = infos
    assert last_infos is not None
    return last_infos


def _summarize_gpu_infos(ep_res: list[dict[str, float]]) -> dict[str, float]:
    team_arr = np.asarray([r["team_return"] for r in ep_res], dtype=np.float64)
    ag_arr = np.asarray([r["mean_agent_return"] for r in ep_res], dtype=np.float64)
    food_arr = np.asarray([r["food_events"] for r in ep_res], dtype=np.float64)
    pois_arr = np.asarray([r["poison_hits"] for r in ep_res], dtype=np.float64)
    return {
        "episodes": float(len(ep_res)),
        "mean_team_return": float(team_arr.mean()),
        "std_team_return": float(team_arr.std()),
        "mean_agent_return": float(ag_arr.mean()),
        "std_agent_return": float(ag_arr.std()),
        "mean_food_events": float(food_arr.mean()),
        "mean_poison_hits": float(pois_arr.mean()),
    }


def train_ga(
    scenario: WaterworldScenario,
    total_episodes: int = 600,
    pop_size: int = 12,
    num_workers: int = 12,
    eval_interval_episodes: int = 100,
    eval_episodes: int = 8,
    seed: int = 42,
    device: str = "auto",
    verbose: bool = True,
    use_gpu_sim: bool = True,
) -> dict[str, Any]:
    """Train 17-param SO(2) equivariant evolutionary policy with antithetic CRN."""
    t_start = time.perf_counter()
    dev = resolve_device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    has_teammates = scenario.n_pursuers > 1
    template = WaterworldGAPolicy(
        n_sensors=scenario.n_sensors,
        speed_features=scenario.speed_features,
        has_teammates=has_teammates,
    )

    crn_seeds_per_gen = 1
    generations = max(1, math.ceil(total_episodes / (pop_size * crn_seeds_per_gen)))
    eval_seeds = [900_000 + i for i in range(eval_episodes)]

    mean_genome = _pack_genome(template).astype(np.float32)
    elite_genome = mean_genome.copy()
    dim = mean_genome.shape[0]
    n_anchors = 2 if pop_size >= 4 and (pop_size % 2 == 0) else 1
    n_pairs = max(1, (pop_size - n_anchors) // 2)

    vel = np.zeros(dim, dtype=np.float64)
    es_lr = 0.18
    momentum = 0.5

    best_eval_team_ret = -float("inf")
    best_state_dict = _genome_to_state_dict(mean_genome, template)

    ep_curve: list[dict[str, float]] = []
    eval_curve: list[dict[str, float]] = []
    episodes_done = 0
    next_eval_at = eval_interval_episodes

    gpu_mode = bool(use_gpu_sim and dev.type == "cuda")
    if gpu_mode:
        basis_dev = template.basis.to(dev)
        dist_sign = torch.ones((1, 1, 1, template.feats_per_sensor), dtype=torch.float32, device=dev)
        dist_bias = torch.zeros((1, 1, 1, template.feats_per_sensor), dtype=torch.float32, device=dev)
        dist_sign[..., template.dist_indices] = -1.0
        dist_bias[..., template.dist_indices] = 1.0
        gpu_pop_sim = VectorizedWaterworldGPU(pop_size, scenario, dev, seed=seed)
        gpu_eval_sim = VectorizedWaterworldGPU(eval_episodes, scenario, dev, seed=900_000)
        pool_cm: Any = nullcontext()

        def eval_population(pop_list: list[np.ndarray], crn_seed: int) -> list[dict[str, float]]:
            return _rollout_genomes_gpu(
                gpu_pop_sim,
                np.stack(pop_list, axis=0),
                basis_dev,
                scenario.n_sensors,
                template.feats_per_sensor,
                dist_sign,
                dist_bias,
                crn_seed=crn_seed,
            )

        def eval_single(genome: np.ndarray) -> dict[str, float]:
            batch_g = np.broadcast_to(genome[None, :], (eval_episodes, dim)).copy()
            ep_res = _rollout_genomes_gpu(
                gpu_eval_sim,
                batch_g,
                basis_dev,
                scenario.n_sensors,
                template.feats_per_sensor,
                dist_sign,
                dist_bias,
                crn_seed=None,
                eval_seed=900_000,
            )
            return _summarize_gpu_infos(ep_res)
    else:
        ctx = mp.get_context("spawn")
        workers = min(num_workers, pop_size * crn_seeds_per_gen)
        pool_cm = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=ga_worker_init,
            initargs=(scenario,),
        )

        def eval_population(pop_list: list[np.ndarray], crn_seed: int) -> list[dict[str, float]]:
            tasks = [(_unpack_to_layers(g, template), crn_seed) for g in pop_list]
            return list(pool_cm.map(ga_worker_eval_one, tasks))

        def eval_single(genome: np.ndarray) -> dict[str, float]:
            return _eval_genome_on_pool(pool_cm, _unpack_to_layers(genome, template), eval_seeds)

    with pool_cm:
        ev0 = eval_single(mean_genome)
        ev0["episode"] = 0.0
        eval_curve.append(ev0)

        for gen in range(generations):
            progress = gen / max(generations - 1, 1)
            sigma = max(0.14, 0.48 - 0.32 * progress)
            noises = rng.normal(0.0, 1.0, size=(n_pairs, dim)).astype(np.float32)

            population: list[np.ndarray] = [mean_genome.copy()]
            if n_anchors == 2:
                population.append(elite_genome.copy())
            for k in range(n_pairs):
                population.append((mean_genome + sigma * noises[k]).astype(np.float32))
                population.append((mean_genome - sigma * noises[k]).astype(np.float32))
            population = population[:pop_size]

            crn_seed = seed * 10_000 + gen * crn_seeds_per_gen
            results = eval_population(population, crn_seed)

            fitness = np.array([r["team_return"] for r in results], dtype=np.float64)

            for r in results:
                if episodes_done < total_episodes:
                    episodes_done += 1
                    ep_curve.append(
                        {
                            "episode": float(episodes_done),
                            "team_return": float(r["team_return"]),
                            "mean_agent_return": float(r["mean_agent_return"]),
                            "food_events": float(r["food_events"]),
                            "poison_hits": float(r["poison_hits"]),
                        }
                    )

            order = np.argsort(-fitness)
            champion_idx = int(order[0])
            champion_fit = float(fitness[champion_idx])
            if n_anchors == 1 or champion_idx != 1:
                elite_genome = population[champion_idx].copy()

            pert_fits = fitness[n_anchors:]
            used_pairs = pert_fits.shape[0] // 2
            if used_pairs >= 1:
                u = _centered_rank_utilities(pert_fits[: 2 * used_pairs])
                grad = np.mean((u[0::2] - u[1::2])[:, None] * noises[:used_pairs], axis=0)
                vel = momentum * vel + (1.0 - momentum) * grad
                mean_genome = (mean_genome + es_lr * vel).astype(np.float32)
                mean_genome[:-1] *= 0.96
                mean_genome = np.clip(mean_genome, -4.0, 4.0).astype(np.float32)

            if episodes_done >= next_eval_at or gen == generations - 1:
                ev = eval_single(mean_genome)
                ev["episode"] = float(episodes_done)
                eval_curve.append(ev)
                if ev["mean_team_return"] > best_eval_team_ret:
                    best_eval_team_ret = ev["mean_team_return"]
                    best_state_dict = _genome_to_state_dict(mean_genome, template)
                if gen == generations - 1:
                    ev_el = eval_single(elite_genome)
                    if ev_el["mean_team_return"] > best_eval_team_ret:
                        best_eval_team_ret = ev_el["mean_team_return"]
                        best_state_dict = _genome_to_state_dict(elite_genome, template)
                if verbose:
                    print(
                        f"  [GA :{scenario.key:11s}] gen={gen+1:2d}/{generations} ep={episodes_done:3d}/{total_episodes} | "
                        f"gen_best={champion_fit:+6.2f} | eval_team={ev['mean_team_return']:+6.2f} "
                        f"(agent={ev['mean_agent_return']:+5.2f}, food={ev['mean_food_events']:4.1f}, "
                        f"poison={ev['mean_poison_hits']:4.1f})",
                        flush=True,
                    )
                next_eval_at += eval_interval_episodes

            if gen == generations - 1 or episodes_done >= total_episodes:
                break

        best_genome_vec = np.concatenate(
            [
                best_state_dict[k].detach().cpu().numpy().reshape(-1)
                for k, _ in template.named_parameters()
            ]
        ).astype(np.float32)
        final_eval = eval_single(best_genome_vec)

    return {
        "algo": "GA",
        "scenario": scenario.key,
        "elapsed_sec": float(time.perf_counter() - t_start),
        "total_episodes": int(episodes_done),
        "train_updates": int(generations),
        "best_eval": final_eval,
        "episode_curve": ep_curve,
        "eval_curve": eval_curve,
        "state_dict": best_state_dict,
    }
