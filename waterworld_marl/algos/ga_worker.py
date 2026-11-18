"""Torch-free NumPy worker functions for compact SO(2)-equivariant GA WaterWorld rollouts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from waterworld_marl.scenarios import WaterworldScenario, make_waterworld_env

_GA_WORKER_ENV: Any = None
_GA_WORKER_AGENTS: list[str] | None = None
_GA_WORKER_BASIS: np.ndarray | None = None
_GA_WORKER_FEAT_PER_SENSOR: int = 8
_GA_WORKER_N_SENSORS: int = 30
_GA_WORKER_HAS_TEAMMATES: bool = True


def build_sensor_basis(n_sensors: int) -> np.ndarray:
    """Return (S, 2, 2) tensor of radial u_k=[cos, sin] and tangential [-sin, cos] unit vectors."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_sensors + 1, dtype=np.float32)[:-1]
    c = np.cos(angles)
    s = np.sin(angles)
    radial = np.stack([c, s], axis=-1)
    tangential = np.stack([-s, c], axis=-1)
    return np.stack([radial, tangential], axis=1).astype(np.float32)


def _proximity_dist_indices(speed_features: bool, has_teammates: bool) -> list[int]:
    if speed_features:
        return [0, 1, 2, 4, 6] if has_teammates else [0, 1, 2, 4]
    return [0, 1, 2, 3, 4] if has_teammates else [0, 1, 2, 3]


def numpy_equivariant_forward(
    obs: np.ndarray,
    phase: np.ndarray,
    layers: tuple[np.ndarray, ...],
    basis: np.ndarray,
    n_sensors: int,
    feats_per_sensor: int,
    has_teammates: bool = True,
) -> np.ndarray:
    """Bit-exact NumPy mirror of WaterworldGAPolicy.forward (empty ring -> exact [0, 0])."""
    del phase
    vw, wait_w = layers
    batch = obs.shape[:-1]
    sensors = obs[..., :-2].reshape(*batch, feats_per_sensor, n_sensors).swapaxes(-1, -2).copy()
    speed_features = feats_per_sensor == 8
    for idx in _proximity_dist_indices(speed_features, has_teammates):
        sensors[..., idx] = 1.0 - sensors[..., idx]

    w_rt = sensors @ vw.T  # (..., S, 2)
    steer = np.einsum("...sk,skd->...d", w_rt, basis)

    food_obs = obs[..., -2 : -1]
    wait_sig = 1.0 / (1.0 + np.exp(-np.clip(food_obs @ wait_w.T, -15.0, 15.0)))
    coll_brake = 1.0 - food_obs * wait_sig

    act = 0.25 * np.tanh(steer) * coll_brake
    norms = np.linalg.norm(act, axis=-1, keepdims=True)
    return np.where(norms < 0.08, np.float32(0.0), act).astype(np.float32)


def ga_worker_init(scenario: WaterworldScenario) -> None:
    global _GA_WORKER_ENV, _GA_WORKER_AGENTS, _GA_WORKER_BASIS
    global _GA_WORKER_FEAT_PER_SENSOR, _GA_WORKER_N_SENSORS, _GA_WORKER_HAS_TEAMMATES
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _GA_WORKER_ENV = make_waterworld_env(scenario, render_mode=None)
    _GA_WORKER_N_SENSORS = scenario.n_sensors
    _GA_WORKER_FEAT_PER_SENSOR = 8 if scenario.speed_features else 5
    _GA_WORKER_HAS_TEAMMATES = scenario.n_pursuers > 1
    _GA_WORKER_BASIS = build_sensor_basis(scenario.n_sensors)
    _GA_WORKER_AGENTS = None


def ga_worker_eval_one(task: tuple[tuple[np.ndarray, ...], int]) -> dict[str, float]:
    global _GA_WORKER_AGENTS
    layers, seed = task
    env = _GA_WORKER_ENV
    obs_dict, _ = env.reset(seed=int(seed))
    if _GA_WORKER_AGENTS is None:
        _GA_WORKER_AGENTS = list(env.possible_agents)
    agents = _GA_WORKER_AGENTS
    n_agents = len(agents)
    dummy_phase = np.zeros((n_agents, 2), dtype=np.float32)

    agent_returns = np.zeros(n_agents, dtype=np.float64)
    food_events = 0
    poison_hits = 0

    while env.agents:
        o_mat = np.stack([obs_dict[a] for a in agents], axis=0).astype(np.float32)
        acts_mat = np.clip(
            numpy_equivariant_forward(
                o_mat,
                dummy_phase,
                layers,
                _GA_WORKER_BASIS,
                _GA_WORKER_N_SENSORS,
                _GA_WORKER_FEAT_PER_SENSOR,
                has_teammates=_GA_WORKER_HAS_TEAMMATES,
            ),
            -1.0,
            1.0,
        )
        act_dict = {a: acts_mat[i] for i, a in enumerate(agents)}
        obs_dict, rew_dict, terms, truncs, _ = env.step(act_dict)
        r_vec = np.asarray([rew_dict[a] for a in agents], dtype=np.float64)
        agent_returns += r_vec
        food_events += int(np.sum(r_vec >= 4.0))
        poison_hits += int(np.sum(o_mat[:, -1] > 0.5))
        if all(terms[a] or truncs[a] for a in agents):
            break

    team_ret = float(agent_returns.sum())
    return {
        "team_return": team_ret,
        "mean_agent_return": float(team_ret / max(n_agents, 1)),
        "food_events": float(food_events),
        "poison_hits": float(poison_hits),
    }
