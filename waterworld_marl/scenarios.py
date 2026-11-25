"""WaterWorld scenario definitions, parallel vectorized environments, and evaluation helpers."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from waterworld_marl.env import parallel_env


@dataclass(frozen=True)
class WaterworldScenario:
    """Configuration for one WaterWorld experiment scenario from the article."""

    key: str
    title_ru: str
    n_pursuers: int = 5
    n_evaders: int = 5
    n_poisons: int = 10
    n_coop: int = 2
    n_sensors: int = 30
    speed_features: bool = True
    max_cycles: int = 500
    pursuer_max_accel: float = 0.5

    def to_env_kwargs(self, render_mode: str | None = None) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "n_pursuers": self.n_pursuers,
            "n_evaders": self.n_evaders,
            "n_poisons": self.n_poisons,
            "n_coop": self.n_coop,
            "n_sensors": self.n_sensors,
            "speed_features": self.speed_features,
            "max_cycles": self.max_cycles,
            "pursuer_max_accel": self.pursuer_max_accel,
        }
        if render_mode is not None:
            kw["render_mode"] = render_mode
        return kw

    @property
    def obs_dim(self) -> int:
        feats_per_sensor = 8 if self.speed_features else 5
        return feats_per_sensor * self.n_sensors + 2


SCENARIOS: dict[str, WaterworldScenario] = {
    "base": WaterworldScenario(
        key="base",
        title_ru="Базовая",
        n_pursuers=5,
        n_evaders=5,
        n_poisons=10,
        n_coop=2,
        n_sensors=30,
        speed_features=True,
    ),
    "single": WaterworldScenario(
        key="single",
        title_ru="Упрощённый вариант",
        n_pursuers=1,
        n_evaders=5,
        n_poisons=10,
        n_coop=1,
        n_sensors=30,
        speed_features=True,
    ),
    "no_speed": WaterworldScenario(
        key="no_speed",
        title_ru="Отключение speed_features",
        n_pursuers=5,
        n_evaders=5,
        n_poisons=10,
        n_coop=2,
        n_sensors=30,
        speed_features=False,
    ),
    "few_sensors": WaterworldScenario(
        key="few_sensors",
        title_ru="Уменьшение количества сенсоров",
        n_pursuers=5,
        n_evaders=5,
        n_poisons=10,
        n_coop=2,
        n_sensors=10,
        speed_features=True,
    ),
    "more_poison": WaterworldScenario(
        key="more_poison",
        title_ru="Увеличение яда и уменьшение еды",
        n_pursuers=5,
        n_evaders=3,
        n_poisons=25,
        n_coop=2,
        n_sensors=30,
        speed_features=True,
    ),
}


def make_waterworld_env(
    scenario: WaterworldScenario,
    render_mode: str | None = None,
) -> Any:
    """Create a PettingZoo Waterworld_v4 ParallelEnv for the given scenario."""
    return parallel_env(**scenario.to_env_kwargs(render_mode=render_mode))


def build_agent_phase_vec(n_agents: int) -> np.ndarray:
    """Return (n_agents, 2) periodic sin/cos agent role coordinates."""
    angles = 2.0 * np.pi * np.arange(n_agents, dtype=np.float32) / max(n_agents, 1)
    return np.stack([np.sin(angles), np.cos(angles)], axis=-1).astype(np.float32)


def _vec_worker(conn: Any, scenario: WaterworldScenario, base_seed: int, rank: int) -> None:
    env = make_waterworld_env(scenario)
    agent_ids = list(env.possible_agents)
    n_ag = len(agent_ids)
    ep_counter = 0
    ep_team_ret = 0.0
    ep_agent_rets = np.zeros(n_ag, dtype=np.float64)
    ep_food_events = 0
    ep_poison_hits = 0

    def _reset_local(seed: int) -> np.ndarray:
        nonlocal ep_team_ret, ep_agent_rets, ep_food_events, ep_poison_hits
        obs_dict, _ = env.reset(seed=int(seed))
        ep_team_ret = 0.0
        ep_agent_rets.fill(0.0)
        ep_food_events = 0
        ep_poison_hits = 0
        return np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)

    try:
        while True:
            cmd = conn.recv()
            tag = cmd[0]
            if tag == "reset":
                seed = cmd[1]
                conn.send(_reset_local(seed))
            elif tag == "step":
                actions_np: np.ndarray = cmd[1]  # (n_ag, 2)
                act_dict = {
                    agent_ids[i]: np.clip(actions_np[i], -1.0, 1.0).astype(np.float32)
                    for i in range(n_ag)
                }
                obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(act_dict)
                r_vec = np.array([float(rew_dict[a]) for a in agent_ids], dtype=np.float32)
                o_mat = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)

                ep_agent_rets += r_vec
                ep_team_ret += float(r_vec.sum())
                # food_reward=10.0, encounter=0.01, max thrust penalty ~ -0.5
                ep_food_events += int(np.sum(r_vec >= 4.0))
                ep_poison_hits += int(np.sum(o_mat[:, -1] > 0.5))

                done = (
                    (not env.agents)
                    or any(term_dict.values())
                    or any(trunc_dict.values())
                )
                ep_info = None
                if done:
                    ep_info = {
                        "team_return": float(ep_team_ret),
                        "mean_agent_return": float(ep_team_ret / max(n_ag, 1)),
                        "agent_returns": ep_agent_rets.copy(),
                        "food_events": int(ep_food_events),
                        "poison_hits": int(ep_poison_hits),
                    }
                    ep_counter += 1
                    o_mat = _reset_local(base_seed + rank * 100000 + ep_counter)
                conn.send((o_mat, r_vec, done, ep_info))
            elif tag == "close":
                break
    finally:
        env.close()
        conn.close()


class SubprocWaterworldVecEnv:
    """Multiprocessing vectorized runner over parallel PettingZoo WaterWorld envs."""

    def __init__(
        self,
        scenario: WaterworldScenario,
        num_envs: int = 8,
        base_seed: int = 1000,
    ) -> None:
        self.scenario = scenario
        self.num_envs = int(num_envs)
        self.base_seed = int(base_seed)
        self.n_agents = scenario.n_pursuers
        self.obs_dim = scenario.obs_dim

        ctx = mp.get_context("spawn")
        self._conns: list[Any] = []
        self._procs: list[Any] = []
        for rank in range(self.num_envs):
            parent_c, child_c = ctx.Pipe()
            p = ctx.Process(
                target=_vec_worker,
                args=(child_c, scenario, self.base_seed, rank),
                daemon=True,
            )
            p.start()
            child_c.close()
            self._conns.append(parent_c)
            self._procs.append(p)
        self._closed = False

    def reset(self, seeds: list[int] | None = None) -> np.ndarray:
        if seeds is None:
            seeds = [self.base_seed + i * 100000 for i in range(self.num_envs)]
        for conn, s in zip(self._conns, seeds):
            conn.send(("reset", int(s)))
        obs_list = [conn.recv() for conn in self._conns]
        return np.stack(obs_list, axis=0)  # (num_envs, n_agents, obs_dim)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any] | None]]:
        for conn, a in zip(self._conns, actions):
            conn.send(("step", a))
        obs_list = []
        rew_list = []
        done_list = []
        infos = []
        for conn in self._conns:
            o, r, d, info = conn.recv()
            obs_list.append(o)
            rew_list.append(r)
            done_list.append(d)
            infos.append(info)
        return (
            np.stack(obs_list, axis=0),
            np.stack(rew_list, axis=0),
            np.asarray(done_list, dtype=bool),
            infos,
        )

    def close(self) -> None:
        if not self._closed:
            for conn in self._conns:
                try:
                    conn.send(("close",))
                except Exception:
                    pass
            for p in self._procs:
                p.join(timeout=2.0)
                if p.is_alive():
                    p.terminate()
            self._closed = True

    def __enter__(self) -> "SubprocWaterworldVecEnv":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def rollout_single_episode(
    scenario: WaterworldScenario,
    action_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    seed: int,
    capture_frames: bool = False,
) -> tuple[dict[str, float], list[np.ndarray]]:
    """Run one deterministic-seed WaterWorld episode sequentially using `action_fn(obs_na, phase_na)`."""
    env = make_waterworld_env(scenario, render_mode="rgb_array" if capture_frames else None)
    try:
        obs_dict, _ = env.reset(seed=int(seed))
        if hasattr(action_fn, "reset"):
            action_fn.reset()
        agent_ids = list(env.possible_agents)
        phase = build_agent_phase_vec(len(agent_ids))
        team_ret = 0.0
        food_events = 0
        poison_hits = 0
        cycles = 0
        frames: list[np.ndarray] = []

        if capture_frames:
            fr = env.render()
            if fr is not None:
                frames.append(fr)

        while env.agents:
            obs_mat = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
            acts = np.clip(action_fn(obs_mat, phase), -1.0, 1.0).astype(np.float32)
            act_dict = {agent_ids[i]: acts[i] for i in range(len(agent_ids))}
            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(act_dict)
            r_vec = np.array([float(rew_dict[a]) for a in agent_ids], dtype=np.float32)
            o_next = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
            team_ret += float(r_vec.sum())
            food_events += int(np.sum(r_vec >= 4.0))
            poison_hits += int(np.sum(o_next[:, -1] > 0.5))
            cycles += 1
            if capture_frames:
                fr = env.render()
                if fr is not None:
                    frames.append(fr)
            if (not env.agents) or any(term_dict.values()) or any(trunc_dict.values()):
                break

        metrics = {
            "team_return": float(team_ret),
            "mean_agent_return": float(team_ret / max(len(agent_ids), 1)),
            "food_events": float(food_events),
            "poison_hits": float(poison_hits),
            "cycles": float(cycles),
        }
        return metrics, frames
    finally:
        env.close()


def evaluate_policy_parallel(
    vec_env: SubprocWaterworldVecEnv,
    action_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    seeds: list[int],
) -> dict[str, float]:
    """Evaluate `action_fn(obs_e_a_d, phase_a_2)` across batches of `num_envs` episodes on held-out seeds."""
    scenario = vec_env.scenario
    phase = build_agent_phase_vec(scenario.n_pursuers)
    results: list[dict[str, Any]] = []

    for start in range(0, len(seeds), vec_env.num_envs):
        chunk = seeds[start : start + vec_env.num_envs]
        # Pad chunk to vec_env.num_envs and ignore padded slots
        padded = chunk + [chunk[-1]] * (vec_env.num_envs - len(chunk))
        obs = vec_env.reset(padded)
        if hasattr(action_fn, "reset"):
            action_fn.reset()
        done_mask = np.zeros(vec_env.num_envs, dtype=bool)
        for _ in range(scenario.max_cycles):
            acts = np.clip(action_fn(obs, phase), -1.0, 1.0).astype(np.float32)
            obs, _, dones, infos = vec_env.step(acts)
            for idx in range(len(chunk)):
                if not done_mask[idx] and dones[idx] and infos[idx] is not None:
                    results.append(infos[idx])
                    done_mask[idx] = True
            if bool(np.all(done_mask[: len(chunk)])):
                break

    team_rets = [r["team_return"] for r in results]
    mean_rets = [r["mean_agent_return"] for r in results]
    foods = [r["food_events"] for r in results]
    poisons = [r["poison_hits"] for r in results]
    return {
        "episodes": float(len(results)),
        "mean_team_return": float(np.mean(team_rets)),
        "std_team_return": float(np.std(team_rets)),
        "mean_agent_return": float(np.mean(mean_rets)),
        "std_agent_return": float(np.std(mean_rets)),
        "mean_food_events": float(np.mean(foods)),
        "mean_poison_hits": float(np.mean(poisons)),
    }
