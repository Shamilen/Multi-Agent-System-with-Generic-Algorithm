"""Discretized N-Step Dueling Double DQN for continuous PettingZoo WaterWorld."""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from waterworld_marl.algos.common import (
    RayTransformerEncoder,
    RunningMeanStd,
    RunningMeanStdGPU,
    append_phase_features,
    append_phase_features_torch,
    resolve_device,
)
from waterworld_marl.env.gpu_waterworld import VectorizedWaterworldGPU
from waterworld_marl.scenarios import (
    SubprocWaterworldVecEnv,
    WaterworldScenario,
    build_agent_phase_vec,
    evaluate_policy_parallel,
)


def build_discrete_thrust_table() -> np.ndarray:
    """9-action thrust palette: STOP + 8 directions at 0.32 normalized thrust."""
    actions = [[0.0, 0.0]]
    for k in range(8):
        ang = 2.0 * math.pi * k / 8.0
        actions.append([0.32 * math.cos(ang), 0.32 * math.sin(ang)])
    return np.asarray(actions, dtype=np.float32)


class DuelingQNet(nn.Module):
    """Parameter-shared Dueling Q-network (Ray-Transformer or MLP) for MARL."""

    def __init__(
        self,
        in_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        arch: str = "transformer",
        n_sensors: int = 30,
        speed_features: bool = True,
    ) -> None:
        super().__init__()
        self.arch = arch
        if arch == "transformer":
            enc = RayTransformerEncoder(
                n_sensors=n_sensors,
                speed_features=speed_features,
                is_ctde_critic=False,
            )
            self.backbone: nn.Module = enc
            feat_dim = enc.out_dim
        elif arch == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            feat_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported DQN arch: {arch!r}")
        self.value = nn.Linear(feat_dim, 1)
        self.adv = nn.Linear(feat_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        v = self.value(h)
        a = self.adv(h)
        return v + (a - a.mean(dim=-1, keepdim=True))


class QMixerNet(nn.Module):
    """Monotonic QMIX hypernetwork mixing per-agent Q-values into joint team Q_tot.

    Non-negative weights (`torch.abs`) + ELU guarantee dQ_tot / dQ_i >= 0 everywhere,
    preserving Individual-Global-Max (IGM) for fully decentralized greedy execution.
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 16) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.embed_dim = int(embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_agents * self.embed_dim),
        )
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.embed_dim),
        )
        self.hyper_v = nn.Sequential(
            nn.Linear(self.state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bsz = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(bsz, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(bsz, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(bsz, self.embed_dim, 1)
        v = self.hyper_v(state).view(bsz, 1, 1)
        return (torch.bmm(hidden, w2) + v).squeeze(-1).squeeze(-1)


class ReplayBuffer:
    """Pre-allocated NumPy circular replay buffer storing N-step transitions."""

    def __init__(self, capacity: int, in_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, in_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.returns = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, in_dim), dtype=np.float32)
        self.boot_gammas = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        next_obs: np.ndarray,
        boot_gammas: np.ndarray,
    ) -> None:
        n = len(actions)
        if n == 0:
            return
        if n >= self.capacity:
            obs = obs[-self.capacity :]
            actions = actions[-self.capacity :]
            returns = returns[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            boot_gammas = boot_gammas[-self.capacity :]
            n = self.capacity
        end = self.ptr + n
        if end <= self.capacity:
            self.obs[self.ptr : end] = obs
            self.actions[self.ptr : end] = actions
            self.returns[self.ptr : end] = returns
            self.next_obs[self.ptr : end] = next_obs
            self.boot_gammas[self.ptr : end] = boot_gammas
        else:
            first = self.capacity - self.ptr
            rem = n - first
            self.obs[self.ptr :] = obs[:first]
            self.actions[self.ptr :] = actions[:first]
            self.returns[self.ptr :] = returns[:first]
            self.next_obs[self.ptr :] = next_obs[:first]
            self.boot_gammas[self.ptr :] = boot_gammas[:first]
            self.obs[:rem] = obs[first:]
            self.actions[:rem] = actions[first:]
            self.returns[:rem] = returns[first:]
            self.next_obs[:rem] = next_obs[first:]
            self.boot_gammas[:rem] = boot_gammas[first:]
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.returns[idx]).to(device),
            torch.from_numpy(self.next_obs[idx]).to(device),
            torch.from_numpy(self.boot_gammas[idx]).to(device),
        )


def build_dqn_action_fn(
    scenario: WaterworldScenario,
    state_dict: dict[str, torch.Tensor],
    obs_rms_state: dict[str, Any] | None = None,
    device: str = "auto",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Reconstruct a greedy evaluation callback from a saved Double DQN checkpoint."""
    dev = resolve_device(device)
    action_table = build_discrete_thrust_table()
    in_dim = scenario.obs_dim + 2
    arch = "transformer" if any("ray_proj" in k for k in state_dict) else "mlp"
    model = DuelingQNet(
        in_dim=in_dim,
        n_actions=len(action_table),
        hidden_dim=128,
        arch=arch,
        n_sensors=scenario.n_sensors,
        speed_features=scenario.speed_features,
    ).to(dev)
    model.load_state_dict(state_dict)
    model.eval()

    rms: RunningMeanStd | None = None
    if obs_rms_state is not None:
        rms = RunningMeanStd((scenario.obs_dim,))
        rms.mean = np.asarray(obs_rms_state["mean"], dtype=np.float64)
        rms.var = np.asarray(obs_rms_state["var"], dtype=np.float64)
        rms.count = float(obs_rms_state["count"])

    def _act(obs_batch: np.ndarray, ph: np.ndarray) -> np.ndarray:
        norm = obs_batch if rms is None else rms.normalize(obs_batch)
        feat = append_phase_features(norm, ph)
        flat = feat.reshape(-1, in_dim)
        with torch.inference_mode():
            q = model(torch.from_numpy(flat).to(dev))
            a_idx = torch.argmax(q, dim=-1).cpu().numpy()
        acts = action_table[a_idx]
        return acts.reshape(*obs_batch.shape[:-1], 2)

    return _act


def _emit_nstep_head(
    q: deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fold deque into an n-step transition starting from q[0]."""
    obs0, act0, _, _, _ = q[0]
    n_ag = len(act0)
    ret = np.zeros(n_ag, dtype=np.float32)
    disc = 1.0
    last_next = q[0][3]
    ended = False
    for _, _, rew_k, next_k, done_k in q:
        ret += disc * rew_k
        disc *= gamma
        last_next = next_k
        if done_k:
            ended = True
            break
    boot = np.zeros(n_ag, dtype=np.float32) if ended else np.full(n_ag, disc, dtype=np.float32)
    return obs0, act0, ret, last_next, boot


class ReplayBufferGPU:
    """Device-resident circular replay buffer eliminating CPU-to-GPU batch copies."""

    def __init__(self, capacity: int, in_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, in_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity,), dtype=torch.int64, device=device)
        self.returns = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, in_dim), dtype=torch.float32, device=device)
        self.boot_gammas = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    @torch.no_grad()
    def add_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        next_obs: torch.Tensor,
        boot_gammas: torch.Tensor,
    ) -> None:
        n = int(actions.shape[0])
        if n == 0:
            return
        if n >= self.capacity:
            obs = obs[-self.capacity :]
            actions = actions[-self.capacity :]
            returns = returns[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            boot_gammas = boot_gammas[-self.capacity :]
            n = self.capacity
        end = self.ptr + n
        if end <= self.capacity:
            self.obs[self.ptr : end].copy_(obs)
            self.actions[self.ptr : end].copy_(actions)
            self.returns[self.ptr : end].copy_(returns)
            self.next_obs[self.ptr : end].copy_(next_obs)
            self.boot_gammas[self.ptr : end].copy_(boot_gammas)
        else:
            first = self.capacity - self.ptr
            rem = n - first
            self.obs[self.ptr :].copy_(obs[:first])
            self.actions[self.ptr :].copy_(actions[:first])
            self.returns[self.ptr :].copy_(returns[:first])
            self.next_obs[self.ptr :].copy_(next_obs[:first])
            self.boot_gammas[self.ptr :].copy_(boot_gammas[:first])
            self.obs[:rem].copy_(obs[first:])
            self.actions[:rem].copy_(actions[first:])
            self.returns[:rem].copy_(returns[first:])
            self.next_obs[:rem].copy_(next_obs[first:])
            self.boot_gammas[:rem].copy_(boot_gammas[first:])
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx],
            self.returns[idx],
            self.next_obs[idx],
            self.boot_gammas[idx],
        )


class TeamReplayBufferGPU:
    """Device-resident joint team replay buffer storing both per-agent and team N-step returns."""

    def __init__(self, capacity: int, n_agents: int, in_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.n_agents = n_agents
        self.device = device
        self.obs = torch.zeros((capacity, n_agents, in_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, n_agents), dtype=torch.int64, device=device)
        self.agent_returns = torch.zeros((capacity, n_agents), dtype=torch.float32, device=device)
        self.team_returns = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, n_agents, in_dim), dtype=torch.float32, device=device)
        self.boot_gammas = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0

    @torch.no_grad()
    def add_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        agent_returns: torch.Tensor,
        team_returns: torch.Tensor,
        next_obs: torch.Tensor,
        boot_gammas: torch.Tensor,
    ) -> None:
        n = int(team_returns.shape[0])
        if n == 0:
            return
        if n >= self.capacity:
            obs = obs[-self.capacity :]
            actions = actions[-self.capacity :]
            agent_returns = agent_returns[-self.capacity :]
            team_returns = team_returns[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            boot_gammas = boot_gammas[-self.capacity :]
            n = self.capacity
        end = self.ptr + n
        if end <= self.capacity:
            self.obs[self.ptr : end].copy_(obs)
            self.actions[self.ptr : end].copy_(actions)
            self.agent_returns[self.ptr : end].copy_(agent_returns)
            self.team_returns[self.ptr : end].copy_(team_returns)
            self.next_obs[self.ptr : end].copy_(next_obs)
            self.boot_gammas[self.ptr : end].copy_(boot_gammas)
        else:
            first = self.capacity - self.ptr
            rem = n - first
            self.obs[self.ptr :].copy_(obs[:first])
            self.actions[self.ptr :].copy_(actions[:first])
            self.agent_returns[self.ptr :].copy_(agent_returns[:first])
            self.team_returns[self.ptr :].copy_(team_returns[:first])
            self.next_obs[self.ptr :].copy_(next_obs[:first])
            self.boot_gammas[self.ptr :].copy_(boot_gammas[:first])
            self.obs[:rem].copy_(obs[first:])
            self.actions[:rem].copy_(actions[first:])
            self.agent_returns[:rem].copy_(agent_returns[first:])
            self.team_returns[:rem].copy_(team_returns[first:])
            self.next_obs[:rem].copy_(next_obs[first:])
            self.boot_gammas[:rem].copy_(boot_gammas[first:])
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx],
            self.agent_returns[idx],
            self.team_returns[idx],
            self.next_obs[idx],
            self.boot_gammas[idx],
        )


def _build_team_state(obs_feat: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant global team state `(B, 2*obs_dim)` from `(B, A, obs_dim+2)`."""
    raw = obs_feat[..., :-2]
    return torch.cat([raw.mean(dim=1), raw.amax(dim=1)], dim=-1)


def _qmix_update_step(
    q_net: DuelingQNet,
    target_net: DuelingQNet,
    mixer: QMixerNet,
    target_mixer: QMixerNet,
    optimizer: torch.optim.Optimizer,
    online_params: list[torch.Tensor],
    target_params: list[torch.Tensor],
    b_s: torch.Tensor,
    b_a: torch.Tensor,
    b_r_ag: torch.Tensor,
    b_r_team: torch.Tensor,
    b_ns: torch.Tensor,
    b_bg: torch.Tensor,
    tau: float,
    qmix_weight: float = 0.35,
) -> None:
    """Hybrid Local-Accountable + Monotonic QMIX TD update.

    Preserves exact per-agent poison/thrust accountability via `b_r_ag` while
    propagating cooperative joint-capture credit through monotonic `QMixerNet`.
    """
    bsz, n_ag, in_dim = b_s.shape
    g_s = _build_team_state(b_s)
    g_ns = _build_team_state(b_ns)

    q_all = q_net(b_s.reshape(bsz * n_ag, in_dim)).reshape(bsz, n_ag, -1)
    chosen_q = q_all.gather(2, b_a.unsqueeze(-1)).squeeze(-1)
    q_tot = mixer(chosen_q, g_s)

    with torch.no_grad():
        next_best_a = torch.argmax(
            q_net(b_ns.reshape(bsz * n_ag, in_dim)), dim=-1
        ).reshape(bsz, n_ag, 1)
        t_q_all = target_net(b_ns.reshape(bsz * n_ag, in_dim)).reshape(bsz, n_ag, -1)
        next_q = t_q_all.gather(2, next_best_a).squeeze(-1)
        target_ag = b_r_ag + b_bg.unsqueeze(1) * next_q
        target_q_tot = target_mixer(next_q, g_ns)
        target_team = b_r_team + b_bg * target_q_tot

    loss_local = F.smooth_l1_loss(chosen_q, target_ag, beta=1.0)
    loss_qmix = F.smooth_l1_loss(q_tot, target_team, beta=2.0) / float(max(n_ag, 1))
    loss = (1.0 - qmix_weight) * loss_local + qmix_weight * loss_qmix
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(list(q_net.parameters()) + list(mixer.parameters()), 10.0)
    optimizer.step()
    with torch.no_grad():
        torch._foreach_mul_(target_params, 1.0 - tau)
        torch._foreach_add_(target_params, online_params, alpha=tau)


@torch.no_grad()
def _fold_gpu_nstep_team(
    q: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]],
    gamma: float,
    pos_share_alpha: float = 0.40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs0, act0, rew0, _, _ = q[0]
    ag_ret = torch.zeros_like(rew0)
    team_ret = torch.zeros((rew0.shape[0],), dtype=rew0.dtype, device=rew0.device)
    disc = 1.0
    last_next = q[0][3]
    ended = False
    for _, _, rew_k, next_k, done_k in q:
        if pos_share_alpha > 0.0 and rew_k.shape[-1] > 1:
            r_pos = torch.clamp(rew_k, min=0.0)
            r_neg = torch.clamp(rew_k, max=0.0)
            r_shaped = r_neg + (1.0 - pos_share_alpha) * r_pos + pos_share_alpha * r_pos.mean(dim=-1, keepdim=True)
        else:
            r_shaped = rew_k
        ag_ret.add_(r_shaped, alpha=disc)
        team_ret.add_(rew_k.sum(dim=-1), alpha=disc)
        disc *= gamma
        last_next = next_k
        if done_k:
            ended = True
            break
    boot = torch.zeros_like(team_ret) if ended else torch.full_like(team_ret, disc)
    return obs0, act0, ag_ret, team_ret, last_next, boot


def _dqn_update_step(
    q_net: DuelingQNet,
    target_net: DuelingQNet,
    optimizer: torch.optim.Optimizer,
    q_params: list[torch.Tensor],
    t_params: list[torch.Tensor],
    b_s: torch.Tensor,
    b_a: torch.Tensor,
    b_r: torch.Tensor,
    b_ns: torch.Tensor,
    b_bg: torch.Tensor,
    tau: float,
) -> None:
    q_pred = q_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_best = torch.argmax(q_net(b_ns), dim=-1, keepdim=True)
        q_next = target_net(b_ns).gather(1, next_best).squeeze(1)
        target = b_r + b_bg * q_next
    loss = F.smooth_l1_loss(q_pred, target, beta=1.0)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
    with torch.no_grad():
        torch._foreach_mul_(t_params, 1.0 - tau)
        torch._foreach_add_(t_params, q_params, alpha=tau)


@torch.no_grad()
def _fold_gpu_nstep(
    q: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]],
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs0, act0, _, _, _ = q[0]
    in_dim = obs0.shape[-1]
    ret = torch.zeros_like(q[0][2])
    disc = 1.0
    last_next = q[0][3]
    ended = False
    for _, _, rew_k, next_k, done_k in q:
        ret.add_(rew_k, alpha=disc)
        disc *= gamma
        last_next = next_k
        if done_k:
            ended = True
            break
    boot = torch.zeros_like(ret) if ended else torch.full_like(ret, disc)
    return (
        obs0.reshape(-1, in_dim),
        act0.reshape(-1),
        ret.reshape(-1),
        last_next.reshape(-1, in_dim),
        boot.reshape(-1),
    )


@torch.no_grad()
def _eval_dqn_gpu(
    eval_sim: VectorizedWaterworldGPU,
    q_net: DuelingQNet,
    obs_rms_gpu: RunningMeanStdGPU,
    action_table_t: torch.Tensor,
    phase_t: torch.Tensor,
    eval_seed: int = 900_000,
) -> dict[str, float]:
    obs = eval_sim.reset(seed=eval_seed)
    b_cnt, a_cnt, obs_dim = obs.shape
    in_dim = obs_dim + 2
    last_infos: list[dict[str, float]] | None = None
    for _ in range(eval_sim.max_cycles):
        feat = append_phase_features_torch(obs_rms_gpu.normalize(obs), phase_t)
        a_idx = torch.argmax(q_net(feat.reshape(-1, in_dim)), dim=-1)
        acts = action_table_t[a_idx].reshape(b_cnt, a_cnt, 2)
        obs, _, _, infos = eval_sim.step(acts)
        if infos is not None:
            last_infos = infos
    assert last_infos is not None
    team_arr = np.asarray([r["team_return"] for r in last_infos], dtype=np.float64)
    ag_arr = np.asarray([r["mean_agent_return"] for r in last_infos], dtype=np.float64)
    food_arr = np.asarray([r["food_events"] for r in last_infos], dtype=np.float64)
    pois_arr = np.asarray([r["poison_hits"] for r in last_infos], dtype=np.float64)
    return {
        "episodes": float(len(last_infos)),
        "mean_team_return": float(team_arr.mean()),
        "std_team_return": float(team_arr.std()),
        "mean_agent_return": float(ag_arr.mean()),
        "std_agent_return": float(ag_arr.std()),
        "mean_food_events": float(food_arr.mean()),
        "mean_poison_hits": float(pois_arr.mean()),
    }


def _run_dqn_gpu(
    scenario: WaterworldScenario,
    q_net: DuelingQNet,
    target_net: DuelingQNet,
    optimizer: torch.optim.Optimizer,
    q_params: list[torch.Tensor],
    t_params: list[torch.Tensor],
    dev: torch.device,
    action_table_t: torch.Tensor,
    total_episodes: int,
    num_envs: int,
    eval_interval_episodes: int,
    eval_episodes: int,
    seed: int,
    gamma: float,
    n_step: int,
    tau: float,
    batch_size: int,
    warmup_transitions: int,
    verbose: bool,
) -> tuple[
    int,
    int,
    dict[str, float],
    list[dict[str, float]],
    list[dict[str, float]],
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    n_actions = int(action_table_t.shape[0])
    n_agents = scenario.n_pursuers
    in_dim = scenario.obs_dim + 2
    phase_t = torch.from_numpy(build_agent_phase_vec(n_agents)).to(dev)
    obs_rms = RunningMeanStdGPU((scenario.obs_dim,), dev)

    buffer = ReplayBufferGPU(capacity=350_000, in_dim=in_dim, device=dev)
    train_sim = VectorizedWaterworldGPU(num_envs, scenario, dev, seed=seed * 1000 + 10)
    eval_sim = VectorizedWaterworldGPU(eval_episodes, scenario, dev, seed=900_000)

    cal_obs = train_sim.reset()
    obs_rms.update(cal_obs)
    cal_steps = min(60, max(5, scenario.max_cycles // 2))
    act_gen = torch.Generator(device=dev)
    act_gen.manual_seed(seed + 12345)
    for _ in range(cal_steps):
        rand_idx = torch.randint(0, n_actions, (num_envs, n_agents), generator=act_gen, device=dev)
        cal_obs, _, _, _ = train_sim.step(action_table_t[rand_idx])
        obs_rms.update(cal_obs)
    obs_rms_state = obs_rms.to_numpy_state()

    best_eval_team_ret = -float("inf")
    best_state_dict = {k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()}
    ep_curve: list[dict[str, float]] = []
    eval_curve: list[dict[str, float]] = []

    ev0 = _eval_dqn_gpu(eval_sim, q_net, obs_rms, action_table_t, phase_t)
    ev0["episode"] = 0.0
    eval_curve.append(ev0)

    eps_decay_episodes = max(int(total_episodes * 0.40), 1)
    episodes_done = 0
    next_eval_at = eval_interval_episodes
    train_steps = 0
    step_counter = 0
    nstep_q: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]] = deque(
        maxlen=n_step
    )
    obs = train_sim.reset()

    while episodes_done < total_episodes:
        progress = min(episodes_done / eps_decay_episodes, 1.0)
        eps = 1.0 - 0.95 * progress

        with torch.inference_mode():
            obs_feat = append_phase_features_torch(obs_rms.normalize(obs), phase_t)
            greedy_idx = torch.argmax(q_net(obs_feat.reshape(-1, in_dim)), dim=-1).reshape(
                num_envs, n_agents
            )
            rand_mask = torch.rand((num_envs, n_agents), generator=act_gen, device=dev) < eps
            rand_idx = torch.randint(
                0, n_actions, (num_envs, n_agents), generator=act_gen, device=dev
            )
            chosen_idx = torch.where(rand_mask, rand_idx, greedy_idx)
            acts = action_table_t[chosen_idx]

        next_obs, rewards, _, infos = train_sim.step(acts)
        done_flag = infos is not None
        next_feat = append_phase_features_torch(obs_rms.normalize(next_obs), phase_t)
        nstep_q.append((obs_feat, chosen_idx, rewards, next_feat, done_flag))

        if len(nstep_q) == n_step:
            b_o, b_a, b_r, b_no, b_bg = _fold_gpu_nstep(nstep_q, gamma)
            buffer.add_batch(b_o, b_a, b_r, b_no, b_bg)
            nstep_q.popleft()
        if done_flag:
            while nstep_q:
                b_o, b_a, b_r, b_no, b_bg = _fold_gpu_nstep(nstep_q, gamma)
                buffer.add_batch(b_o, b_a, b_r, b_no, b_bg)
                nstep_q.popleft()
            for info in infos:  # type: ignore[union-attr]
                if episodes_done < total_episodes:
                    episodes_done += 1
                    ep_curve.append(
                        {
                            "episode": float(episodes_done),
                            "team_return": float(info["team_return"]),
                            "mean_agent_return": float(info["mean_agent_return"]),
                            "food_events": float(info["food_events"]),
                            "poison_hits": float(info["poison_hits"]),
                        }
                    )

        obs = next_obs
        step_counter += 1

        if buffer.size >= warmup_transitions and step_counter % 8 == 0:
            n_grad = max(2, round((num_envs * n_agents * 8) / 240.0))
            bs_now = 1024 if num_envs >= 20 else batch_size
            for _ in range(n_grad):
                b_s, b_a, b_r, b_ns, b_bg = buffer.sample(bs_now)
                _dqn_update_step(
                    q_net, target_net, optimizer, q_params, t_params, b_s, b_a, b_r, b_ns, b_bg, tau
                )
                train_steps += 1

        if episodes_done >= next_eval_at or episodes_done >= total_episodes:
            ev = _eval_dqn_gpu(eval_sim, q_net, obs_rms, action_table_t, phase_t)
            ev["episode"] = float(episodes_done)
            eval_curve.append(ev)
            if ev["mean_team_return"] > best_eval_team_ret:
                best_eval_team_ret = ev["mean_team_return"]
                best_state_dict = {k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()}
            if verbose:
                print(
                    f"  [DQN:{scenario.key:11s}] ep={episodes_done:3d}/{total_episodes} | "
                    f"eps={eps:.2f} | eval_team={ev['mean_team_return']:+6.2f} "
                    f"(agent={ev['mean_agent_return']:+5.2f}, food={ev['mean_food_events']:4.1f}, "
                    f"poison={ev['mean_poison_hits']:4.1f})",
                    flush=True,
                )
            next_eval_at += eval_interval_episodes

    q_net.load_state_dict(best_state_dict)
    q_net.to(dev)
    final_eval = _eval_dqn_gpu(eval_sim, q_net, obs_rms, action_table_t, phase_t)
    return episodes_done, train_steps, final_eval, ep_curve, eval_curve, best_state_dict, obs_rms_state


def _run_qmix_gpu(
    scenario: WaterworldScenario,
    q_net: DuelingQNet,
    target_net: DuelingQNet,
    dev: torch.device,
    action_table_t: torch.Tensor,
    total_episodes: int,
    num_envs: int,
    eval_interval_episodes: int,
    eval_episodes: int,
    seed: int,
    gamma: float,
    n_step: int,
    tau: float,
    batch_size: int,
    warmup_transitions: int,
    verbose: bool,
    curriculum_frac: float = 0.20,
) -> tuple[
    int,
    int,
    dict[str, float],
    list[dict[str, float]],
    list[dict[str, float]],
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    """Cooperative Hybrid Local+QMIX + Correlated Exploration + n_coop Curriculum."""
    n_actions = int(action_table_t.shape[0])
    n_agents = scenario.n_pursuers
    in_dim = scenario.obs_dim + 2
    state_dim = 2 * scenario.obs_dim
    phase_t = torch.from_numpy(build_agent_phase_vec(n_agents)).to(dev)
    obs_rms = RunningMeanStdGPU((scenario.obs_dim,), dev)

    mixer = QMixerNet(n_agents=n_agents, state_dim=state_dim, embed_dim=16).to(dev)
    target_mixer = QMixerNet(n_agents=n_agents, state_dim=state_dim, embed_dim=16).to(dev)
    target_mixer.load_state_dict(mixer.state_dict())
    target_mixer.eval()

    optimizer = torch.optim.Adam(
        list(q_net.parameters()) + list(mixer.parameters()), lr=3e-4
    )
    online_params = [p.data for p in q_net.parameters()] + [p.data for p in mixer.parameters()]
    target_params = [p.data for p in target_net.parameters()] + [
        p.data for p in target_mixer.parameters()
    ]

    buffer = TeamReplayBufferGPU(capacity=75_000, n_agents=n_agents, in_dim=in_dim, device=dev)
    train_sim = VectorizedWaterworldGPU(num_envs, scenario, dev, seed=seed * 1000 + 10)
    eval_sim = VectorizedWaterworldGPU(eval_episodes, scenario, dev, seed=900_000)

    use_curriculum = scenario.n_coop > 1 and curriculum_frac > 0.0
    curriculum_end = int(total_episodes * curriculum_frac) if use_curriculum else 0
    if use_curriculum:
        train_sim.set_n_coop(1)
    curriculum_switched = not use_curriculum

    cal_obs = train_sim.reset()
    obs_rms.update(cal_obs)
    cal_steps = min(60, max(5, scenario.max_cycles // 2))
    act_gen = torch.Generator(device=dev)
    act_gen.manual_seed(seed + 12345)
    for _ in range(cal_steps):
        rand_idx = torch.randint(0, n_actions, (num_envs, n_agents), generator=act_gen, device=dev)
        cal_obs, _, _, _ = train_sim.step(action_table_t[rand_idx])
        obs_rms.update(cal_obs)
    obs_rms_state = obs_rms.to_numpy_state()

    best_eval_team_ret = -float("inf")
    best_state_dict = {k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()}
    ep_curve: list[dict[str, float]] = []
    eval_curve: list[dict[str, float]] = []

    ev0 = _eval_dqn_gpu(eval_sim, q_net, obs_rms, action_table_t, phase_t)
    ev0["episode"] = 0.0
    eval_curve.append(ev0)

    episodes_done = 0
    next_eval_at = eval_interval_episodes
    train_steps = 0
    step_counter = 0
    warmup_team = max(1_000, warmup_transitions // max(n_agents, 1))
    nstep_q: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]] = deque(
        maxlen=n_step
    )
    obs = train_sim.reset()

    while episodes_done < total_episodes:
        if not curriculum_switched and episodes_done >= curriculum_end:
            curriculum_switched = True
            train_sim.set_n_coop(scenario.n_coop)
            nstep_q.clear()
            obs = train_sim.reset()
            best_eval_team_ret = -float("inf")
            if verbose:
                print(
                    f"  [QMIX:{scenario.key:11s}] curriculum switch -> n_coop={scenario.n_coop} at ep={episodes_done}",
                    flush=True,
                )

        if not curriculum_switched:
            stage_prog = min(episodes_done / max(curriculum_end, 1), 1.0)
            eps = 1.0 - 0.80 * stage_prog
        else:
            rem_eps = max(total_episodes - curriculum_end, 1)
            stage_prog = min((episodes_done - curriculum_end) / max(int(rem_eps * 0.55), 1), 1.0)
            eps = (0.25 - 0.20 * stage_prog) if use_curriculum else (1.0 - 0.95 * stage_prog)

        with torch.inference_mode():
            obs_feat = append_phase_features_torch(obs_rms.normalize(obs), phase_t)
            greedy_idx = torch.argmax(q_net(obs_feat.reshape(-1, in_dim)), dim=-1).reshape(
                num_envs, n_agents
            )
            rand_mask = torch.rand((num_envs, n_agents), generator=act_gen, device=dev) < eps
            indep_rand = torch.randint(
                0, n_actions, (num_envs, n_agents), generator=act_gen, device=dev
            )
            # Correlated team exploration on 50% of envs so teammates drift cohesively onto food.
            shared_dir = torch.randint(
                0, n_actions, (num_envs, 1), generator=act_gen, device=dev
            ).expand(num_envs, n_agents)
            corr_env = (torch.rand((num_envs, 1), generator=act_gen, device=dev) < 0.5).expand(
                num_envs, n_agents
            )
            rand_idx = torch.where(corr_env, shared_dir, indep_rand)
            chosen_idx = torch.where(rand_mask, rand_idx, greedy_idx)
            acts = action_table_t[chosen_idx]

        next_obs, rewards, _, infos = train_sim.step(acts)
        done_flag = infos is not None
        next_feat = append_phase_features_torch(obs_rms.normalize(next_obs), phase_t)
        nstep_q.append((obs_feat, chosen_idx, rewards, next_feat, done_flag))

        if len(nstep_q) == n_step:
            b_o, b_a, b_ra, b_rt, b_no, b_bg = _fold_gpu_nstep_team(nstep_q, gamma)
            buffer.add_batch(b_o, b_a, b_ra, b_rt, b_no, b_bg)
            nstep_q.popleft()
        if done_flag:
            while nstep_q:
                b_o, b_a, b_ra, b_rt, b_no, b_bg = _fold_gpu_nstep_team(nstep_q, gamma)
                buffer.add_batch(b_o, b_a, b_ra, b_rt, b_no, b_bg)
                nstep_q.popleft()
            for info in infos:  # type: ignore[union-attr]
                if episodes_done < total_episodes:
                    episodes_done += 1
                    ep_curve.append(
                        {
                            "episode": float(episodes_done),
                            "team_return": float(info["team_return"]),
                            "mean_agent_return": float(info["mean_agent_return"]),
                            "food_events": float(info["food_events"]),
                            "poison_hits": float(info["poison_hits"]),
                        }
                    )

        obs = next_obs
        step_counter += 1

        if buffer.size >= warmup_team and step_counter % 8 == 0:
            n_grad = max(2, round((num_envs * 8) / 48.0))
            bs_now = min(256, max(64, batch_size // max(n_agents, 1) * 2))
            for _ in range(n_grad):
                b_s, b_a, b_ra, b_rt, b_ns, b_bg = buffer.sample(bs_now)
                _qmix_update_step(
                    q_net,
                    target_net,
                    mixer,
                    target_mixer,
                    optimizer,
                    online_params,
                    target_params,
                    b_s,
                    b_a,
                    b_ra,
                    b_rt,
                    b_ns,
                    b_bg,
                    tau,
                )
                train_steps += 1

        if episodes_done >= next_eval_at or episodes_done >= total_episodes:
            ev = _eval_dqn_gpu(eval_sim, q_net, obs_rms, action_table_t, phase_t)
            ev["episode"] = float(episodes_done)
            eval_curve.append(ev)
            if ev["mean_team_return"] > best_eval_team_ret:
                best_eval_team_ret = ev["mean_team_return"]
                best_state_dict = {k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()}
            if verbose:
                print(
                    f"  [QMIX:{scenario.key:11s}] ep={episodes_done:3d}/{total_episodes} | "
                    f"eps={eps:.2f} | eval_team={ev['mean_team_return']:+6.2f} "
                    f"(agent={ev['mean_agent_return']:+5.2f}, food={ev['mean_food_events']:4.1f}, "
                    f"poison={ev['mean_poison_hits']:4.1f})",
                    flush=True,
                )
            next_eval_at += eval_interval_episodes

    q_net.load_state_dict(best_state_dict)
    q_net.to(dev)
    final_eval = _eval_dqn_gpu(eval_sim, q_net, obs_rms, action_table_t, phase_t)
    return episodes_done, train_steps, final_eval, ep_curve, eval_curve, best_state_dict, obs_rms_state


def train_double_dqn(
    scenario: WaterworldScenario,
    total_episodes: int = 600,
    num_envs: int = 12,
    eval_interval_episodes: int = 100,
    eval_episodes: int = 8,
    seed: int = 42,
    device: str = "auto",
    verbose: bool = True,
    warmup_transitions: int = 5_000,
    use_gpu_sim: bool = True,
    arch: str = "transformer",
    coop_mode: bool = False,
) -> dict[str, Any]:
    """Train 5-step Dueling Double DQN with discretized thrust actions on WaterWorld."""
    t_start = time.perf_counter()
    dev = resolve_device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    action_table = build_discrete_thrust_table()
    n_actions = len(action_table)
    n_agents = scenario.n_pursuers
    in_dim = scenario.obs_dim + 2
    phase = build_agent_phase_vec(n_agents)

    q_net = DuelingQNet(
        in_dim=in_dim,
        n_actions=n_actions,
        hidden_dim=128,
        arch=arch,
        n_sensors=scenario.n_sensors,
        speed_features=scenario.speed_features,
    ).to(dev)
    target_net = DuelingQNet(
        in_dim=in_dim,
        n_actions=n_actions,
        hidden_dim=128,
        arch=arch,
        n_sensors=scenario.n_sensors,
        speed_features=scenario.speed_features,
    ).to(dev)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=3e-4)
    gamma = 0.99
    n_step = 5
    tau = 0.015
    batch_size = 512
    q_params = [p.data for p in q_net.parameters()]
    t_params = [p.data for p in target_net.parameters()]

    if use_gpu_sim and dev.type == "cuda":
        action_table_t = torch.from_numpy(action_table).to(dev)
        if coop_mode:
            (
                episodes_done,
                train_steps,
                final_eval,
                ep_curve,
                eval_curve,
                best_state_dict,
                obs_rms_state,
            ) = _run_qmix_gpu(
                scenario=scenario,
                q_net=q_net,
                target_net=target_net,
                dev=dev,
                action_table_t=action_table_t,
                total_episodes=total_episodes,
                num_envs=num_envs,
                eval_interval_episodes=eval_interval_episodes,
                eval_episodes=eval_episodes,
                seed=seed,
                gamma=gamma,
                n_step=n_step,
                tau=tau,
                batch_size=batch_size,
                warmup_transitions=warmup_transitions,
                verbose=verbose,
            )
        else:
            (
                episodes_done,
                train_steps,
                final_eval,
                ep_curve,
                eval_curve,
                best_state_dict,
                obs_rms_state,
            ) = _run_dqn_gpu(
                scenario=scenario,
                q_net=q_net,
                target_net=target_net,
                optimizer=optimizer,
                q_params=q_params,
                t_params=t_params,
                dev=dev,
                action_table_t=action_table_t,
                total_episodes=total_episodes,
                num_envs=num_envs,
                eval_interval_episodes=eval_interval_episodes,
                eval_episodes=eval_episodes,
                seed=seed,
                gamma=gamma,
                n_step=n_step,
                tau=tau,
                batch_size=batch_size,
                warmup_transitions=warmup_transitions,
                verbose=verbose,
            )
        return {
            "algo": "QMIX" if coop_mode else "DQN",
            "scenario": scenario.key,
            "elapsed_sec": float(time.perf_counter() - t_start),
            "total_episodes": int(episodes_done),
            "train_updates": int(train_steps),
            "best_eval": final_eval,
            "episode_curve": ep_curve,
            "eval_curve": eval_curve,
            "state_dict": best_state_dict,
            "obs_rms_state": obs_rms_state,
        }

    buffer = ReplayBuffer(capacity=350_000, in_dim=in_dim)
    train_steps = 0
    eps_decay_episodes = max(int(total_episodes * 0.40), 1)
    eval_seeds = [900_000 + i for i in range(eval_episodes)]

    best_eval_team_ret = -float("inf")
    best_state_dict = {k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()}
    ep_curve: list[dict[str, float]] = []
    eval_curve: list[dict[str, float]] = []
    obs_rms = RunningMeanStd((scenario.obs_dim,))

    def make_eval_fn() -> Any:
        def _act(obs_batch: np.ndarray, ph: np.ndarray) -> np.ndarray:
            norm = obs_rms.normalize(obs_batch)
            feat = append_phase_features(norm, ph)
            flat = feat.reshape(-1, in_dim)
            with torch.inference_mode():
                q = q_net(torch.from_numpy(flat).to(dev))
                a_idx = torch.argmax(q, dim=-1).cpu().numpy()
            acts = action_table[a_idx]
            return acts.reshape(*obs_batch.shape[:-1], 2)

        return _act

    episodes_done = 0
    next_eval_at = eval_interval_episodes
    nstep_queues: list[deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]]] = [
        deque(maxlen=n_step) for _ in range(num_envs)
    ]

    with SubprocWaterworldVecEnv(scenario, num_envs=num_envs, base_seed=seed * 1000 + 10) as train_env, \
         SubprocWaterworldVecEnv(scenario, num_envs=min(4, num_envs, eval_episodes), base_seed=910_000) as eval_env:

        # Pre-calibrate static observation normalizer over random exploration steps, then freeze it
        # so the off-policy ReplayBuffer always holds stationary normalized features.
        cal_obs = train_env.reset()
        obs_rms.update(cal_obs)
        cal_steps = min(60, max(5, scenario.max_cycles // 2))
        for _ in range(cal_steps):
            rand_acts = action_table[rng.integers(0, n_actions, size=(num_envs, n_agents))]
            cal_obs, _, _, _ = train_env.step(rand_acts)
            obs_rms.update(cal_obs)
        obs_rms_state = {
            "mean": obs_rms.mean.copy(),
            "var": obs_rms.var.copy(),
            "count": float(obs_rms.count),
        }

        ev0 = evaluate_policy_parallel(eval_env, make_eval_fn(), eval_seeds)
        ev0["episode"] = 0.0
        eval_curve.append(ev0)

        obs = train_env.reset()
        step_counter = 0

        while episodes_done < total_episodes:
            progress = min(episodes_done / eps_decay_episodes, 1.0)
            eps = 1.0 - 0.95 * progress

            obs_feat = append_phase_features(obs_rms.normalize(obs), phase)  # (E, A, in_dim)
            flat_feat = obs_feat.reshape(-1, in_dim)

            with torch.inference_mode():
                q_vals = q_net(torch.from_numpy(flat_feat).to(dev))
                greedy_idx = torch.argmax(q_vals, dim=-1).cpu().numpy()

            rand_mask = rng.random(len(greedy_idx)) < eps
            rand_idx = rng.integers(0, n_actions, size=len(greedy_idx))
            chosen_idx = np.where(rand_mask, rand_idx, greedy_idx).reshape(num_envs, n_agents)
            acts = action_table[chosen_idx]

            next_obs, rewards, dones, infos = train_env.step(acts)
            next_feat = append_phase_features(obs_rms.normalize(next_obs), phase)  # (E, A, in_dim)

            push_obs = []
            push_act = []
            push_ret = []
            push_next = []
            push_boot = []
            for e in range(num_envs):
                done_e = bool(dones[e])
                nstep_queues[e].append(
                    (obs_feat[e], chosen_idx[e], rewards[e], next_feat[e], done_e)
                )
                if len(nstep_queues[e]) == n_step:
                    o0, a0, r_n, nk, bg = _emit_nstep_head(nstep_queues[e], gamma)
                    push_obs.append(o0)
                    push_act.append(a0)
                    push_ret.append(r_n)
                    push_next.append(nk)
                    push_boot.append(bg)
                    nstep_queues[e].popleft()
                if done_e:
                    while nstep_queues[e]:
                        o0, a0, r_n, nk, bg = _emit_nstep_head(nstep_queues[e], gamma)
                        push_obs.append(o0)
                        push_act.append(a0)
                        push_ret.append(r_n)
                        push_next.append(nk)
                        push_boot.append(bg)
                        nstep_queues[e].popleft()

            if push_obs:
                buffer.add_batch(
                    np.concatenate(push_obs, axis=0),
                    np.concatenate(push_act, axis=0),
                    np.concatenate(push_ret, axis=0),
                    np.concatenate(push_next, axis=0),
                    np.concatenate(push_boot, axis=0),
                )

            obs = next_obs
            step_counter += 1

            for info in infos:
                if info is not None and episodes_done < total_episodes:
                    episodes_done += 1
                    ep_curve.append(
                        {
                            "episode": float(episodes_done),
                            "team_return": float(info["team_return"]),
                            "mean_agent_return": float(info["mean_agent_return"]),
                            "food_events": float(info["food_events"]),
                            "poison_hits": float(info["poison_hits"]),
                        }
                    )

            if buffer.size >= warmup_transitions and step_counter % 4 == 0:
                for _ in range(2):
                    b_s, b_a, b_r, b_ns, b_bg = buffer.sample(batch_size, rng, dev)
                    _dqn_update_step(
                        q_net,
                        target_net,
                        optimizer,
                        q_params,
                        t_params,
                        b_s,
                        b_a,
                        b_r,
                        b_ns,
                        b_bg,
                        tau,
                    )
                    train_steps += 1

            if episodes_done >= next_eval_at or episodes_done >= total_episodes:
                ev = evaluate_policy_parallel(eval_env, make_eval_fn(), eval_seeds)
                ev["episode"] = float(episodes_done)
                eval_curve.append(ev)
                if ev["mean_team_return"] > best_eval_team_ret:
                    best_eval_team_ret = ev["mean_team_return"]
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()}
                if verbose:
                    print(
                        f"  [DQN:{scenario.key:11s}] ep={episodes_done:3d}/{total_episodes} | "
                        f"eps={eps:.2f} | eval_team={ev['mean_team_return']:+6.2f} "
                        f"(agent={ev['mean_agent_return']:+5.2f}, food={ev['mean_food_events']:4.1f}, "
                        f"poison={ev['mean_poison_hits']:4.1f})",
                        flush=True,
                    )
                next_eval_at += eval_interval_episodes

        q_net.load_state_dict(best_state_dict)
        q_net.to(dev)
        final_eval = evaluate_policy_parallel(eval_env, make_eval_fn(), eval_seeds)

    return {
        "algo": "DQN",
        "scenario": scenario.key,
        "elapsed_sec": float(time.perf_counter() - t_start),
        "total_episodes": int(episodes_done),
        "train_updates": int(train_steps),
        "best_eval": final_eval,
        "episode_curve": ep_curve,
        "eval_curve": eval_curve,
        "state_dict": best_state_dict,
        "obs_rms_state": obs_rms_state,
    }
