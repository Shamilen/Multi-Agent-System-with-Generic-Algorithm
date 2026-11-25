"""Differential equivalence tests: VectorizedPursuitGPU & BatchedPopulationPolicyGPU vs PettingZoo."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pursuit_ga.config import EnvConfig, ModelConfig
from pursuit_ga.env_utils import get_evaders_remaining, make_pursuit_env, preprocess_observations
from pursuit_ga.gpu_sim import BatchedPopulationPolicyGPU, VectorizedPursuitGPU
from pursuit_ga.networks import PursuitPolicyNet, build_agent_phase_features


@pytest.mark.parametrize("shared_reward", [True, False])
def test_gpu_simulator_matches_pettingzoo_step_by_step(shared_reward: bool) -> None:
    """Verify GPU simulator matches real PettingZoo sisl/pursuit-v5 step-by-step."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EnvConfig(max_cycles=60, shared_reward=shared_reward, freeze_evaders=True)
    pz_env = make_pursuit_env(cfg)
    try:
        obs_dict, _ = pz_env.reset(seed=42)
        agents = list(pz_env.possible_agents)

        raw_pz = pz_env.unwrapped.env
        px = np.array([raw_pz.pursuer_layer.get_position(i)[0] for i in range(cfg.n_pursuers)], dtype=np.int64)
        py = np.array([raw_pz.pursuer_layer.get_position(i)[1] for i in range(cfg.n_pursuers)], dtype=np.int64)
        ex = np.array([raw_pz.evader_layer.get_position(i)[0] for i in range(cfg.n_evaders)], dtype=np.int64)
        ey = np.array([raw_pz.evader_layer.get_position(i)[1] for i in range(cfg.n_evaders)], dtype=np.int64)

        gpu_sim = VectorizedPursuitGPU(batch_size=1, env_cfg=cfg, device=device)
        gpu_sim.set_state(
            torch.from_numpy(px).unsqueeze(0).to(device),
            torch.from_numpy(py).unsqueeze(0).to(device),
            torch.from_numpy(ex).unsqueeze(0).to(device),
            torch.from_numpy(ey).unsqueeze(0).to(device),
        )

        # Check initial raw observations and preprocessed observations
        pz_obs_np = np.stack([obs_dict[ag] for ag in agents], axis=0)
        gpu_raw = gpu_sim.get_raw_observations()[0].cpu().numpy()
        assert np.allclose(gpu_raw, pz_obs_np, atol=1e-6)

        pz_prep = preprocess_observations(obs_dict, agents)
        gpu_prep = gpu_sim.get_preprocessed_observations()[0].cpu().numpy()
        assert np.allclose(gpu_prep, pz_prep, atol=1e-6)

        rng = np.random.default_rng(777)
        for step_idx in range(50):
            acts_np = rng.integers(0, 5, size=cfg.n_pursuers)
            act_dict = {ag: int(acts_np[i]) for i, ag in enumerate(agents)}

            obs_dict, rew_dict, term_dict, trunc_dict, _ = pz_env.step(act_dict)
            gpu_rew, _ = gpu_sim.step(torch.from_numpy(acts_np).unsqueeze(0).to(device))

            pz_rew_np = np.array([rew_dict[ag] for ag in agents], dtype=np.float32)
            assert np.allclose(
                gpu_rew[0].cpu().numpy(), pz_rew_np, atol=1e-5
            ), f"Reward mismatch at step {step_idx}"

            assert int(gpu_sim.ev_alive[0].sum().item()) == get_evaders_remaining(pz_env)
            pz_done = (not pz_env.agents) or any(term_dict.values()) or any(trunc_dict.values())
            assert bool(gpu_sim.done[0].item()) == pz_done

            if pz_done:
                break

            pz_obs_np = np.stack([obs_dict[ag] for ag in agents], axis=0)
            gpu_raw = gpu_sim.get_raw_observations()[0].cpu().numpy()
            assert np.allclose(
                gpu_raw, pz_obs_np, atol=1e-6
            ), f"Observation mismatch at step {step_idx}"
    finally:
        pz_env.close()


@pytest.mark.parametrize(
    "arch",
    ["cnn", "cnn_gru", "eq_cnn_gru", "transformer", "eq_transformer", "eq_comm_transformer"],
)
def test_batched_population_policy_matches_single_policy(arch: str) -> None:
    """Verify BatchedPopulationPolicyGPU logits & hidden states match PursuitPolicyNet."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = EnvConfig()
    model_cfg = ModelConfig(arch=arch)

    pop_size = 5
    nets = [PursuitPolicyNet(model_cfg, obs_range=7, n_actions=5).to(device) for _ in range(pop_size)]
    genomes = np.stack([n.get_genome() for n in nets], axis=0)

    batched = BatchedPopulationPolicyGPU(genomes, model_cfg, env_cfg, device=device)

    B_per_g = 16
    obs = torch.randn(pop_size, B_per_g, 3, 7, 7, device=device)
    prev = torch.zeros(pop_size, B_per_g, 5, device=device)
    prev[:, :, 2] = 1.0
    phase = torch.from_numpy(build_agent_phase_features(8)).to(device).repeat(2, 1).unsqueeze(0).expand(pop_size, -1, -1)
    hidden = (
        torch.zeros(pop_size, B_per_g, model_cfg.gru_dim, device=device)
        if arch in ("cnn_gru", "eq_cnn_gru", "transformer", "eq_transformer", "eq_comm_transformer")
        else None
    )

    b_logits, b_hidden = batched.forward(obs, prev, phase, hidden, team_size=8)

    for p_idx, net in enumerate(nets):
        h_single = hidden[p_idx] if hidden is not None else None
        s_logits, s_hidden = net(obs[p_idx], prev[p_idx], phase[p_idx], h_single, team_size=8)
        assert torch.allclose(b_logits[p_idx], s_logits, atol=1e-3)
        if b_hidden is not None and s_hidden is not None:
            assert torch.allclose(b_hidden[p_idx], s_hidden, atol=1e-3)
