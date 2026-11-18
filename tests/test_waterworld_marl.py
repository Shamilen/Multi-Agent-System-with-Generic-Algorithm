"""Unit and smoke tests for WaterWorld MARL environments and GA/DQN/PPO trainers."""

from __future__ import annotations

import numpy as np
import torch

from waterworld_marl.algos.dqn import build_dqn_action_fn, train_double_dqn
from waterworld_marl.algos.ga import WaterworldGAPolicy, build_ga_action_fn, train_ga
from waterworld_marl.algos.ga_worker import build_sensor_basis, numpy_equivariant_forward
from waterworld_marl.algos.ppo import build_ppo_action_fn, train_ppo
from waterworld_marl.scenarios import (
    SCENARIOS,
    SubprocWaterworldVecEnv,
    WaterworldScenario,
    build_agent_phase_vec,
    evaluate_policy_parallel,
    make_waterworld_env,
)


def test_all_five_scenarios_reset_and_step() -> None:
    expected_dims = {
        "base": (5, 242),
        "single": (1, 242),
        "no_speed": (5, 152),
        "few_sensors": (5, 82),
        "more_poison": (5, 242),
    }
    assert set(SCENARIOS.keys()) == set(expected_dims.keys())
    for key, sc in SCENARIOS.items():
        exp_agents, exp_obs = expected_dims[key]
        assert sc.n_pursuers == exp_agents
        assert sc.obs_dim == exp_obs
        env = make_waterworld_env(sc)
        try:
            obs, _ = env.reset(seed=11)
            assert len(obs) == exp_agents
            for a in env.possible_agents:
                assert obs[a].shape == (exp_obs,)
                assert np.all(np.isfinite(obs[a]))
            for _ in range(4):
                acts = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
                obs, rews, terms, truncs, _ = env.step(acts)
                assert len(rews) == exp_agents
        finally:
            env.close()


def test_subproc_vec_env_auto_reset_and_eval() -> None:
    tiny = WaterworldScenario(
        key="tiny",
        title_ru="Tiny",
        n_pursuers=2,
        n_evaders=3,
        n_poisons=3,
        n_coop=1,
        n_sensors=10,
        speed_features=True,
        max_cycles=15,
    )
    with SubprocWaterworldVecEnv(tiny, num_envs=2, base_seed=50) as vec:
        obs = vec.reset()
        assert obs.shape == (2, 2, tiny.obs_dim)
        completed = 0
        for _ in range(18):
            acts = np.zeros((2, 2, 2), dtype=np.float32)
            obs, rews, dones, infos = vec.step(acts)
            assert obs.shape == (2, 2, tiny.obs_dim)
            assert rews.shape == (2, 2)
            assert dones.shape == (2,)
            completed += sum(1 for info in infos if info is not None)
        assert completed == 2

        res = evaluate_policy_parallel(
            vec,
            lambda o, p: np.zeros((*o.shape[:-1], 2), dtype=np.float32),
            seeds=[100, 101, 102],
        )
        assert res["episodes"] == 3.0
        assert np.isfinite(res["mean_team_return"])


def test_smoke_dqn_ppo_ga_tiny_scenario() -> None:
    tiny = WaterworldScenario(
        key="tiny_smoke",
        title_ru="TinySmoke",
        n_pursuers=2,
        n_evaders=3,
        n_poisons=3,
        n_coop=1,
        n_sensors=10,
        speed_features=True,
        max_cycles=30,
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    phase = build_agent_phase_vec(tiny.n_pursuers)
    dummy_obs = np.zeros((2, tiny.obs_dim), dtype=np.float32)

    dqn_res = train_double_dqn(
        tiny,
        total_episodes=4,
        num_envs=2,
        eval_interval_episodes=4,
        eval_episodes=2,
        seed=1,
        device=dev,
        verbose=False,
        warmup_transitions=32,
    )
    assert dqn_res["total_episodes"] == 4
    assert dqn_res["train_updates"] > 0
    assert np.isfinite(dqn_res["best_eval"]["mean_team_return"])
    dqn_act = build_dqn_action_fn(
        tiny, dqn_res["state_dict"], obs_rms_state=dqn_res["obs_rms_state"], device=dev
    )(dummy_obs, phase)
    assert dqn_act.shape == (2, 2)
    assert np.all(np.abs(dqn_act) <= 1.0 + 1e-6)

    ppo_res = train_ppo(
        tiny,
        total_episodes=4,
        num_envs=2,
        eval_interval_episodes=4,
        eval_episodes=2,
        seed=1,
        device=dev,
        verbose=False,
        arch="transformer",
    )
    assert ppo_res["total_episodes"] == 4
    assert ppo_res["train_updates"] > 0
    assert any("ray_proj" in k for k in ppo_res["state_dict"])
    assert np.isfinite(ppo_res["best_eval"]["mean_team_return"])
    ppo_act = build_ppo_action_fn(
        tiny, ppo_res["state_dict"], ppo_res["obs_rms_state"], device=dev
    )(dummy_obs, phase)
    assert ppo_act.shape == (2, 2)
    assert np.all(np.abs(ppo_act) <= 1.0 + 1e-6)

    ga_res = train_ga(
        tiny,
        total_episodes=6,
        pop_size=3,
        num_workers=2,
        eval_interval_episodes=6,
        eval_episodes=2,
        seed=1,
        device=dev,
        verbose=False,
    )
    assert ga_res["total_episodes"] == 6
    assert ga_res["train_updates"] >= 1
    assert np.isfinite(ga_res["best_eval"]["mean_team_return"])
    ga_act = build_ga_action_fn(tiny, ga_res["state_dict"], device=dev)(dummy_obs, phase)
    assert ga_act.shape == (2, 2)
    assert np.all(np.abs(ga_act) <= 1.0 + 1e-6)


def test_equivariant_ga_numpy_matches_torch() -> None:
    rng = np.random.default_rng(7)
    for speed in (True, False):
        n_sensors = 12
        feats = 8 if speed else 5
        obs_dim = n_sensors * feats + 2
        model = WaterworldGAPolicy(n_sensors=n_sensors, speed_features=speed).eval()
        obs = rng.uniform(0.0, 1.0, size=(5, obs_dim)).astype(np.float32)
        phase = build_agent_phase_vec(5)
        with torch.no_grad():
            t_out = model(torch.from_numpy(obs), torch.from_numpy(phase)).numpy()
        layers = tuple(p.detach().numpy().astype(np.float32) for p in model.parameters())
        np_out = numpy_equivariant_forward(
            obs, phase, layers, build_sensor_basis(n_sensors), n_sensors, feats
        )
        assert np.allclose(t_out, np_out, atol=1e-5)

        # Verify true SO(2) directional alignment (ray pi/2 -> +y thrust for positive radial weight)
        basis = build_sensor_basis(n_sensors)
        assert np.allclose(basis[n_sensors // 4, 0], [0.0, 1.0], atol=1e-5)
        model_dir = WaterworldGAPolicy(n_sensors=n_sensors, speed_features=speed).eval()
        with torch.no_grad():
            model_dir.ray_vec.weight.zero_()
            model_dir.ray_vec.weight[0, 2] = 2.0
        o_ray = np.zeros((1, obs_dim), dtype=np.float32)
        for dist_idx in (0, 1, 2, 4, 6) if speed else (0, 1, 2, 3, 4):
            o_ray[0, dist_idx * n_sensors : (dist_idx + 1) * n_sensors] = 1.0
        o_ray[0, 2 * n_sensors + n_sensors // 4] = 0.2
        with torch.no_grad():
            act_dir = model_dir(torch.from_numpy(o_ray)).numpy()[0]
        assert abs(float(act_dir[0])) < 1e-5 and float(act_dir[1]) > 0.15
