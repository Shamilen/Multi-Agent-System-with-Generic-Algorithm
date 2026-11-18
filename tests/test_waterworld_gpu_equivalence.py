"""Equivalence tests between vectorized NumPy/CUDA WaterWorld and reference PettingZoo pymunk."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from waterworld_marl.env.gpu_waterworld import VectorizedWaterworldGPU
from waterworld_marl.scenarios import SCENARIOS, make_waterworld_env


def test_numpy_vectorized_sensors_match_reference_loop() -> None:
    for sc in SCENARIOS.values():
        env = make_waterworld_env(sc)
        try:
            env.reset(seed=17)
            base = env.unwrapped.env
            obs_vec = base.observe_list()
            s_cnt = base.n_sensors
            for i, p in enumerate(base.pursuers):
                ev_d_list, ev_v_list = zip(
                    *[
                        p.get_sensor_reading(e.body.position, e.radius, e.body.velocity, base.evader_speed)
                        for e in base.evaders
                    ]
                )
                ref_ed, ref_ev = base.get_sensor_readings(
                    list(ev_d_list), p.sensor_range, list(ev_v_list)
                )
                ref_bar = p.get_sensor_barrier_readings()
                assert np.allclose(obs_vec[i][s_cnt : 2 * s_cnt], ref_bar, atol=1e-12)
                assert np.allclose(obs_vec[i][2 * s_cnt : 3 * s_cnt], ref_ed, atol=1e-12)
                if sc.speed_features:
                    assert np.allclose(obs_vec[i][3 * s_cnt : 4 * s_cnt], ref_ev, atol=1e-12)
        finally:
            env.close()


@pytest.mark.parametrize("sc_key", ["base", "single", "no_speed", "few_sensors", "more_poison"])
def test_gpu_simulator_observations_and_step_match_pettingzoo(sc_key: str) -> None:
    sc = SCENARIOS[sc_key]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_waterworld_env(sc)
    try:
        obs_dict, _ = env.reset(seed=123)
        base = env.unwrapped.env
        sim = VectorizedWaterworldGPU(1, sc, dev, use_cuda_graph=False)
        gpu_obs0 = sim.load_from_waterworld_bases([base])[0].cpu().numpy()
        cpu_obs0 = np.stack([obs_dict[a] for a in env.possible_agents], axis=0).astype(np.float32)
        assert np.allclose(gpu_obs0, cpu_obs0, atol=1e-5)

        rng = np.random.default_rng(77)
        for _ in range(5):
            act = rng.uniform(-1.0, 1.0, size=(sc.n_pursuers, 2)).astype(np.float32)
            obs_dict, rew_dict, _, _, _ = env.step(
                {a: act[i] for i, a in enumerate(env.possible_agents)}
            )
            cpu_rew = np.array([float(rew_dict[a]) for a in env.possible_agents], dtype=np.float32)
            gpu_obs_t, gpu_rew_t, _, _ = sim.step(torch.from_numpy(act[None]).to(dev))
            assert np.allclose(gpu_rew_t[0].cpu().numpy(), cpu_rew, atol=1e-4)
            cpu_p_pos = np.asarray([p.body.position for p in base.pursuers], dtype=np.float32)
            assert np.allclose(sim.p_pos[0].cpu().numpy(), cpu_p_pos, atol=1e-2)
            assert np.allclose(
                gpu_obs_t[0].cpu().numpy(),
                np.stack([obs_dict[a] for a in env.possible_agents], axis=0).astype(np.float32),
                atol=1e-3,
            )
    finally:
        env.close()


def test_gpu_simulator_capture_callbacks_and_cuda_graph_parity() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Single-agent begin (+10.01) then separate (+10.00) parity
    env_s = make_waterworld_env(SCENARIOS["single"])
    try:
        env_s.reset(seed=0)
        base_s = env_s.unwrapped.env
        base_s.evaders[0].body.position = (300.0, 300.0)
        base_s.evaders[0].body.velocity = (0.0, 0.0)
        base_s.pursuers[0].body.position = (300.0, 300.0)
        base_s.pursuers[0].body.velocity = (0.0, 0.0)

        sim_s = VectorizedWaterworldGPU(1, SCENARIOS["single"], dev, use_cuda_graph=False)
        sim_s.load_from_waterworld_bases([base_s])
        z_cpu = {"pursuer_0": np.zeros(2, dtype=np.float32)}
        z_gpu = torch.zeros((1, 1, 2), dtype=torch.float32, device=dev)

        _, r0_c, *_ = env_s.step(z_cpu)
        _, r0_g, _, _ = sim_s.step(z_gpu)
        assert abs(float(r0_g[0, 0].item()) - float(r0_c["pursuer_0"])) < 1e-4

        # Separate both cleanly outside capture radius
        base_s.pursuers[0].body.position = (100.0, 100.0)
        sim_s.p_pos[0, 0] = torch.tensor([100.0, 100.0], device=dev)
        _, r1_c, *_ = env_s.step(z_cpu)
        _, r1_g, _, _ = sim_s.step(z_gpu)
        assert abs(float(r1_g[0, 0].item()) - float(r1_c["pursuer_0"])) < 1e-4
    finally:
        env_s.close()

    if dev.type == "cuda":
        sc = SCENARIOS["base"]
        out_eager, out_graph = [], []
        for use_cg, bucket in ((False, out_eager), (True, out_graph)):
            sim = VectorizedWaterworldGPU(4, sc, dev, seed=11, use_cuda_graph=use_cg)
            sim.reset(seed=11)
            gen = torch.Generator(device=dev)
            gen.manual_seed(99)
            for _ in range(sc.max_cycles + 10):
                act = torch.rand((4, sc.n_pursuers, 2), generator=gen, device=dev) * 2.0 - 1.0
                obs, rew, _, infos = sim.step(act)
                if infos is not None:
                    bucket.append(infos)
            bucket.append((obs.cpu().numpy(), rew.cpu().numpy()))
        assert np.allclose(out_eager[-1][0], out_graph[-1][0], atol=1e-5)
        assert np.allclose(out_eager[-1][1], out_graph[-1][1], atol=1e-5)
        assert len(out_eager[0]) == len(out_graph[0]) == 4
