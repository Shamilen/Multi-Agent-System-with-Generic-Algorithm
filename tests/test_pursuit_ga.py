"""Tests for PettingZoo Pursuit Deep MARL Genetic Algorithm components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pursuit_ga.config import EnvConfig, GAConfig, ModelConfig, TrainConfig
from pursuit_ga.env_utils import make_pursuit_env, preprocess_observations
from pursuit_ga.evaluate import load_checkpoint
from pursuit_ga.genetic_algorithm import GeneticAlgorithm
from pursuit_ga.networks import PursuitPolicyNet, build_agent_phase_features
from pursuit_ga.train import train


def test_env_creation_and_preprocessing() -> None:
    cfg = EnvConfig(max_cycles=10)
    env = make_pursuit_env(cfg)
    try:
        obs_dict, _ = env.reset(seed=123)
        agents = list(env.possible_agents)
        assert len(agents) == cfg.n_pursuers
        assert obs_dict[agents[0]].shape == (cfg.obs_range, cfg.obs_range, 3)

        batch = preprocess_observations(obs_dict, agents)
        assert batch.shape == (cfg.n_pursuers, 3, cfg.obs_range, cfg.obs_range)
        assert batch.dtype == np.float32
        assert np.all(batch >= 0.0) and np.all(batch <= 1.0)
        # Center of channel 1 removes self count (1.0) unless another pursuer overlaps
        assert np.all(batch[:, 1, 3, 3] <= 0.99)
    finally:
        env.close()


@pytest.mark.parametrize(
    "arch",
    ["cnn", "cnn_gru", "eq_cnn_gru", "transformer", "eq_transformer", "eq_comm_transformer"],
)
def test_policy_genome_roundtrip_and_forward(arch: str) -> None:
    model_cfg = ModelConfig(arch=arch)
    net = PursuitPolicyNet(model_cfg=model_cfg, obs_range=7, n_actions=5)
    genome = net.get_genome()
    assert genome.shape == (net.num_params,)
    assert np.all(np.isfinite(genome))

    modified = genome + 0.05
    net.set_genome(modified)
    recovered = net.get_genome()
    assert np.allclose(recovered, modified, atol=1e-6)

    obs = torch.zeros(8, 3, 7, 7)
    prev = torch.zeros(8, 5)
    phase = torch.from_numpy(build_agent_phase_features(8))
    hidden = net.init_hidden(8)

    logits, next_hidden = net(obs, prev, phase, hidden)
    assert logits.shape == (8, 5)
    if arch in ("cnn_gru", "eq_cnn_gru", "transformer", "eq_transformer", "eq_comm_transformer"):
        assert next_hidden is not None
        assert next_hidden.shape == (8, model_cfg.gru_dim)
    else:
        assert next_hidden is None


@pytest.mark.parametrize("arch", ["eq_cnn_gru", "eq_transformer", "eq_comm_transformer"])
def test_eq_cnn_gru_exact_c4_equivariance(arch: str) -> None:
    """Rotating spatial observation, prev_action, and agent_phase permutes action logits exactly."""
    torch.manual_seed(123)
    net = PursuitPolicyNet(
        ModelConfig(arch=arch, conv_channels=(8, 12, 8), hidden_dim=24, gru_dim=20),
        obs_range=7,
        n_actions=5,
    )
    obs = torch.randn(4, 3, 7, 7)
    prev = torch.eye(5)[[0, 2, 1, 4]]
    phase = torch.randn(4, 2)
    h0 = torch.randn(4, 20)

    logits, next_h = net(obs, prev, phase, h0)

    # 90-deg grid rotation (rot90 k=1): new [0,1,2,3,4] corresponds to old [3,2,0,1,4],
    # and phase (py=sin, px=cos) rotates as py'=-px, px'=py -> [-px, py].
    perm = [3, 2, 0, 1, 4]
    obs_rot = torch.rot90(obs, k=1, dims=(-2, -1))
    prev_rot = prev[:, perm]
    phase_rot = torch.stack([-phase[:, 1], phase[:, 0]], dim=-1)

    logits_rot, next_h_rot = net(obs_rot, prev_rot, phase_rot, h0)
    assert next_h is not None and next_h_rot is not None
    assert torch.allclose(next_h_rot, next_h, atol=1e-5)
    assert torch.allclose(logits_rot, logits[:, perm], atol=1e-5)


@pytest.mark.parametrize("crossover_type", ["hybrid", "neuron", "arithmetic", "none"])
def test_ga_elitism_and_operators(crossover_type: str) -> None:
    model_cfg = ModelConfig(arch="cnn_gru")
    ga_cfg = GAConfig(
        ga_mode="standard",
        population_size=10,
        elite_count=2,
        immigrant_count=1,
        crossover_type=crossover_type,
        crossover_rate=0.8,
        mutation_sigma=0.05,
        stagnation_patience=2,
    )
    ga = GeneticAlgorithm(
        policy_factory=lambda: PursuitPolicyNet(model_cfg, obs_range=7, n_actions=5),
        ga_cfg=ga_cfg,
        seed=7,
    )
    initial_pop = ga.population.copy()
    fitnesses = np.array([float(i) for i in range(10)], dtype=np.float32)

    next_pop = ga.step(fitnesses)
    assert next_pop.shape == initial_pop.shape
    assert np.all(np.isfinite(next_pop))
    # Elites (indices 9 and 8) must be preserved verbatim at slots 0 and 1
    assert np.allclose(next_pop[0], initial_pop[9])
    assert np.allclose(next_pop[1], initial_pop[8])
    # Offspring should differ from elites due to mutation
    assert not np.allclose(next_pop[2], initial_pop[9])

    # Trigger stagnation boost after 2 non-improving generations
    sigma_before = ga.sigma
    ga.step(np.full(10, -10.0, dtype=np.float32))
    ga.step(np.full(10, -10.0, dtype=np.float32))
    assert ga.sigma >= sigma_before


def test_ga_guided_mode() -> None:
    model_cfg = ModelConfig(arch="cnn")
    ga_cfg = GAConfig(
        ga_mode="guided",
        population_size=12,
        elite_count=3,
        immigrant_count=1,
        mutation_sigma=0.03,
    )
    ga = GeneticAlgorithm(
        policy_factory=lambda: PursuitPolicyNet(model_cfg, obs_range=7, n_actions=5),
        ga_cfg=ga_cfg,
        seed=11,
    )
    f0 = np.arange(12, dtype=np.float32)
    p1 = ga.step(f0)
    assert p1.shape == (12, ga.num_params)
    assert np.all(np.isfinite(p1))
    f1 = np.arange(12, dtype=np.float32) + 1.0
    p2 = ga.step(f1)
    assert p2.shape == (12, ga.num_params)
    assert np.all(np.isfinite(p2))


def test_smoke_train_and_checkpoint_load(tmp_path: Path) -> None:
    cfg = TrainConfig(
        env=EnvConfig(max_cycles=25),
        model=ModelConfig(arch="cnn_gru"),
        ga=GAConfig(
            population_size=6,
            generations=2,
            elite_count=2,
            episodes_per_genome=1,
            elite_eval_episodes=1,
            immigrant_count=1,
        ),
        seed=123,
        device="cpu",
        num_workers=1,
        eval_episodes=2,
        output_dir=str(tmp_path / "smoke_run"),
        save_gif=False,
    )
    summary = train(cfg)
    assert "benchmarks" in summary

    out_dir = Path(cfg.output_dir)
    assert (out_dir / "best_model.pt").is_file()
    assert (out_dir / "last_checkpoint.pt").is_file()
    assert (out_dir / "history.json").is_file()
    assert (out_dir / "eval_summary.json").is_file()
    assert (out_dir / "training_curves.png").is_file()

    loaded_net, loaded_env_cfg, loaded_model_cfg, _ = load_checkpoint(
        out_dir / "best_model.pt"
    )
    assert loaded_env_cfg.max_cycles == 25
    assert loaded_model_cfg.arch == "cnn_gru"
    assert loaded_net.num_params == summary["num_params"]

    # Legacy pre-transformer checkpoints lacked the non-learned `rel_coords` buffer.
    legacy_ckpt_path = out_dir / "legacy_best_model.pt"
    raw_ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu", weights_only=False)
    raw_ckpt["state_dict"] = {
        k: v for k, v in raw_ckpt["state_dict"].items() if k != "rel_coords"
    }
    torch.save(raw_ckpt, legacy_ckpt_path)
    legacy_net, _, _, _ = load_checkpoint(legacy_ckpt_path)
    assert legacy_net.num_params == summary["num_params"]
