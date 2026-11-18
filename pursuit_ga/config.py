"""Configuration dataclasses for PettingZoo Pursuit Deep Neuroevolution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EnvConfig:
    """Configuration for PettingZoo SISL Pursuit (sisl/pursuit-v5)."""

    env_id: str = "sisl/pursuit-v5"
    max_cycles: int = 500
    x_size: int = 16
    y_size: int = 16
    shared_reward: bool = True
    n_evaders: int = 30
    n_pursuers: int = 8
    obs_range: int = 7
    n_catch: int = 2
    freeze_evaders: bool = False
    tag_reward: float = 0.01
    catch_reward: float = 5.0
    urgency_reward: float = -0.1
    surround: bool = True
    constraint_window: float = 1.0

    def to_make_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by pettingzoo.make('parallel', env_id, ...)."""
        return {
            "max_cycles": self.max_cycles,
            "x_size": self.x_size,
            "y_size": self.y_size,
            "shared_reward": self.shared_reward,
            "n_evaders": self.n_evaders,
            "n_pursuers": self.n_pursuers,
            "obs_range": self.obs_range,
            "n_catch": self.n_catch,
            "freeze_evaders": self.freeze_evaders,
            "tag_reward": self.tag_reward,
            "catch_reward": self.catch_reward,
            "urgency_reward": self.urgency_reward,
            "surround": self.surround,
            "constraint_window": self.constraint_window,
        }


@dataclass
class ModelConfig:
    """Architecture and action selection configuration for the Deep MARL policy."""

    arch: str = "eq_transformer"  # "cnn", "cnn_gru", "eq_cnn_gru", "transformer", or "eq_transformer"
    conv_channels: tuple[int, int, int] = (8, 12, 8)
    hidden_dim: int = 24
    gru_dim: int = 20
    transformer_dim: int = 16
    transformer_heads: int = 2
    transformer_layers: int = 1
    transformer_ffn_dim: int = 32
    activation: str = "tanh"  # "tanh", "elu", "leaky_relu"
    use_prev_action: bool = True
    use_agent_phase: bool = True
    stochastic_eval: bool = True
    temperature: float = 0.15


@dataclass
class GAConfig:
    """Hyperparameters for the Genetic Algorithm."""

    ga_mode: str = "standard"  # "standard" or "guided"
    population_size: int = 96
    generations: int = 70
    elite_count: int = 10
    elite_eval_episodes: int = 8
    episodes_per_genome: int = 3
    tournament_size: int = 4
    crossover_rate: float = 0.30
    crossover_type: str = "neuron"  # "neuron", "arithmetic", "hybrid", "none"
    mutation_rate: float = 1.0  # probability an offspring undergoes mutation
    mutation_sigma: float = 0.055
    mutation_sigma_min: float = 0.007
    mutation_decay: float = 0.978
    consensus_lr: float = 1.5
    stagnation_patience: int = 12
    stagnation_boost: float = 1.25
    immigrant_count: int = 3
    layer_scaled_mutation: bool = True
    catch_bonus_weight: float = 0.8  # auxiliary capture shaping; eval uses pure PettingZoo return


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", or "cpu"
    num_workers: int = 12
    eval_episodes: int = 20
    output_dir: str = "runs/pursuit_ga"
    save_gif: bool = True

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        model_raw = raw.get("model", {})
        if "conv_channels" in model_raw:
            model_raw["conv_channels"] = tuple(model_raw["conv_channels"])
        return cls(
            env=EnvConfig(**raw.get("env", {})),
            model=ModelConfig(**model_raw),
            ga=GAConfig(**raw.get("ga", {})),
            seed=raw.get("seed", 42),
            device=raw.get("device", "auto"),
            num_workers=raw.get("num_workers", 12),
            eval_episodes=raw.get("eval_episodes", 20),
            output_dir=raw.get("output_dir", "runs/pursuit_ga"),
            save_gif=raw.get("save_gif", True),
        )
