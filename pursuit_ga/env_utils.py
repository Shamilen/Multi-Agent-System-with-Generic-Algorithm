"""Environment creation and observation preprocessing utilities for SISL Pursuit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pettingzoo

from pursuit_ga.config import EnvConfig


def make_pursuit_env(
    env_cfg: EnvConfig | None = None,
    render_mode: str | None = None,
) -> Any:
    """Create a PettingZoo ParallelEnv instance for sisl/pursuit-v5."""
    cfg = env_cfg or EnvConfig()
    kwargs = cfg.to_make_kwargs()
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return pettingzoo.make("parallel", cfg.env_id, **kwargs)


def preprocess_observations(
    obs_dict: dict[str, np.ndarray],
    agent_ids: list[str],
) -> np.ndarray:
    """Convert PettingZoo Pursuit observation dict into (N, 3, H, W) float32 tensor.

    Raw PettingZoo Pursuit observations have shape (obs_range, obs_range, 3):
      - Channel 0: walls / obstacles (0 or 1)
      - Channel 1: pursuer count per cell (center includes self >= 1)
      - Channel 2: evader count per cell

    We convert to channel-first (N, 3, H, W), remove the constant self count of 1
    from the center of channel 1 so channel 1 encodes allied pursuers, and compress
    agent counts smoothly into [0, 1) via tanh(count / 2.0).
    """
    obs_batch = np.stack([obs_dict[ag] for ag in agent_ids], axis=0).astype(np.float32)
    # (N, H, W, 3) -> (N, 3, H, W)
    obs_chw = np.transpose(obs_batch, (0, 3, 1, 2)).copy()
    h, w = obs_chw.shape[2], obs_chw.shape[3]
    cy, cx = h // 2, w // 2
    obs_chw[:, 1, cy, cx] = np.maximum(obs_chw[:, 1, cy, cx] - 1.0, 0.0)
    obs_chw[:, 1, :, :] = np.tanh(obs_chw[:, 1, :, :] * 0.5)
    obs_chw[:, 2, :, :] = np.tanh(obs_chw[:, 2, :, :] * 0.5)
    return obs_chw


def get_evaders_remaining(env: Any) -> int:
    """Return the number of uncaught evaders currently remaining in the environment."""
    return int(env.unwrapped.env.evader_layer.n_agents())


def get_pursuer_count(env: Any) -> int:
    """Return the number of pursuers in the environment."""
    return int(env.unwrapped.env.pursuer_layer.n_agents())
