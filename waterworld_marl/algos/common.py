"""Shared utilities and normalization helpers for WaterWorld MARL algorithms."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    """Online Welford running mean/variance tracker for observation normalization."""

    def __init__(self, shape: tuple[int, ...], eps: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        if x64.ndim == 1:
            x64 = x64[None, :]
        else:
            x64 = x64.reshape(-1, *self.mean.shape)
        batch_mean = np.mean(x64, axis=0)
        batch_var = np.var(x64, axis=0)
        batch_count = x64.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * (self.count * batch_count / tot_count)
        self.mean = new_mean
        self.var = m2 / max(tot_count, 1e-8)
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 5.0) -> np.ndarray:
        norm = (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-6)
        return np.clip(norm, -clip, clip).astype(np.float32)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA GPU is available.")
    return dev


def append_phase_features(obs: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Append periodic (sin, cos) agent phase features to observations.

    Supports both (A, obs_dim) and (E, A, obs_dim).
    """
    if obs.ndim == 2:
        return np.concatenate([obs, phase], axis=-1).astype(np.float32)
    e = obs.shape[0]
    phase_e = np.broadcast_to(phase[None, :, :], (e, phase.shape[0], phase.shape[1]))
    return np.concatenate([obs, phase_e], axis=-1).astype(np.float32)


class RunningMeanStdGPU:
    """Tensor-native online Welford mean/variance tracker running directly on CUDA/CPU."""

    def __init__(self, shape: tuple[int, ...], device: torch.device, eps: float = 1e-4) -> None:
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = float(eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, *self.mean.shape)
        b_cnt = int(flat.shape[0])
        if b_cnt == 0:
            return
        b_mean = flat.mean(dim=0)
        b_var = flat.var(dim=0, unbiased=False)
        delta = b_mean - self.mean
        tot = self.count + b_cnt
        new_mean = self.mean + delta * (b_cnt / tot)
        m2 = self.var * self.count + b_var * b_cnt + (delta * delta) * (self.count * b_cnt / tot)
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / max(tot, 1e-8))
        self.count = tot

    def normalize(self, x: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        norm = (x - self.mean) / torch.sqrt(self.var + 1e-6)
        return torch.clamp(norm, -clip, clip)

    def to_numpy_state(self) -> dict[str, Any]:
        return {
            "mean": self.mean.detach().cpu().numpy().astype(np.float64),
            "var": self.var.detach().cpu().numpy().astype(np.float64),
            "count": float(self.count),
        }


def append_phase_features_torch(obs: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Append (A, 2) agent phase features to GPU observation batch (E, A, obs_dim)."""
    phase_e = phase.unsqueeze(0).expand(obs.shape[0], -1, -1)
    return torch.cat([obs, phase_e], dim=-1)


class _RayTransformerBlock(nn.Module):
    """Pre-LN Multi-Head Self-Attention + FFN block using scaled_dot_product_attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_sz, seq_len, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b_sz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn_out = self.proj(attn.transpose(1, 2).reshape(b_sz, seq_len, self.d_model))
        x = x + attn_out
        return x + self.ffn(self.norm2(x))


class RayTransformerEncoder(nn.Module):
    """Radial-sensor Self-Attention encoder over WaterWorld sensor rays + global state token.

    Converts flat WaterWorld actor/DQN features `(obs_dim + 2)` or CTDE critic
    features `(2 * obs_dim + 2)` into `(S + 1)` tokens (`[STATE, ray_0..ray_{S-1}]`)
    tagged with cyclic `[cos(theta), sin(theta)]` geometric ray encodings.
    """

    ray_pos: torch.Tensor

    def __init__(
        self,
        n_sensors: int,
        speed_features: bool,
        is_ctde_critic: bool = False,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 64,
    ) -> None:
        super().__init__()
        self.n_sensors = int(n_sensors)
        self.feats_per_sensor = 8 if speed_features else 5
        self.is_ctde_critic = bool(is_ctde_critic)
        self.obs_dim = self.n_sensors * self.feats_per_sensor + 2

        angles = np.linspace(0.0, 2.0 * math.pi, self.n_sensors, endpoint=False, dtype=np.float32)
        ray_pos = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
        self.register_buffer("ray_pos", torch.from_numpy(ray_pos).unsqueeze(0))

        ray_in_dim = (2 * self.feats_per_sensor if is_ctde_critic else self.feats_per_sensor) + 2
        state_in_dim = 6 if is_ctde_critic else 4
        self.ray_proj = nn.Linear(ray_in_dim, d_model)
        self.state_proj = nn.Linear(state_in_dim, d_model)
        self.blocks = nn.ModuleList(
            [_RayTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_dim = 4 * d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_sz = x.shape[0]
        s_cnt = self.n_sensors
        f_cnt = self.feats_per_sensor
        s_flat = s_cnt * f_cnt

        if self.is_ctde_critic:
            od = self.obs_dim
            local_s = x[:, :s_flat].reshape(b_sz, f_cnt, s_cnt).transpose(1, 2)
            local_fl = x[:, s_flat:od]
            team_s = x[:, od : od + s_flat].reshape(b_sz, f_cnt, s_cnt).transpose(1, 2)
            team_fl = x[:, od + s_flat : 2 * od]
            phase = x[:, 2 * od :]
            rays = torch.cat([local_s, team_s], dim=-1)
            state = torch.cat([local_fl, team_fl, phase], dim=-1)
        else:
            rays = x[:, :s_flat].reshape(b_sz, f_cnt, s_cnt).transpose(1, 2)
            state = x[:, s_flat:]

        ray_xy = self.ray_pos.expand(b_sz, -1, -1)
        ray_enc = torch.cat([rays, ray_xy], dim=-1)
        r_tok = self.ray_proj(ray_enc)
        s_tok = self.state_proj(state).unsqueeze(1)
        tok = torch.cat([s_tok, r_tok], dim=1)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.out_norm(tok)
        cls_tok = tok[:, 0]
        ray_out = tok[:, 1:]
        mean_tok = ray_out.mean(dim=1)
        cos_tok = (ray_out * self.ray_pos[:, :, 0:1]).mean(dim=1)
        sin_tok = (ray_out * self.ray_pos[:, :, 1:2]).mean(dim=1)
        return torch.cat([cls_tok, mean_tok, cos_tok, sin_tok], dim=-1)
