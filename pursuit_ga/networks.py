"""Deep neural network policy architectures for MARL neuroevolution on Pursuit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pursuit_ga.config import ModelConfig


@dataclass(frozen=True)
class ParamSlice:
    """Metadata describing a contiguous parameter tensor slice inside a flat genome."""

    name: str
    start: int
    end: int
    shape: tuple[int, ...]
    scale: float


def _make_activation(name: str) -> nn.Module:
    name_lower = name.lower()
    if name_lower == "tanh":
        return nn.Tanh()
    if name_lower == "elu":
        return nn.ELU()
    if name_lower == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name_lower == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name!r}")


def build_agent_phase_features(n_agents: int) -> np.ndarray:
    """Build static (N, 2) sin/cos phase identifiers for symmetry breaking."""
    angles = 2.0 * math.pi * np.arange(n_agents, dtype=np.float32) / max(n_agents, 1)
    return np.stack([np.sin(angles), np.cos(angles)], axis=-1).astype(np.float32)


def rotate_obs_c4(obs_chw: torch.Tensor) -> torch.Tensor:
    """Orient local observation grids into the 4 movement frames [LEFT, RIGHT, DOWN, UP].

    Each oriented frame aligns the candidate movement step to canonical AHEAD (-Y).
    Input shape: (..., C, H, W) -> Output shape: (..., 4, C, H, W).
    """
    return torch.stack(
        [
            torch.rot90(obs_chw, k=3, dims=(-2, -1)),
            torch.rot90(obs_chw, k=1, dims=(-2, -1)),
            torch.rot90(obs_chw, k=2, dims=(-2, -1)),
            obs_chw,
        ],
        dim=-4,
    )


def rotate_prev_action_c4(prev_actions: torch.Tensor) -> torch.Tensor:
    """Permute 5-action history into [same, opposite, right90, left90, stay] per move dir."""
    p0, p1, p2, p3, p4 = prev_actions.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([p0, p1, p3, p2, p4], dim=-1),
            torch.stack([p1, p0, p2, p3, p4], dim=-1),
            torch.stack([p2, p3, p0, p1, p4], dim=-1),
            torch.stack([p3, p2, p1, p0, p4], dim=-1),
        ],
        dim=-2,
    )


def rotate_agent_phase_c4(agent_phase: torch.Tensor) -> torch.Tensor:
    """Project (sin, cos) agent phase onto each candidate move direction [ahead, perp]."""
    py, px = agent_phase.unbind(dim=-1)
    ahead = torch.stack([-px, px, py, -py], dim=-1)
    perp = torch.stack([-py, py, -px, px], dim=-1)
    return torch.stack([ahead, perp], dim=-1)


class _GridTransformerBlock(nn.Module):
    """Pre-LN Multi-Head Self-Attention + GELU FFN over Pursuit grid cell tokens."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"transformer_dim={d_model} must be divisible by heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_sz, seq_len, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b_sz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        out = self.proj(attn.transpose(1, 2).reshape(b_sz, seq_len, self.d_model))
        x = x + out
        return x + self.ffn(self.ln2(x))


def _build_grid_coords(obs_range: int) -> torch.Tensor:
    """Static normalized (H*W, 2) relative (dy, dx) grid coordinates in [-1, 1]."""
    coords = torch.linspace(-1.0, 1.0, obs_range, dtype=torch.float32)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)


class _AgentCommBlock(nn.Module):
    """C4-invariant Cross-Agent Self-Attention over teammates' recurrent GRU states."""

    def __init__(self, in_dim: int, comm_dim: int, n_heads: int) -> None:
        super().__init__()
        if comm_dim % n_heads != 0:
            raise ValueError(f"comm_dim={comm_dim} must be divisible by comm_heads={n_heads}")
        self.comm_dim = comm_dim
        self.n_heads = n_heads
        self.head_dim = comm_dim // n_heads
        self.ln = nn.LayerNorm(in_dim)
        self.qkv = nn.Linear(in_dim, 3 * comm_dim)
        self.proj = nn.Linear(comm_dim, comm_dim)
        self.out_norm = nn.LayerNorm(comm_dim)

    def forward(self, h_team: torch.Tensor) -> torch.Tensor:
        """Exchange messages within each team group: (G, A, in_dim) -> (G, A, comm_dim)."""
        g_sz, a_sz, _ = h_team.shape
        hn = self.ln(h_team)
        qkv = self.qkv(hn).view(g_sz, a_sz, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        out = self.proj(attn.transpose(1, 2).reshape(g_sz, a_sz, self.comm_dim))
        return self.out_norm(out)


class PursuitPolicyNet(nn.Module):
    """Deep CNN or Patch-Transformer (optionally C4-equivariant + GRU + Comm) policy for Pursuit.

    Each agent receives its local `(3, obs_range, obs_range)` grid observation,
    optionally concatenated with its previous action one-hot vector and a
    lightweight periodic agent-phase cue to avoid symmetric deadlocks in empty grid regions.
    """

    rel_coords: torch.Tensor

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        obs_range: int = 7,
        n_actions: int = 5,
    ) -> None:
        super().__init__()
        self.cfg = model_cfg or ModelConfig()
        self.obs_range = obs_range
        self.n_actions = n_actions
        self.has_conv_stem = self.cfg.arch == "eq_comm_transformer"
        self.has_comm = self.cfg.arch == "eq_comm_transformer"
        self.is_transformer = self.cfg.arch in (
            "transformer",
            "eq_transformer",
            "eq_comm_transformer",
        )
        self.is_equivariant = self.cfg.arch in (
            "eq_cnn_gru",
            "eq_transformer",
            "eq_comm_transformer",
        )
        self.is_recurrent = self.cfg.arch in (
            "cnn_gru",
            "eq_cnn_gru",
            "transformer",
            "eq_transformer",
            "eq_comm_transformer",
        )
        act_name = self.cfg.activation

        self.register_buffer("rel_coords", _build_grid_coords(obs_range).unsqueeze(0))
        self.center_idx = (obs_range * obs_range) // 2

        if self.is_transformer:
            d_model = self.cfg.transformer_dim
            self.conv = None
            if self.has_conv_stem:
                c_stem = self.cfg.conv_stem_channels
                self.stem_conv: nn.Sequential | None = nn.Sequential(
                    nn.Conv2d(3, c_stem, kernel_size=3, padding=1),
                    _make_activation(act_name),
                )
                patch_in_dim = c_stem + 2
            else:
                self.stem_conv = None
                patch_in_dim = 5
            self.patch_proj: nn.Linear | None = nn.Linear(patch_in_dim, d_model)
            self.tx_blocks: nn.ModuleList | None = nn.ModuleList(
                [
                    _GridTransformerBlock(
                        d_model=d_model,
                        n_heads=self.cfg.transformer_heads,
                        d_ff=self.cfg.transformer_ffn_dim,
                    )
                    for _ in range(self.cfg.transformer_layers)
                ]
            )
            self.tx_norm: nn.LayerNorm | None = nn.LayerNorm(d_model)
            spatial_out_dim = 2 * d_model
        else:
            c1, c2, c3 = self.cfg.conv_channels
            self.stem_conv = None
            self.patch_proj = None
            self.tx_blocks = None
            self.tx_norm = None
            self.conv = nn.Sequential(
                nn.Conv2d(3, c1, kernel_size=3, padding=1),
                _make_activation(act_name),
                nn.Conv2d(c1, c2, kernel_size=3, padding=0),
                _make_activation(act_name),
                nn.Conv2d(c2, c3, kernel_size=3, padding=0),
                _make_activation(act_name),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, obs_range, obs_range)
                spatial_out_dim = int(self.conv(dummy).numel())

        aux_dim = 0
        if self.cfg.use_prev_action:
            aux_dim += n_actions
        if self.cfg.use_agent_phase:
            aux_dim += 2
        self.aux_dim = aux_dim

        if self.is_equivariant:
            inv_in_dim = spatial_out_dim + (1 if self.cfg.use_prev_action else 0)
            self.fc_in = nn.Sequential(
                nn.Linear(inv_in_dim, self.cfg.hidden_dim),
                _make_activation(act_name),
            )
            self.gru = nn.GRUCell(self.cfg.hidden_dim, self.cfg.gru_dim)
            if self.has_comm:
                self.comm: _AgentCommBlock | None = _AgentCommBlock(
                    in_dim=self.cfg.gru_dim,
                    comm_dim=self.cfg.comm_dim,
                    n_heads=self.cfg.comm_heads,
                )
                comm_extra = self.cfg.comm_dim
            else:
                self.comm = None
                comm_extra = 0
            dir_in_dim = spatial_out_dim + self.cfg.gru_dim + comm_extra + aux_dim
            self.dir_fc = nn.Sequential(
                nn.Linear(dir_in_dim, self.cfg.hidden_dim),
                _make_activation(act_name),
            )
            self.dir_head: nn.Linear | None = nn.Linear(self.cfg.hidden_dim, 1)
            self.stay_head: nn.Linear | None = nn.Linear(self.cfg.gru_dim + comm_extra, 1)
            self.head: nn.Linear | None = None
        else:
            self.comm = None
            self.fc_in = nn.Sequential(
                nn.Linear(spatial_out_dim + aux_dim, self.cfg.hidden_dim),
                _make_activation(act_name),
            )
            if self.is_recurrent:
                self.gru = nn.GRUCell(self.cfg.hidden_dim, self.cfg.gru_dim)
                head_in_dim = self.cfg.gru_dim
            elif self.cfg.arch == "cnn":
                self.gru = None
                head_in_dim = self.cfg.hidden_dim
            else:
                raise ValueError(f"Unsupported architecture: {self.cfg.arch!r}")

            self.dir_fc = None
            self.dir_head = None
            self.stay_head = None
            self.head = nn.Linear(head_in_dim, n_actions)

        self._init_weights()
        self._param_slices = self._compute_param_slices()
        self._num_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self) -> None:
        gain = nn.init.calculate_gain("tanh") if self.cfg.activation == "tanh" else 1.0
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRUCell):
                nn.init.orthogonal_(module.weight_ih, gain=1.0)
                nn.init.orthogonal_(module.weight_hh, gain=1.0)
                nn.init.zeros_(module.bias_ih)
                nn.init.zeros_(module.bias_hh)
        # Moderate gain gives diverse initial policies without saturating logits
        if self.head is not None:
            nn.init.orthogonal_(self.head.weight, gain=0.8)
        if self.dir_head is not None:
            nn.init.orthogonal_(self.dir_head.weight, gain=0.8)
        if self.stay_head is not None:
            nn.init.orthogonal_(self.stay_head.weight, gain=0.8)

    def _compute_param_slices(self) -> list[ParamSlice]:
        slices: list[ParamSlice] = []
        offset = 0
        raw_scales: list[float] = []
        temp: list[tuple[str, int, int, tuple[int, ...], float]] = []

        for name, param in self.named_parameters():
            numel = param.numel()
            if param.ndim >= 2:
                fan_in = int(np.prod(param.shape[1:]))
                scale = 1.0 / math.sqrt(max(fan_in, 1))
            else:
                scale = 0.25
            raw_scales.append(scale)
            temp.append((name, offset, offset + numel, tuple(param.shape), scale))
            offset += numel

        median_scale = float(np.median(raw_scales)) if raw_scales else 1.0
        for name, start, end, shape, scale in temp:
            norm_scale = float(np.clip(scale / max(median_scale, 1e-8), 0.35, 2.5))
            slices.append(ParamSlice(name, start, end, shape, norm_scale))
        return slices

    @property
    def num_params(self) -> int:
        return self._num_params

    @property
    def param_slices(self) -> list[ParamSlice]:
        return self._param_slices

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor | None:
        if self.gru is None:
            return None
        return torch.zeros(batch_size, self.cfg.gru_dim, dtype=torch.float32, device=device)

    def _encode_spatial(self, obs_chw: torch.Tensor) -> torch.Tensor:
        if self.is_transformer:
            assert self.patch_proj is not None and self.tx_blocks is not None and self.tx_norm is not None
            if self.stem_conv is not None:
                obs_chw = self.stem_conv(obs_chw)
            bsz, c_ch, h_sz, w_sz = obs_chw.shape
            n_tok = h_sz * w_sz
            cells = obs_chw.permute(0, 2, 3, 1).reshape(bsz, n_tok, c_ch)
            coords = self.rel_coords.expand(bsz, -1, -1)
            tok = self.patch_proj(torch.cat([cells, coords], dim=-1))
            for blk in self.tx_blocks:
                tok = blk(tok)
            tok = self.tx_norm(tok)
            return torch.cat([tok[:, self.center_idx], tok.mean(dim=1)], dim=-1)
        assert self.conv is not None
        return self.conv(obs_chw).flatten(start_dim=1)

    def forward(
        self,
        obs_chw: torch.Tensor,
        prev_actions_onehot: torch.Tensor | None = None,
        agent_phase: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        team_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz = obs_chw.shape[0]
        if self.cfg.use_prev_action and prev_actions_onehot is None:
            prev_actions_onehot = torch.zeros(
                bsz, self.n_actions, dtype=obs_chw.dtype, device=obs_chw.device
            )
        if self.cfg.use_agent_phase and agent_phase is None:
            agent_phase = torch.zeros(
                bsz, 2, dtype=obs_chw.dtype, device=obs_chw.device
            )

        if self.is_equivariant:
            assert self.gru is not None and self.dir_fc is not None
            assert self.dir_head is not None and self.stay_head is not None
            obs_rot = rotate_obs_c4(obs_chw)  # (N, 4, C, H, W)
            feat_rot = self._encode_spatial(
                obs_rot.reshape(bsz * 4, obs_chw.shape[1], obs_chw.shape[2], obs_chw.shape[3])
            ).view(bsz, 4, -1)
            feat_mean = feat_rot.mean(dim=1)

            inv_parts = [feat_mean]
            if self.cfg.use_prev_action:
                assert prev_actions_onehot is not None
                inv_parts.append(prev_actions_onehot[:, 4:5])
            inv_x = torch.cat(inv_parts, dim=1) if len(inv_parts) > 1 else inv_parts[0]
            h = self.fc_in(inv_x)
            if hidden is None:
                hidden = self.init_hidden(bsz, device=obs_chw.device)
            next_hidden = self.gru(h, hidden)

            dir_parts = [feat_rot, next_hidden.unsqueeze(1).expand(-1, 4, -1)]
            if self.comm is not None:
                t_sz = team_size if (team_size is not None and bsz % team_size == 0) else bsz
                comm_ctx = self.comm(next_hidden.view(bsz // t_sz, t_sz, -1)).reshape(bsz, -1)
                dir_parts.append(comm_ctx.unsqueeze(1).expand(-1, 4, -1))
                stay_in = torch.cat([next_hidden, comm_ctx], dim=-1)
            else:
                stay_in = next_hidden
            if self.cfg.use_prev_action:
                assert prev_actions_onehot is not None
                dir_parts.append(rotate_prev_action_c4(prev_actions_onehot))
            if self.cfg.use_agent_phase:
                assert agent_phase is not None
                dir_parts.append(rotate_agent_phase_c4(agent_phase))

            dir_in = torch.cat(dir_parts, dim=-1).reshape(bsz * 4, -1)
            dir_logits = self.dir_head(self.dir_fc(dir_in)).view(bsz, 4)
            stay_logit = self.stay_head(stay_in)
            logits = torch.cat([dir_logits, stay_logit], dim=-1)
            return logits, next_hidden

        feat = self._encode_spatial(obs_chw)
        parts = [feat]
        if self.cfg.use_prev_action:
            assert prev_actions_onehot is not None
            parts.append(prev_actions_onehot)
        if self.cfg.use_agent_phase:
            assert agent_phase is not None
            parts.append(agent_phase)

        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        h = self.fc_in(x)

        assert self.head is not None
        if self.gru is not None:
            if hidden is None:
                hidden = self.init_hidden(bsz, device=obs_chw.device)
            next_hidden = self.gru(h, hidden)
            logits = self.head(next_hidden)
            return logits, next_hidden

        logits = self.head(h)
        return logits, None

    @torch.inference_mode()
    def get_genome(self) -> np.ndarray:
        """Flatten all model parameters into a 1D float32 numpy vector."""
        parts = [p.detach().cpu().numpy().ravel() for p in self.parameters()]
        return np.concatenate(parts).astype(np.float32, copy=False)

    @torch.inference_mode()
    def set_genome(self, genome: np.ndarray) -> None:
        """Load a flat 1D parameter vector into the network in-place."""
        if genome.shape != (self._num_params,):
            raise ValueError(
                f"Expected genome of shape ({self._num_params},), got {genome.shape}"
            )
        tensor_vec = torch.as_tensor(genome, dtype=torch.float32)
        for p, sl in zip(self.parameters(), self._param_slices):
            p.copy_(tensor_vec[sl.start : sl.end].view(sl.shape).to(p.device))
