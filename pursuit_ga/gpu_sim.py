"""CUDA-accelerated vectorized PettingZoo SISL Pursuit simulator and batched GA evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pettingzoo.sisl.pursuit.utils import agent_utils, two_d_maps

from pursuit_ga.config import EnvConfig, ModelConfig
from pursuit_ga.evaluator import PopulationEvaluation
from pursuit_ga.networks import (
    PursuitPolicyNet,
    _build_grid_coords,
    build_agent_phase_features,
    rotate_agent_phase_c4,
    rotate_obs_c4,
    rotate_prev_action_c4,
)


def _compute_need_to_surround_map(map_matrix: np.ndarray) -> np.ndarray:
    """Exact replica of Pursuit.need_to_surround(x, y) over all grid coordinates."""
    xs, ys = map_matrix.shape
    out = np.zeros((xs, ys), dtype=np.int64)
    surround_mask = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
    for x in range(xs):
        for y in range(ys):
            tosur = 4
            if x == 0 or x == (xs - 1):
                tosur -= 1
            if y == 0 or y == (ys - 1):
                tosur -= 1
            for dx, dy in surround_mask:
                xn, yn = x + dx, y + dy
                if not 0 < xn < xs or not 0 < yn < ys:
                    continue
                if map_matrix[xn, yn] == -1:
                    tosur -= 1
            out[x, y] = tosur
    return out


class VectorizedPursuitGPU:
    """Batched PyTorch/CUDA implementation of PettingZoo sisl/pursuit-v5.

    Matches `pursuit_base.Pursuit` + `pursuit.raw_env` + `aec_to_parallel_wrapper`:
      - Exact rectangle_map obstacle layout and non-adjacent spawn rules
      - Exact sequential pursuer move + tag_reward accumulation per cycle
      - Exact surround capture check & evader random movement
      - Exact (N, 3, obs_range, obs_range) observation layout matching
        `preprocess_observations(obs_dict, agent_ids)` after `np.swapaxes(o, 2, 0)`.
    """

    def __init__(
        self,
        batch_size: int,
        env_cfg: EnvConfig,
        device: torch.device,
    ) -> None:
        self.batch_size = int(batch_size)
        self.cfg = env_cfg
        self.device = device

        self.xs = env_cfg.x_size
        self.ys = env_cfg.y_size
        self.n_p = env_cfg.n_pursuers
        self.n_e = env_cfg.n_evaders
        self.obs_range = env_cfg.obs_range
        self.obs_offset = (self.obs_range - 1) // 2

        map_np = two_d_maps.rectangle_map(self.xs, self.ys)
        self.map_matrix_np = map_np
        self.obstacle_map = torch.from_numpy((map_np == -1)).to(device=device)
        self.abs_map = torch.from_numpy(np.abs(map_np).astype(np.float32)).to(device=device)
        self.need_surr_map = torch.from_numpy(_compute_need_to_surround_map(map_np)).to(
            device=device
        )

        # Action deltas from DiscreteAgent: [[-1,0], [1,0], [0,1], [0,-1], [0,0]]
        self.motion_dx = torch.tensor([-1, 1, 0, 0, 0], dtype=torch.long, device=device)
        self.motion_dy = torch.tensor([0, 0, 1, -1, 0], dtype=torch.long, device=device)
        self.surr_dx = torch.tensor([-1, 1, 0, 0], dtype=torch.long, device=device)
        self.surr_dy = torch.tensor([0, 0, 1, -1], dtype=torch.long, device=device)

        # Precompute relative crop grid offsets (obs_range, obs_range)
        # Note: pursuit.py returns np.swapaxes(obs_3xy, 2, 0), so axis H=dy, axis W=dx!
        offs = torch.arange(-self.obs_offset, self.obs_offset + 1, dtype=torch.long, device=device)
        grid_dy, grid_dx = torch.meshgrid(offs, offs, indexing="ij")
        self.crop_dy = grid_dy.reshape(1, 1, self.obs_range, self.obs_range) + self.obs_offset
        self.crop_dx = grid_dx.reshape(1, 1, self.obs_range, self.obs_range) + self.obs_offset

        # State tensors allocated on reset
        self.px = torch.zeros(self.batch_size, self.n_p, dtype=torch.long, device=device)
        self.py = torch.zeros(self.batch_size, self.n_p, dtype=torch.long, device=device)
        self.ex = torch.zeros(self.batch_size, self.n_e, dtype=torch.long, device=device)
        self.ey = torch.zeros(self.batch_size, self.n_e, dtype=torch.long, device=device)
        self.ev_alive = torch.ones(self.batch_size, self.n_e, dtype=torch.bool, device=device)
        self.p_grid = torch.zeros(self.batch_size, self.xs * self.ys, dtype=torch.float32, device=device)
        self.e_grid = torch.zeros(self.batch_size, self.xs * self.ys, dtype=torch.float32, device=device)
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        self.cycles = torch.zeros(self.batch_size, dtype=torch.long, device=device)

        self.b_arange = torch.arange(self.batch_size, device=device)
        self.batch_idx_p = self.b_arange.unsqueeze(1).expand(self.batch_size, self.n_p)
        self.batch_idx_e = self.b_arange.unsqueeze(1).expand(self.batch_size, self.n_e)
        self.ones_p = torch.ones(self.batch_size, self.n_p, dtype=torch.float32, device=device)
        self.ones_p_single = torch.ones(self.batch_size, dtype=torch.float32, device=device)
        self.catch_cells = torch.zeros(self.batch_size, self.xs * self.ys, dtype=torch.float32, device=device)
        self.surr_dx_114 = self.surr_dx.view(1, 1, 4)
        self.surr_dy_114 = self.surr_dy.view(1, 1, 4)

    def reset_from_seeds(self, episode_seeds: list[int], replicate_count: int = 1) -> None:
        """Reset episodes using exact PettingZoo initial spawn rules.

        If `replicate_count > 1`, generates `len(episode_seeds)` initial layouts
        and broadcasts them across `replicate_count` genomes (Common Random Numbers).
        """
        n_unique = len(episode_seeds)
        assert n_unique * replicate_count == self.batch_size

        px_np = np.zeros((n_unique, self.n_p), dtype=np.int64)
        py_np = np.zeros((n_unique, self.n_p), dtype=np.int64)
        ex_np = np.zeros((n_unique, self.n_e), dtype=np.int64)
        ey_np = np.zeros((n_unique, self.n_e), dtype=np.int64)

        cw = self.cfg.constraint_window
        for idx, s in enumerate(episode_seeds):
            rng = np.random.default_rng(int(s))
            # Match gymnasium.utils.seeding.np_random(seed) layout flow
            xws = float(rng.uniform(0.0, 1.0 - cw))
            yws = float(rng.uniform(0.0, 1.0 - cw))
            constraints = [
                [int(self.xs * xws), int(self.xs * (xws + cw))],
                [int(self.ys * yws), int(self.ys * (yws + cw))],
            ]
            pursuers = agent_utils.create_agents(
                self.n_p, self.map_matrix_np, self.obs_range, rng, randinit=True, constraints=constraints
            )
            evaders = agent_utils.create_agents(
                self.n_e, self.map_matrix_np, self.obs_range, rng, randinit=True, constraints=constraints
            )
            for j, ag in enumerate(pursuers):
                pos = ag.current_position()
                px_np[idx, j], py_np[idx, j] = int(pos[0]), int(pos[1])
            for j, ag in enumerate(evaders):
                pos = ag.current_position()
                ex_np[idx, j], ey_np[idx, j] = int(pos[0]), int(pos[1])

        px_u = torch.from_numpy(px_np).to(self.device)
        py_u = torch.from_numpy(py_np).to(self.device)
        ex_u = torch.from_numpy(ex_np).to(self.device)
        ey_u = torch.from_numpy(ey_np).to(self.device)

        if replicate_count > 1:
            px_u = px_u.unsqueeze(0).expand(replicate_count, -1, -1).reshape(self.batch_size, self.n_p)
            py_u = py_u.unsqueeze(0).expand(replicate_count, -1, -1).reshape(self.batch_size, self.n_p)
            ex_u = ex_u.unsqueeze(0).expand(replicate_count, -1, -1).reshape(self.batch_size, self.n_e)
            ey_u = ey_u.unsqueeze(0).expand(replicate_count, -1, -1).reshape(self.batch_size, self.n_e)

        self.set_state(px_u, py_u, ex_u, ey_u)

    def set_state(
        self,
        px: torch.Tensor,
        py: torch.Tensor,
        ex: torch.Tensor,
        ey: torch.Tensor,
        ev_alive: torch.Tensor | None = None,
    ) -> None:
        self.px.copy_(px)
        self.py.copy_(py)
        self.ex.copy_(ex)
        self.ey.copy_(ey)
        if ev_alive is None:
            self.ev_alive.fill_(True)
        else:
            self.ev_alive.copy_(ev_alive)
        self.done.fill_(False)
        self.cycles.zero_()
        self._rebuild_grids()

    def _rebuild_grids(self) -> None:
        self.p_grid.zero_()
        p_flat = self.px * self.ys + self.py
        self.p_grid.scatter_add_(1, p_flat, self.ones_p)
        self.e_grid.zero_()
        e_flat = self.ex * self.ys + self.ey
        self.e_grid.scatter_add_(1, e_flat, self.ev_alive.to(torch.float32))

    def _compute_tag_rewards(self) -> torch.Tensor:
        """Compute exact Pursuit.reward() / num_agents across all pursuers (B, n_p)."""
        nx = torch.clamp(self.px.unsqueeze(-1) + self.surr_dx_114, 0, self.xs - 1)
        ny = torch.clamp(self.py.unsqueeze(-1) + self.surr_dy_114, 0, self.ys - 1)
        n_flat = (nx * self.ys + ny).reshape(self.batch_size, self.n_p * 4)
        ev_sum = self.e_grid.gather(1, n_flat).reshape(self.batch_size, self.n_p, 4).sum(dim=-1)
        return (self.cfg.tag_reward / float(self.n_p)) * ev_sum

    def _move_coords(
        self,
        cx: torch.Tensor,
        cy: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nx = cx + self.motion_dx[actions]
        ny = cy + self.motion_dy[actions]
        in_bounds = (nx >= 0) & (nx < self.xs) & (ny >= 0) & (ny < self.ys)
        safe_nx = torch.clamp(nx, 0, self.xs - 1)
        safe_ny = torch.clamp(ny, 0, self.ys - 1)
        free = ~self.obstacle_map[safe_nx, safe_ny]
        valid = in_bounds & free
        out_x = torch.where(valid, nx, cx)
        out_y = torch.where(valid, ny, cy)
        return out_x, out_y

    def get_raw_observations(self) -> torch.Tensor:
        """Return raw observations of shape (B, n_p, obs_range, obs_range, 3) matching PettingZoo."""
        pad = self.obs_offset
        ch0 = F.pad(self.abs_map.unsqueeze(0).expand(self.batch_size, -1, -1), (pad, pad, pad, pad), value=1.0)
        ch1 = F.pad(self.p_grid.view(self.batch_size, self.xs, self.ys), (pad, pad, pad, pad), value=0.0)
        ch2 = F.pad(self.e_grid.view(self.batch_size, self.xs, self.ys), (pad, pad, pad, pad), value=0.0)

        iy = self.py.view(self.batch_size, self.n_p, 1, 1) + self.crop_dy
        ix = self.px.view(self.batch_size, self.n_p, 1, 1) + self.crop_dx
        b_idx = self.b_arange.view(self.batch_size, 1, 1, 1)

        o0 = ch0[b_idx, ix, iy]
        o1 = ch1[b_idx, ix, iy]
        o2 = ch2[b_idx, ix, iy]
        return torch.stack([o0, o1, o2], dim=-1)

    def get_preprocessed_observations(self) -> torch.Tensor:
        """Return preprocessed observations of shape (B, n_p, 3, obs_range, obs_range)."""
        pad = self.obs_offset
        ch0 = F.pad(self.abs_map.unsqueeze(0).expand(self.batch_size, -1, -1), (pad, pad, pad, pad), value=1.0)
        ch1 = F.pad(self.p_grid.view(self.batch_size, self.xs, self.ys), (pad, pad, pad, pad), value=0.0)
        ch2 = F.pad(self.e_grid.view(self.batch_size, self.xs, self.ys), (pad, pad, pad, pad), value=0.0)

        iy = self.py.view(self.batch_size, self.n_p, 1, 1) + self.crop_dy
        ix = self.px.view(self.batch_size, self.n_p, 1, 1) + self.crop_dx
        b_idx = self.b_arange.view(self.batch_size, 1, 1, 1)

        o0 = ch0[b_idx, ix, iy]
        o1 = ch1[b_idx, ix, iy]
        o2 = ch2[b_idx, ix, iy]

        o1[:, :, pad, pad] = torch.clamp(o1[:, :, pad, pad] - 1.0, min=0.0)
        o1 = torch.tanh(o1 * 0.5)
        o2 = torch.tanh(o2 * 0.5)
        return torch.stack([o0, o1, o2], dim=2)

    def step(
        self,
        pursuer_actions: torch.Tensor,
        evader_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one full parallel cycle across all `batch_size` environments."""
        active = ~self.done
        active_f = active.to(torch.float32)
        cycle_rewards = torch.zeros(self.batch_size, self.n_p, dtype=torch.float32, device=self.device)

        for p in range(self.n_p):
            act_p = pursuer_actions[:, p]
            old_x = self.px[:, p]
            old_y = self.py[:, p]
            new_x, new_y = self._move_coords(old_x, old_y, act_p)
            new_x = torch.where(active, new_x, old_x)
            new_y = torch.where(active, new_y, old_y)

            old_flat = old_x * self.ys + old_y
            new_flat = new_x * self.ys + new_y
            self.p_grid.scatter_add_(1, old_flat.unsqueeze(1), (-active_f).unsqueeze(1))
            self.p_grid.scatter_add_(1, new_flat.unsqueeze(1), active_f.unsqueeze(1))
            self.px[:, p] = new_x
            self.py[:, p] = new_y

            step_r = self._compute_tag_rewards()

            if p == self.n_p - 1:
                purs_sur = torch.zeros(self.batch_size, self.n_p, dtype=torch.bool, device=self.device)
                if self.cfg.surround:
                    nx = self.ex.unsqueeze(-1) + self.surr_dx_114
                    ny = self.ey.unsqueeze(-1) + self.surr_dy_114
                    in_b = (nx >= 0) & (nx < self.xs) & (ny >= 0) & (ny < self.ys)
                    cnx = torch.clamp(nx, 0, self.xs - 1)
                    cny = torch.clamp(ny, 0, self.ys - 1)
                    n_flat = (cnx * self.ys + cny).reshape(self.batch_size, self.n_e * 4)
                    p_cnt = self.p_grid.gather(1, n_flat).reshape(self.batch_size, self.n_e, 4)
                    has_p = in_b & (p_cnt > 0)
                    distinct_sides = has_p.sum(dim=-1)
                    need = self.need_surr_map[self.ex, self.ey]
                    caught = self.ev_alive & active.unsqueeze(1) & (distinct_sides == need)
                    flank_sides = torch.clamp(distinct_sides - 1, min=0)
                    self.last_flank_score = (
                        flank_sides.to(torch.float32) * (self.ev_alive & active.unsqueeze(1)).to(torch.float32)
                    ).sum(dim=1)

                    self.catch_cells.zero_()
                    mark_f = (caught.unsqueeze(-1) & in_b).reshape(self.batch_size, self.n_e * 4).to(torch.float32)
                    self.catch_cells.scatter_add_(1, n_flat, mark_f)
                    p_flat = self.px * self.ys + self.py
                    purs_sur = self.catch_cells.gather(1, p_flat) > 0.0
                else:
                    e_flat = self.ex * self.ys + self.ey
                    stacked = self.p_grid.gather(1, e_flat) >= self.cfg.n_catch
                    caught = self.ev_alive & active.unsqueeze(1) & stacked
                    self.last_flank_score = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
                    self.catch_cells.zero_()
                    self.catch_cells.scatter_add_(1, e_flat, caught.to(torch.float32))
                    p_flat = self.px * self.ys + self.py
                    purs_sur = self.catch_cells.gather(1, p_flat) > 0.0

                self.last_overlaps = torch.clamp(self.p_grid - 1.0, min=0.0).sum(dim=1) * active_f

                self.ev_alive = self.ev_alive & (~caught)

                if not self.cfg.freeze_evaders:
                    if evader_actions is None:
                        ev_act = torch.randint(
                            0, 5, (self.batch_size, self.n_e), device=self.device
                        )
                    else:
                        ev_act = evader_actions
                    nex, ney = self._move_coords(self.ex, self.ey, ev_act)
                    move_mask = self.ev_alive & active.unsqueeze(1)
                    self.ex = torch.where(move_mask, nex, self.ex)
                    self.ey = torch.where(move_mask, ney, self.ey)

                self.e_grid.zero_()
                self.e_grid.scatter_add_(
                    1, self.ex * self.ys + self.ey, self.ev_alive.to(torch.float32)
                )

                step_r = step_r + self.cfg.catch_reward * purs_sur.to(torch.float32) + self.cfg.urgency_reward

            if self.cfg.shared_reward:
                step_r = step_r.mean(dim=1, keepdim=True).expand_as(step_r)

            cycle_rewards += step_r * active_f.unsqueeze(1)

        self.cycles += active.to(torch.long)
        all_caught = self.ev_alive.sum(dim=1) == 0
        self.done = self.done | all_caught | (self.cycles >= self.cfg.max_cycles)
        return cycle_rewards, active


class BatchedPopulationPolicyGPU:
    """Executes `P` distinct PursuitPolicyNet genomes simultaneously on CUDA."""

    def __init__(
        self,
        genomes: np.ndarray,
        model_cfg: ModelConfig,
        env_cfg: EnvConfig,
        device: torch.device,
    ) -> None:
        self.P = len(genomes)
        self.model_cfg = model_cfg
        self.env_cfg = env_cfg
        self.device = device
        self.n_actions = 5

        self.has_conv_stem = self.model_cfg.arch == "eq_comm_transformer"
        self.has_comm = self.model_cfg.arch == "eq_comm_transformer"
        self.is_transformer = self.model_cfg.arch in (
            "transformer",
            "eq_transformer",
            "eq_comm_transformer",
        )
        self.is_equivariant = self.model_cfg.arch in (
            "eq_cnn_gru",
            "eq_transformer",
            "eq_comm_transformer",
        )
        self.is_recurrent = self.model_cfg.arch in (
            "cnn_gru",
            "eq_cnn_gru",
            "transformer",
            "eq_transformer",
            "eq_comm_transformer",
        )
        self.rel_coords = _build_grid_coords(env_cfg.obs_range).to(device)
        self.center_idx = (env_cfg.obs_range * env_cfg.obs_range) // 2

        template = PursuitPolicyNet(model_cfg=model_cfg, obs_range=env_cfg.obs_range, n_actions=5)
        self.param_slices = template.param_slices
        self.num_params = template.num_params
        self.update_genomes(genomes)

    def update_genomes(self, genomes: np.ndarray) -> None:
        assert genomes.shape == (self.P, self.num_params)
        g_t = torch.from_numpy(genomes).to(device=self.device, dtype=torch.float32)
        self._tensors: dict[str, torch.Tensor] = {}
        for sl in self.param_slices:
            # Slice is (P, *shape)
            t = g_t[:, sl.start : sl.end].reshape((self.P,) + sl.shape).contiguous()
            self._tensors[sl.name] = t

        if self.is_transformer:
            if self.has_conv_stem:
                c_stem = self.model_cfg.conv_stem_channels
                self.stem_w = self._tensors["stem_conv.0.weight"].reshape(self.P * c_stem, 3, 3, 3)
                self.stem_b = self._tensors["stem_conv.0.bias"].reshape(self.P * c_stem)
            self.patch_w = self._tensors["patch_proj.weight"].transpose(1, 2).contiguous()
            self.patch_b = self._tensors["patch_proj.bias"].unsqueeze(1).contiguous()
            self.tx_norm_w = self._tensors["tx_norm.weight"].unsqueeze(1).contiguous()
            self.tx_norm_b = self._tensors["tx_norm.bias"].unsqueeze(1).contiguous()
            self.tx_layers: list[tuple[torch.Tensor, ...]] = []
            for l_idx in range(self.model_cfg.transformer_layers):
                p = f"tx_blocks.{l_idx}"
                self.tx_layers.append(
                    (
                        self._tensors[f"{p}.ln1.weight"].unsqueeze(1).contiguous(),
                        self._tensors[f"{p}.ln1.bias"].unsqueeze(1).contiguous(),
                        self._tensors[f"{p}.qkv.weight"].transpose(1, 2).contiguous(),
                        self._tensors[f"{p}.qkv.bias"].unsqueeze(1).contiguous(),
                        self._tensors[f"{p}.proj.weight"].transpose(1, 2).contiguous(),
                        self._tensors[f"{p}.proj.bias"].unsqueeze(1).contiguous(),
                        self._tensors[f"{p}.ln2.weight"].unsqueeze(1).contiguous(),
                        self._tensors[f"{p}.ln2.bias"].unsqueeze(1).contiguous(),
                        self._tensors[f"{p}.ffn.0.weight"].transpose(1, 2).contiguous(),
                        self._tensors[f"{p}.ffn.0.bias"].unsqueeze(1).contiguous(),
                        self._tensors[f"{p}.ffn.2.weight"].transpose(1, 2).contiguous(),
                        self._tensors[f"{p}.ffn.2.bias"].unsqueeze(1).contiguous(),
                    )
                )
        else:
            # Pre-pack grouped conv weights across genomes
            self.c1_w = self._tensors["conv.0.weight"].reshape(self.P * self.model_cfg.conv_channels[0], 3, 3, 3)
            self.c1_b = self._tensors["conv.0.bias"].reshape(self.P * self.model_cfg.conv_channels[0])
            self.c2_w = self._tensors["conv.2.weight"].reshape(
                self.P * self.model_cfg.conv_channels[1], self.model_cfg.conv_channels[0], 3, 3
            )
            self.c2_b = self._tensors["conv.2.bias"].reshape(self.P * self.model_cfg.conv_channels[1])
            self.c3_w = self._tensors["conv.4.weight"].reshape(
                self.P * self.model_cfg.conv_channels[2], self.model_cfg.conv_channels[1], 3, 3
            )
            self.c3_b = self._tensors["conv.4.bias"].reshape(self.P * self.model_cfg.conv_channels[2])

        self.fc_in_w = self._tensors["fc_in.0.weight"].transpose(1, 2).contiguous()
        self.fc_in_b = self._tensors["fc_in.0.bias"].unsqueeze(1).contiguous()

        if self.is_recurrent:
            self.gru_w_ih = self._tensors["gru.weight_ih"].transpose(1, 2).contiguous()
            self.gru_b_ih = self._tensors["gru.bias_ih"].unsqueeze(1).contiguous()
            self.gru_w_hh = self._tensors["gru.weight_hh"].transpose(1, 2).contiguous()
            self.gru_b_hh = self._tensors["gru.bias_hh"].unsqueeze(1).contiguous()

        if self.has_comm:
            self.comm_ln_w = self._tensors["comm.ln.weight"].unsqueeze(1).contiguous()
            self.comm_ln_b = self._tensors["comm.ln.bias"].unsqueeze(1).contiguous()
            self.comm_qkv_w = self._tensors["comm.qkv.weight"].transpose(1, 2).contiguous()
            self.comm_qkv_b = self._tensors["comm.qkv.bias"].unsqueeze(1).contiguous()
            self.comm_proj_w = self._tensors["comm.proj.weight"].transpose(1, 2).contiguous()
            self.comm_proj_b = self._tensors["comm.proj.bias"].unsqueeze(1).contiguous()
            self.comm_out_w = self._tensors["comm.out_norm.weight"].unsqueeze(1).contiguous()
            self.comm_out_b = self._tensors["comm.out_norm.bias"].unsqueeze(1).contiguous()

        if self.is_equivariant:
            self.dir_fc_w = self._tensors["dir_fc.0.weight"].transpose(1, 2).contiguous()
            self.dir_fc_b = self._tensors["dir_fc.0.bias"].unsqueeze(1).contiguous()
            self.dir_head_w = self._tensors["dir_head.weight"].transpose(1, 2).contiguous()
            self.dir_head_b = self._tensors["dir_head.bias"].unsqueeze(1).contiguous()
            self.stay_head_w = self._tensors["stay_head.weight"].transpose(1, 2).contiguous()
            self.stay_head_b = self._tensors["stay_head.bias"].unsqueeze(1).contiguous()
        else:
            self.head_w = self._tensors["head.weight"].transpose(1, 2).contiguous()
            self.head_b = self._tensors["head.bias"].unsqueeze(1).contiguous()

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        name = self.model_cfg.activation.lower()
        if name == "tanh":
            return torch.tanh(x)
        if name == "elu":
            return F.elu(x)
        if name == "leaky_relu":
            return F.leaky_relu(x, 0.1)
        if name == "relu":
            return F.relu(x)
        raise ValueError(f"Unsupported activation: {name!r}")

    def _run_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self._act(F.conv2d(x, self.c1_w, self.c1_b, padding=1, groups=self.P))
        x = self._act(F.conv2d(x, self.c2_w, self.c2_b, padding=0, groups=self.P))
        return self._act(F.conv2d(x, self.c3_w, self.c3_b, padding=0, groups=self.P))

    def _gru_step(self, h: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        gi = torch.bmm(h, self.gru_w_ih) + self.gru_b_ih
        gh = torch.bmm(hidden, self.gru_w_hh) + self.gru_b_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * hidden

    @staticmethod
    def _batched_ln(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(var + 1e-5) * w + b

    def _run_transformer_views(self, obs_views: torch.Tensor) -> torch.Tensor:
        """Encode (P, B_v, C, H, W) grid views across P genomes cleanly using bmm + SDPA."""
        P, B_v, C, H, W = obs_views.shape
        if self.has_conv_stem:
            c_stem = self.model_cfg.conv_stem_channels
            x_stem = obs_views.permute(1, 0, 2, 3, 4).reshape(B_v, P * C, H, W)
            x_stem = self._act(F.conv2d(x_stem, self.stem_w, self.stem_b, padding=1, groups=self.P))
            obs_views = x_stem.view(B_v, P, c_stem, H, W).transpose(0, 1).contiguous()
            C = c_stem

        n_tok = H * W
        d_model = self.model_cfg.transformer_dim
        n_heads = self.model_cfg.transformer_heads
        head_dim = d_model // n_heads

        cells = obs_views.permute(0, 1, 3, 4, 2).reshape(P, B_v, n_tok, C)
        coords = self.rel_coords.view(1, 1, n_tok, 2).expand(P, B_v, -1, -1)
        tok = (
            torch.bmm(torch.cat([cells, coords], dim=-1).reshape(P, B_v * n_tok, C + 2), self.patch_w)
            + self.patch_b
        )

        for (
            ln1_w,
            ln1_b,
            qkv_w,
            qkv_b,
            proj_w,
            proj_b,
            ln2_w,
            ln2_b,
            ffn0_w,
            ffn0_b,
            ffn2_w,
            ffn2_b,
        ) in self.tx_layers:
            x_n = self._batched_ln(tok, ln1_w, ln1_b)
            qkv = (torch.bmm(x_n, qkv_w) + qkv_b).view(P * B_v, n_tok, 3, n_heads, head_dim)
            q, k, v = qkv.unbind(dim=2)
            attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            attn_flat = attn.transpose(1, 2).reshape(P, B_v * n_tok, d_model)
            tok = tok + torch.bmm(attn_flat, proj_w) + proj_b

            ffn_h = F.gelu(torch.bmm(self._batched_ln(tok, ln2_w, ln2_b), ffn0_w) + ffn0_b)
            tok = tok + torch.bmm(ffn_h, ffn2_w) + ffn2_b

        tok = self._batched_ln(tok, self.tx_norm_w, self.tx_norm_b).view(P, B_v, n_tok, d_model)
        return torch.cat([tok[:, :, self.center_idx], tok.mean(dim=2)], dim=-1)

    def _run_comm(self, next_h: torch.Tensor, team_size: int | None = None) -> torch.Tensor:
        """Cross-agent Self-Attention within each (genome, episode) team: (P, E*A, gru_dim)."""
        P, B, _ = next_h.shape
        a_sz = (
            team_size
            if (team_size is not None and B % team_size == 0)
            else (self.env_cfg.n_pursuers if B % self.env_cfg.n_pursuers == 0 else B)
        )
        e_sz = B // a_sz
        c_dim = self.model_cfg.comm_dim
        c_heads = self.model_cfg.comm_heads
        c_hdim = c_dim // c_heads

        hn = self._batched_ln(next_h, self.comm_ln_w, self.comm_ln_b)
        qkv = (torch.bmm(hn, self.comm_qkv_w) + self.comm_qkv_b).view(P * e_sz, a_sz, 3, c_heads, c_hdim)
        q, k, v = qkv.unbind(dim=2)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        attn_flat = attn.transpose(1, 2).reshape(P, B, c_dim)
        out = torch.bmm(attn_flat, self.comm_proj_w) + self.comm_proj_b
        return self._batched_ln(out, self.comm_out_w, self.comm_out_b)

    @torch.inference_mode()
    def forward(
        self,
        obs_chw: torch.Tensor,  # (P, B_per_genome, 3, H, W)
        prev_actions: torch.Tensor,  # (P, B_per_genome, 5)
        agent_phase: torch.Tensor,  # (P, B_per_genome, 2)
        hidden: torch.Tensor | None,  # (P, B_per_genome, gru_dim)
        team_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        P, B, C, H, W = obs_chw.shape

        if self.is_equivariant:
            assert hidden is not None
            obs_rot = rotate_obs_c4(obs_chw)  # (P, B, 4, C, H, W)
            if self.is_transformer:
                feat_rot = self._run_transformer_views(
                    obs_rot.reshape(P, B * 4, C, H, W)
                ).view(P, B, 4, -1)
            else:
                c3_out = self.model_cfg.conv_channels[2]
                x_in = obs_rot.permute(1, 2, 0, 3, 4, 5).reshape(B * 4, P * C, H, W)
                x_conv = self._run_conv(x_in)
                feat_dim = c3_out * x_conv.shape[2] * x_conv.shape[3]
                feat_rot = x_conv.view(B, 4, P, feat_dim).permute(2, 0, 1, 3)

            feat_mean = feat_rot.mean(dim=2)
            inv_parts = [feat_mean]
            if self.model_cfg.use_prev_action:
                inv_parts.append(prev_actions[:, :, 4:5])
            inv_x = torch.cat(inv_parts, dim=-1) if len(inv_parts) > 1 else inv_parts[0]
            h = self._act(torch.bmm(inv_x, self.fc_in_w) + self.fc_in_b)
            next_h = self._gru_step(h, hidden)

            dir_parts = [feat_rot, next_h.unsqueeze(2).expand(-1, -1, 4, -1)]
            if self.has_comm:
                comm_ctx = self._run_comm(next_h, team_size=team_size)
                dir_parts.append(comm_ctx.unsqueeze(2).expand(-1, -1, 4, -1))
                stay_in = torch.cat([next_h, comm_ctx], dim=-1)
            else:
                stay_in = next_h
            if self.model_cfg.use_prev_action:
                dir_parts.append(rotate_prev_action_c4(prev_actions))
            if self.model_cfg.use_agent_phase:
                dir_parts.append(rotate_agent_phase_c4(agent_phase))
            dir_in = torch.cat(dir_parts, dim=-1).reshape(P, B * 4, -1)
            dir_h = self._act(torch.bmm(dir_in, self.dir_fc_w) + self.dir_fc_b)
            dir_logits = (torch.bmm(dir_h, self.dir_head_w) + self.dir_head_b).view(P, B, 4)
            stay_logit = torch.bmm(stay_in, self.stay_head_w) + self.stay_head_b
            return torch.cat([dir_logits, stay_logit], dim=-1), next_h

        if self.is_transformer:
            x = self._run_transformer_views(obs_chw)
        else:
            c3_out = self.model_cfg.conv_channels[2]
            x = obs_chw.permute(1, 0, 2, 3, 4).reshape(B, P * C, H, W)
            x = self._run_conv(x)
            x = x.view(B, P, c3_out * x.shape[2] * x.shape[3]).transpose(0, 1)

        parts = [x]
        if self.model_cfg.use_prev_action:
            parts.append(prev_actions)
        if self.model_cfg.use_agent_phase:
            parts.append(agent_phase)
        cat = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        h = self._act(torch.bmm(cat, self.fc_in_w) + self.fc_in_b)

        if self.is_recurrent:
            assert hidden is not None
            next_h = self._gru_step(h, hidden)
            logits = torch.bmm(next_h, self.head_w) + self.head_b
            return logits, next_h

        logits = torch.bmm(h, self.head_w) + self.head_b
        return logits, None


class GPUPopulationEvaluator:
    """Ultra-fast GPU evaluator running all genomes x episodes concurrently on CUDA."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        model_cfg: ModelConfig,
        catch_bonus_weight: float = 0.0,
        overlap_penalty_weight: float = 0.0,
        flank_bonus_weight: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.catch_bonus_weight = catch_bonus_weight
        self.overlap_penalty_weight = overlap_penalty_weight
        self.flank_bonus_weight = flank_bonus_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._sim_cache: dict[tuple[int, int], VectorizedPursuitGPU] = {}
        self._policy_cache: dict[int, BatchedPopulationPolicyGPU] = {}

        phase_np = build_agent_phase_features(env_cfg.n_pursuers)
        self._phase_base = torch.from_numpy(phase_np).to(self.device)

    def _get_sim(self, pop_size: int, n_eps: int) -> VectorizedPursuitGPU:
        key = (pop_size, n_eps)
        if key not in self._sim_cache:
            self._sim_cache[key] = VectorizedPursuitGPU(
                batch_size=pop_size * n_eps,
                env_cfg=self.env_cfg,
                device=self.device,
            )
        return self._sim_cache[key]

    @torch.inference_mode()
    def evaluate_population(
        self,
        genomes: np.ndarray,
        episode_seeds: list[int],
    ) -> PopulationEvaluation:
        P = len(genomes)
        E = len(episode_seeds)
        A = self.env_cfg.n_pursuers
        B_per_g = E * A

        if P not in self._policy_cache:
            self._policy_cache[P] = BatchedPopulationPolicyGPU(
                genomes, self.model_cfg, self.env_cfg, self.device
            )
        else:
            self._policy_cache[P].update_genomes(genomes)

        batched_policy = self._policy_cache[P]
        sim = self._get_sim(P, E)
        sim.reset_from_seeds(episode_seeds, replicate_count=P)

        # Common random evader moves and action-sampling draws across genomes per episode seed
        ev_gen = torch.Generator(device=self.device)
        ev_gen.manual_seed(sum(int(s) for s in episode_seeds) + 31337)
        act_gen = torch.Generator(device=self.device)
        act_gen.manual_seed(sum(int(s) for s in episode_seeds) + 99991)

        prev_actions = torch.zeros(P, B_per_g, 5, dtype=torch.float32, device=self.device)
        phase_t = (
            self._phase_base.unsqueeze(0)
            .unsqueeze(0)
            .expand(P, E, A, 2)
            .reshape(P, B_per_g, 2)
        )
        hidden = (
            torch.zeros(P, B_per_g, self.model_cfg.gru_dim, dtype=torch.float32, device=self.device)
            if self.model_cfg.arch in (
                "cnn_gru",
                "eq_cnn_gru",
                "transformer",
                "eq_transformer",
                "eq_comm_transformer",
            )
            else None
        )

        total_team_returns = torch.zeros(P * E, dtype=torch.float32, device=self.device)
        total_overlaps = torch.zeros(P * E, dtype=torch.float32, device=self.device)
        total_flanks = torch.zeros(P * E, dtype=torch.float32, device=self.device)
        temp = max(self.model_cfg.temperature, 1e-4)

        for step_idx in range(self.env_cfg.max_cycles):
            if (step_idx & 15) == 0 and step_idx > 0 and torch.all(sim.done):
                break
            obs = sim.get_preprocessed_observations()  # (P*E, A, 3, H, W)
            obs_5d = obs.reshape(P, B_per_g, 3, self.env_cfg.obs_range, self.env_cfg.obs_range)

            logits, hidden = batched_policy.forward(
                obs_5d, prev_actions, phase_t, hidden, team_size=A
            )

            if self.model_cfg.stochastic_eval:
                probs = torch.softmax(logits / temp, dim=-1).reshape(P * B_per_g, 5)
                # Shared CRN noise across genomes or direct multinomial
                actions_flat = torch.multinomial(probs, num_samples=1, generator=act_gen).squeeze(1)
                actions_p = actions_flat.reshape(P, B_per_g)
            else:
                actions_p = torch.argmax(logits, dim=-1)

            prev_actions.zero_()
            prev_actions.scatter_(2, actions_p.unsqueeze(-1), 1.0)

            # Common random evader actions shared across genomes for each episode seed
            ev_acts_e = torch.randint(
                0, 5, (1, E, self.env_cfg.n_evaders), generator=ev_gen, device=self.device
            )
            ev_acts = ev_acts_e.expand(P, E, self.env_cfg.n_evaders).reshape(
                P * E, self.env_cfg.n_evaders
            )

            rew, _ = sim.step(actions_p.reshape(P * E, A), evader_actions=ev_acts)
            total_team_returns += rew.sum(dim=1)
            if self.overlap_penalty_weight != 0.0:
                total_overlaps += sim.last_overlaps
            if self.flank_bonus_weight != 0.0:
                total_flanks += sim.last_flank_score

        evaders_caught = (self.env_cfg.n_evaders - sim.ev_alive.sum(dim=1)).to(torch.float32)
        cycles = sim.cycles.to(torch.float32)
        cleared = (sim.ev_alive.sum(dim=1) == 0).to(torch.float32)

        team_ret_pe = total_team_returns.reshape(P, E).mean(dim=1)
        mean_agent_ret_pe = team_ret_pe / float(A)
        caught_pe = evaders_caught.reshape(P, E).mean(dim=1)
        cycles_pe = cycles.reshape(P, E).mean(dim=1)
        clear_pe = cleared.reshape(P, E).mean(dim=1)
        overlaps_pe = (total_overlaps / cycles.clamp(min=1.0)).reshape(P, E).mean(dim=1)
        flanks_pe = (total_flanks / cycles.clamp(min=1.0)).reshape(P, E).mean(dim=1)
        fitness_pe = (
            mean_agent_ret_pe
            + self.catch_bonus_weight * caught_pe
            - self.overlap_penalty_weight * overlaps_pe
            + self.flank_bonus_weight * flanks_pe
        )

        return PopulationEvaluation(
            fitnesses=fitness_pe.cpu().numpy().astype(np.float32),
            mean_agent_returns=mean_agent_ret_pe.cpu().numpy().astype(np.float32),
            team_returns=team_ret_pe.cpu().numpy().astype(np.float32),
            evaders_caught=caught_pe.cpu().numpy().astype(np.float32),
            cycles=cycles_pe.cpu().numpy().astype(np.float32),
            clear_rates=clear_pe.cpu().numpy().astype(np.float32),
        )

    def close(self) -> None:
        self._sim_cache.clear()
        self._policy_cache.clear()

    def __enter__(self) -> "GPUPopulationEvaluator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
