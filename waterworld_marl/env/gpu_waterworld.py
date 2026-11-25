"""PyTorch CUDA/CPU batched vectorized WaterWorld simulator matching PettingZoo physics & sensors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from waterworld_marl.scenarios import WaterworldScenario


class VectorizedWaterworldGPU:
    """Fused batched PyTorch implementation of PettingZoo SISL WaterWorld.

    Maintains bit-level observation equivalence (`< 1e-6`) with PettingZoo
    WaterWorld sensors and uses CUDA Graphs (when on CUDA) to execute all physics,
    collisions, rewards, and 30-ray sensor projections in minimal kernel launches.
    """

    PIXEL_SCALE: float = 750.0
    FPS: float = 15.0
    DT: float = 1.0 / 15.0
    RESPAWN_BANK_STEPS: int = 64

    def __init__(
        self,
        batch_size: int,
        scenario: WaterworldScenario,
        device: torch.device,
        seed: int = 42,
        sensor_range: float = 0.2,
        base_radius: float = 0.015,
        obstacle_radius: float = 0.1,
        pursuer_speed: float = 0.2,
        evader_speed: float = 0.1,
        poison_speed: float = 0.1,
        poison_reward: float = -1.0,
        food_reward: float = 10.0,
        encounter_reward: float = 0.01,
        thrust_penalty: float = -0.5,
        local_ratio: float = 1.0,
        use_cuda_graph: bool = True,
    ) -> None:
        self.batch_size = int(batch_size)
        self.scenario = scenario
        self.device = device
        self.dtype = torch.float32
        self.use_cuda_graph = bool(use_cuda_graph and device.type == "cuda")

        self.n_pursuers = int(scenario.n_pursuers)
        self.n_evaders = int(scenario.n_evaders)
        self.n_poisons = int(scenario.n_poisons)
        self.n_coop = int(scenario.n_coop)
        self._g_n_coop = torch.tensor(self.n_coop, dtype=torch.int32, device=self.device)
        self.n_sensors = int(scenario.n_sensors)
        self.speed_features = bool(scenario.speed_features)
        self.max_cycles = int(scenario.max_cycles)
        self.pursuer_max_accel = float(scenario.pursuer_max_accel)
        self.obs_dim = scenario.obs_dim

        self.sensor_range_px = float(sensor_range) * self.PIXEL_SCALE
        self.r_p = float(base_radius) * self.PIXEL_SCALE
        self.r_e = 2.0 * float(base_radius) * self.PIXEL_SCALE
        self.r_poi = 0.75 * float(base_radius) * self.PIXEL_SCALE
        self.r_obs = float(obstacle_radius) * self.PIXEL_SCALE

        self.v_p_max = float(pursuer_speed) * self.PIXEL_SCALE
        self.v_e_max = float(evader_speed) * self.PIXEL_SCALE
        self.v_poi_max = float(poison_speed) * self.PIXEL_SCALE

        self.poison_reward = float(poison_reward)
        self.food_reward = float(food_reward)
        self.encounter_reward = float(encounter_reward)
        self.thrust_penalty = float(thrust_penalty)
        self.local_ratio = float(local_ratio)

        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(int(seed))

        angles = np.linspace(0.0, 2.0 * np.pi, self.n_sensors + 1, dtype=np.float64)[:-1]
        sensors_np = np.c_[np.cos(angles), np.sin(angles)].astype(np.float32)
        self.sensors = torch.from_numpy(sensors_np).to(device=device)
        self.sensors_t = self.sensors.t().contiguous()
        self.sv = self.sensors * self.sensor_range_px
        self.nz_sv = (self.sv.abs() > 1e-4).view(1, 1, self.n_sensors, 2)
        self.safe_sv = torch.where(self.nz_sv, self.sv.view(1, 1, self.n_sensors, 2), 1.0)

        self.obs_pos = torch.tensor(
            [[0.5 * self.PIXEL_SCALE, 0.5 * self.PIXEL_SCALE]], dtype=self.dtype, device=device
        )
        self.obs_vel = torch.zeros((1, 2), dtype=self.dtype, device=device)

        B, A, E, P = self.batch_size, self.n_pursuers, self.n_evaders, self.n_poisons
        self.all_radii = torch.cat(
            [
                torch.full((A,), self.r_p, dtype=self.dtype, device=device),
                torch.full((E,), self.r_e, dtype=self.dtype, device=device),
                torch.full((P,), self.r_poi, dtype=self.dtype, device=device),
            ]
        ).view(1, A + E + P)

        # Fused entity metadata buffers across [obstacles(1), evaders(E), poisons(P), pursuers(A)]
        self.ent_radii = torch.cat(
            [
                torch.full((1,), self.r_obs, dtype=self.dtype, device=device),
                torch.full((E,), self.r_e, dtype=self.dtype, device=device),
                torch.full((P,), self.r_poi, dtype=self.dtype, device=device),
                torch.full((A,), self.r_p, dtype=self.dtype, device=device),
            ]
        ).view(1, 1, 1 + E + P + A, 1)
        self.ent_radii_sq = self.ent_radii * self.ent_radii
        self.ent_inv_vsum = torch.cat(
            [
                torch.full((1,), 1.0 / self.v_p_max, dtype=self.dtype, device=device),
                torch.full((E,), 1.0 / (self.v_e_max + self.v_p_max), dtype=self.dtype, device=device),
                torch.full((P,), 1.0 / (self.v_poi_max + self.v_p_max), dtype=self.dtype, device=device),
                torch.full((A,), 1.0 / (2.0 * self.v_p_max), dtype=self.dtype, device=device),
            ]
        ).view(1, 1, 1 + E + P + A, 1)

        m_tot_obs = 1 + E + P + A
        self_mask = torch.zeros((1, A, m_tot_obs, 1), dtype=torch.bool, device=device)
        self_mask[0, :, 1 + E + P :, 0] = torch.eye(A, dtype=torch.bool, device=device)
        self.ent_self_mask = self_mask

        self.ep_radii = torch.cat(
            [
                torch.full((E,), self.r_e, dtype=self.dtype, device=device),
                torch.full((P,), self.r_poi, dtype=self.dtype, device=device),
            ]
        ).view(1, 1, E + P, 1)
        self.ep_max_v = torch.cat(
            [
                torch.full((E,), self.v_e_max, dtype=self.dtype, device=device),
                torch.full((P,), self.v_poi_max, dtype=self.dtype, device=device),
            ]
        ).view(1, 1, E + P, 1)

        # Persistent address-stable state buffers (required for CUDA Graph replay).
        self.p_pos = torch.zeros((B, A, 2), dtype=self.dtype, device=device)
        self.p_vel = torch.zeros((B, A, 2), dtype=self.dtype, device=device)
        self.e_pos = torch.zeros((B, E, 2), dtype=self.dtype, device=device)
        self.e_vel = torch.zeros((B, E, 2), dtype=self.dtype, device=device)
        self.poi_pos = torch.zeros((B, P, 2), dtype=self.dtype, device=device)
        self.poi_vel = torch.zeros((B, P, 2), dtype=self.dtype, device=device)
        self.food_obs = torch.zeros((B, A), dtype=self.dtype, device=device)
        self.poison_obs = torch.zeros((B, A), dtype=self.dtype, device=device)
        self.prev_overlap_pe = torch.zeros((B, A, E), dtype=torch.bool, device=device)
        self.ev_counter = torch.zeros((B, E), dtype=torch.int32, device=device)

        self.ep_team_ret = torch.zeros(B, dtype=self.dtype, device=device)
        self.ep_food_events = torch.zeros(B, dtype=self.dtype, device=device)
        self.ep_poison_hits = torch.zeros(B, dtype=self.dtype, device=device)
        self._env_cycles = np.zeros(B, dtype=np.int32)

        # Static graph I/O buffers
        self._g_actions = torch.zeros((B, A, 2), dtype=self.dtype, device=device)
        self._g_bank_pos = torch.zeros((B, E + P, 2), dtype=self.dtype, device=device)
        self._g_bank_vel = torch.zeros((B, E + P, 2), dtype=self.dtype, device=device)
        self._g_rewards = torch.zeros((B, A), dtype=self.dtype, device=device)
        self._g_next_obs = torch.zeros((B, A, self.obs_dim), dtype=self.dtype, device=device)
        self._step_graph: Any = None
        self._warmup_count = 0

        # Pre-sampled respawn bank eliminates per-step RNG overhead.
        self._bank_pos = torch.zeros(
            (self.RESPAWN_BANK_STEPS, B, E + P, 2), dtype=self.dtype, device=device
        )
        self._bank_vel = torch.zeros(
            (self.RESPAWN_BANK_STEPS, B, E + P, 2), dtype=self.dtype, device=device
        )
        self._bank_idx = self.RESPAWN_BANK_STEPS

    def _sample_coords_with_radii(
        self, shape_prefix: tuple[int, ...], obj_radii: torch.Tensor
    ) -> torch.Tensor:
        r_excl = 2.0 * obj_radii + self.r_obs
        r_excl_sq = r_excl * r_excl
        obs_xy = self.obs_pos.view(*([1] * (len(shape_prefix) - 1)), 1, 2)
        coords = (
            torch.rand((*shape_prefix, 2), generator=self.rng, device=self.device, dtype=self.dtype)
            * self.PIXEL_SCALE
        )
        for _ in range(3):
            diff = coords - obs_xy
            bad = (diff * diff).sum(dim=-1, keepdim=True) <= r_excl_sq
            resample = (
                torch.rand(
                    (*shape_prefix, 2), generator=self.rng, device=self.device, dtype=self.dtype
                )
                * self.PIXEL_SCALE
            )
            coords = torch.where(bad, resample, coords)

        diff = coords - obs_xy
        dist = torch.linalg.norm(diff, dim=-1, keepdim=True)
        bad = dist <= r_excl
        unit = diff / torch.clamp(dist, min=1e-6)
        pushed = torch.clamp(
            obs_xy + unit * (r_excl + 2.0), obj_radii, self.PIXEL_SCALE - obj_radii
        )
        return torch.where(bad, pushed, coords)

    def _refill_respawn_bank(self) -> None:
        shape_ep = (self.RESPAWN_BANK_STEPS, self.batch_size, self.n_evaders + self.n_poisons)
        self._bank_pos.copy_(self._sample_coords_with_radii(shape_ep, self.ep_radii))
        self._bank_vel.copy_(
            (
                torch.rand((*shape_ep, 2), generator=self.rng, device=self.device, dtype=self.dtype)
                * 2.0
                - 1.0
            )
            * self.ep_max_v
        )
        self._bank_idx = 0

    def _reset_indices(self, env_mask: torch.Tensor | None = None) -> None:
        B, A, E, P = self.batch_size, self.n_pursuers, self.n_evaders, self.n_poisons
        all_p = self._sample_coords_with_radii((B, A + E + P), self.all_radii.view(1, A + E + P, 1))
        rp, re, rpoi = torch.split(all_p, [A, E, P], dim=1)
        ve = (
            torch.rand((B, E, 2), generator=self.rng, device=self.device, dtype=self.dtype) * 2.0
            - 1.0
        ) * self.v_e_max
        vpoi = (
            torch.rand((B, P, 2), generator=self.rng, device=self.device, dtype=self.dtype) * 2.0
            - 1.0
        ) * self.v_poi_max

        if env_mask is None:
            self.p_pos.copy_(rp)
            self.p_vel.zero_()
            self.e_pos.copy_(re)
            self.e_vel.copy_(ve)
            self.poi_pos.copy_(rpoi)
            self.poi_vel.copy_(vpoi)
            self.food_obs.zero_()
            self.poison_obs.zero_()
            self.prev_overlap_pe.zero_()
            self.ev_counter.zero_()
            self.ep_team_ret.zero_()
            self.ep_food_events.zero_()
            self.ep_poison_hits.zero_()
            self._env_cycles.fill(0)
            self._refill_respawn_bank()
            return

        m3 = env_mask.view(B, 1, 1)
        self.p_pos.copy_(torch.where(m3, rp, self.p_pos))
        self.p_vel.copy_(torch.where(m3, 0.0, self.p_vel))
        self.e_pos.copy_(torch.where(m3, re, self.e_pos))
        self.e_vel.copy_(torch.where(m3, ve, self.e_vel))
        self.poi_pos.copy_(torch.where(m3, rpoi, self.poi_pos))
        self.poi_vel.copy_(torch.where(m3, vpoi, self.poi_vel))
        m2 = env_mask.view(B, 1)
        self.food_obs.copy_(torch.where(m2, 0.0, self.food_obs))
        self.poison_obs.copy_(torch.where(m2, 0.0, self.poison_obs))
        self.prev_overlap_pe.copy_(torch.where(m3, False, self.prev_overlap_pe))
        self.ev_counter.copy_(torch.where(m2, 0, self.ev_counter))
        self.ep_team_ret.copy_(torch.where(env_mask, 0.0, self.ep_team_ret))
        self.ep_food_events.copy_(torch.where(env_mask, 0.0, self.ep_food_events))
        self.ep_poison_hits.copy_(torch.where(env_mask, 0.0, self.ep_poison_hits))

    def reset(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            self.rng.manual_seed(int(seed))
        self._reset_indices(None)
        return self.observe()

    def reset_shared_crn(self, crn_seed: int) -> torch.Tensor:
        """Reset all B environments to the exact same layout and shared respawn sequence (CRN)."""
        B, A, E, P = self.batch_size, self.n_pursuers, self.n_evaders, self.n_poisons
        crn_rng = torch.Generator(device=self.device)
        crn_rng.manual_seed(int(crn_seed))
        saved_rng = self.rng
        self.rng = crn_rng
        try:
            all_p = self._sample_coords_with_radii(
                (1, A + E + P), self.all_radii.view(1, A + E + P, 1)
            )
            rp, re, rpoi = torch.split(all_p, [A, E, P], dim=1)
            ve = (
                torch.rand((1, E, 2), generator=crn_rng, device=self.device, dtype=self.dtype) * 2.0
                - 1.0
            ) * self.v_e_max
            vpoi = (
                torch.rand((1, P, 2), generator=crn_rng, device=self.device, dtype=self.dtype) * 2.0
                - 1.0
            ) * self.v_poi_max

            self.p_pos.copy_(rp.expand(B, -1, -1))
            self.p_vel.zero_()
            self.e_pos.copy_(re.expand(B, -1, -1))
            self.e_vel.copy_(ve.expand(B, -1, -1))
            self.poi_pos.copy_(rpoi.expand(B, -1, -1))
            self.poi_vel.copy_(vpoi.expand(B, -1, -1))
            self.food_obs.zero_()
            self.poison_obs.zero_()
            self.prev_overlap_pe.zero_()
            self.ev_counter.zero_()
            self.ep_team_ret.zero_()
            self.ep_food_events.zero_()
            self.ep_poison_hits.zero_()
            self._env_cycles.fill(0)

            shape_1ep = (self.RESPAWN_BANK_STEPS, 1, E + P)
            bank_p1 = self._sample_coords_with_radii(shape_1ep, self.ep_radii)
            bank_v1 = (
                torch.rand((*shape_1ep, 2), generator=crn_rng, device=self.device, dtype=self.dtype)
                * 2.0
                - 1.0
            ) * self.ep_max_v
            self._bank_pos.copy_(bank_p1.expand(-1, B, -1, -1))
            self._bank_vel.copy_(bank_v1.expand(-1, B, -1, -1))
            self._bank_idx = 0
        finally:
            self.rng = saved_rng
        return self.observe()

    def load_from_waterworld_bases(self, bases: list[Any]) -> torch.Tensor:
        """Copy exact entity positions and velocities from CPU WaterworldBase instances."""
        if len(bases) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} bases, got {len(bases)}")
        p_pos = np.asarray([[p.body.position for p in b.pursuers] for b in bases], dtype=np.float32)
        p_vel = np.asarray([[p.body.velocity for p in b.pursuers] for b in bases], dtype=np.float32)
        e_pos = np.asarray([[e.body.position for e in b.evaders] for b in bases], dtype=np.float32)
        e_vel = np.asarray([[e.body.velocity for e in b.evaders] for b in bases], dtype=np.float32)
        poi_pos = np.asarray([[po.body.position for po in b.poisons] for b in bases], dtype=np.float32)
        poi_vel = np.asarray([[po.body.velocity for po in b.poisons] for b in bases], dtype=np.float32)
        f_obs = np.asarray(
            [[1.0 if p.shape.food_touched_indicator >= 1 else 0.0 for p in b.pursuers] for b in bases],
            dtype=np.float32,
        )
        p_obs = np.asarray(
            [[1.0 if p.shape.poison_indicator >= 1 else 0.0 for p in b.pursuers] for b in bases],
            dtype=np.float32,
        )

        self.p_pos.copy_(torch.from_numpy(p_pos).to(self.device))
        self.p_vel.copy_(torch.from_numpy(p_vel).to(self.device))
        self.e_pos.copy_(torch.from_numpy(e_pos).to(self.device))
        self.e_vel.copy_(torch.from_numpy(e_vel).to(self.device))
        self.poi_pos.copy_(torch.from_numpy(poi_pos).to(self.device))
        self.poi_vel.copy_(torch.from_numpy(poi_vel).to(self.device))
        self.food_obs.copy_(torch.from_numpy(f_obs).to(self.device))
        self.poison_obs.copy_(torch.from_numpy(p_obs).to(self.device))
        ev_cnts = np.asarray([[int(e.shape.counter) for e in b.evaders] for b in bases], dtype=np.int32)
        self.ev_counter.copy_(torch.from_numpy(ev_cnts).to(self.device))
        over_init = torch.cdist(self.p_pos, self.e_pos) <= (self.r_p + self.r_e)
        self.prev_overlap_pe.copy_(over_init & (self.ev_counter.unsqueeze(1) > 0))
        self.ep_team_ret.zero_()
        self.ep_food_events.zero_()
        self.ep_poison_hits.zero_()
        self._env_cycles.fill(0)
        self._refill_respawn_bank()
        return self.observe()

    def _advance_all_bodies(
        self, pos: torch.Tensor, vel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single fused Euler step + wall (e=0.999) + obstacle (e=1.0) bounce for all entities."""
        pos = pos + vel * self.DT

        rad = self.all_radii
        hi = self.PIXEL_SCALE - rad
        px = pos[..., 0]
        py = pos[..., 1]
        vx = vel[..., 0]
        vy = vel[..., 1]

        hit_x_lo = px < rad
        hit_x_hi = px > hi
        hit_y_lo = py < rad
        hit_y_hi = py > hi

        vx = torch.where(hit_x_lo & (vx < 0.0), -0.999 * vx, vx)
        vx = torch.where(hit_x_hi & (vx > 0.0), -0.999 * vx, vx)
        vy = torch.where(hit_y_lo & (vy < 0.0), -0.999 * vy, vy)
        vy = torch.where(hit_y_hi & (vy > 0.0), -0.999 * vy, vy)
        vel = torch.stack([vx, vy], dim=-1)

        px = torch.clamp(
            torch.where(hit_x_hi, 2.0 * hi - px, torch.where(hit_x_lo, 2.0 * rad - px, px)),
            rad,
            hi,
        )
        py = torch.clamp(
            torch.where(hit_y_hi, 2.0 * hi - py, torch.where(hit_y_lo, 2.0 * rad - py, py)),
            rad,
            hi,
        )
        pos = torch.stack([px, py], dim=-1)

        r_sum = self.r_obs + rad.unsqueeze(-1)
        obs_xy = self.obs_pos.view(1, 1, 2)
        d_obs = pos - obs_xy
        dist = torch.linalg.norm(d_obs, dim=-1, keepdim=True)
        hit_obs = dist < r_sum
        normal = d_obs / torch.clamp(dist, min=1e-6)
        vn = (vel * normal).sum(dim=-1, keepdim=True)
        vel = torch.where(hit_obs & (vn < 0.0), vel - 2.0 * vn * normal, vel)
        pos = torch.where(hit_obs, obs_xy + normal * r_sum, pos)
        return pos, vel

    def observe(self) -> torch.Tensor:
        """Compute batched observations (B, A, obs_dim) via fused entity projection."""
        B, A, E, P = self.batch_size, self.n_pursuers, self.n_evaders, self.n_poisons
        obs_p = self.obs_pos.view(1, 1, 2).expand(B, 1, 2)
        obs_v = self.obs_vel.view(1, 1, 2).expand(B, 1, 2)

        ent_pos = torch.cat([obs_p, self.e_pos, self.poi_pos, self.p_pos], dim=1)
        ent_vel = torch.cat([obs_v, self.e_vel, self.poi_vel, self.p_vel], dim=1)

        diff = ent_pos.unsqueeze(1) - self.p_pos.unsqueeze(2)
        dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
        rel_vel = (ent_vel.unsqueeze(1) - self.p_vel.unsqueeze(2)) * self.ent_inv_vsum

        sensor_dists = diff @ self.sensors_t
        sensor_vels = rel_vel @ self.sensors_t

        not_sensed = (
            (sensor_dists < 0.0)
            | ((sensor_dists - self.ent_radii) > self.sensor_range_px)
            | ((dist_sq - sensor_dists * sensor_dists) > self.ent_radii_sq)
            | self.ent_self_mask
        )

        d_all = torch.where(
            not_sensed, 1.0, torch.clamp(sensor_dists / self.sensor_range_px, 0.0, 1.0)
        )
        v_all = torch.where(not_sensed, 0.0, sensor_vels)

        obs_dg, ev_dg, poi_dg, pur_dg = torch.split(d_all, [1, E, P, A], dim=2)
        obs_d = obs_dg.squeeze(2)
        ev_d, ev_idx = ev_dg.min(dim=2)
        poi_d, poi_idx = poi_dg.min(dim=2)

        endpoints = self.p_pos.unsqueeze(2) + self.sv.view(1, 1, self.n_sensors, 2)
        clipped_vec = torch.clamp(endpoints, 0.0, self.PIXEL_SCALE) - self.p_pos.unsqueeze(2)
        min_ratios = torch.where(self.nz_sv, clipped_vec / self.safe_sv, 1.0).amin(dim=-1)
        bar_d = torch.where(min_ratios < (1.0 - 1e-4), min_ratios, 1.0)

        if A > 1:
            pur_d, pur_idx = pur_dg.min(dim=2)
        else:
            pur_d = torch.zeros_like(ev_d)
            pur_idx = None

        food_col = self.food_obs.unsqueeze(-1)
        pois_col = self.poison_obs.unsqueeze(-1)

        if self.speed_features:
            _, ev_vg, poi_vg, pur_vg = torch.split(v_all, [1, E, P, A], dim=2)
            ev_v = torch.gather(ev_vg, dim=2, index=ev_idx.unsqueeze(2)).squeeze(2)
            poi_v = torch.gather(poi_vg, dim=2, index=poi_idx.unsqueeze(2)).squeeze(2)
            if A > 1 and pur_idx is not None:
                pur_v = torch.gather(pur_vg, dim=2, index=pur_idx.unsqueeze(2)).squeeze(2)
            else:
                pur_v = torch.zeros_like(ev_d)
            return torch.cat(
                [obs_d, bar_d, ev_d, ev_v, poi_d, poi_v, pur_d, pur_v, food_col, pois_col],
                dim=-1,
            )

        return torch.cat([obs_d, bar_d, ev_d, poi_d, pur_d, food_col, pois_col], dim=-1)

    def _compute_control_rewards(self, act: torch.Tensor) -> torch.Tensor:
        accel_pen = self.thrust_penalty * torch.linalg.norm(act, dim=-1)
        L = self.local_ratio
        mean_all = accel_pen.mean(dim=-1, keepdim=True)
        ctrl = L * accel_pen + (1.0 - L) * mean_all

        last_only = torch.zeros_like(accel_pen)
        last_only[:, -1] = accel_pen[:, -1]
        return ctrl + L * last_only + (1.0 - L) * last_only.mean(dim=-1, keepdim=True)

    def _step_core_inplace(self) -> None:
        """Static-address physics + collision + reward + sensor kernel for CUDA Graph capture."""
        B, A, E, P = self.batch_size, self.n_pursuers, self.n_evaders, self.n_poisons

        act_scaled = torch.clamp(self._g_actions, -1.0, 1.0) * self.pursuer_max_accel
        thrust = torch.linalg.norm(act_scaled, dim=-1, keepdim=True)
        scale = torch.where(
            thrust > self.pursuer_max_accel,
            self.pursuer_max_accel / torch.clamp(thrust, min=1e-12),
            1.0,
        )
        act = act_scaled * scale

        self.p_vel.copy_(
            torch.clamp(self.p_vel + act * self.PIXEL_SCALE, -self.v_p_max, self.v_p_max)
        )
        ctrl_rewards = self._compute_control_rewards(act)

        all_pos = torch.cat([self.p_pos, self.e_pos, self.poi_pos], dim=1)
        all_vel = torch.cat([self.p_vel, self.e_vel, self.poi_vel], dim=1)
        all_pos, all_vel = self._advance_all_bodies(all_pos, all_vel)
        p_p, e_p, poi_p = torch.split(all_pos, [A, E, P], dim=1)
        p_v, e_v, poi_v = torch.split(all_vel, [A, E, P], dim=1)
        self.p_pos.copy_(p_p)
        self.p_vel.copy_(p_v)
        self.e_pos.copy_(e_p)
        self.e_vel.copy_(e_v)
        self.poi_pos.copy_(poi_p)
        self.poi_vel.copy_(poi_v)

        new_e_pos, new_poi_pos = torch.split(self._g_bank_pos, [E, P], dim=1)
        new_e_vel, new_poi_vel = torch.split(self._g_bank_vel, [E, P], dim=1)

        ep_pos = torch.cat([self.e_pos, self.poi_pos], dim=1)
        dist_a_ep = torch.cdist(self.p_pos, ep_pos)
        dist_pe, dist_pp = torch.split(dist_a_ep, [E, P], dim=2)

        hit_pp = dist_pp <= (self.r_p + self.r_poi)
        any_hit_poi = hit_pp.any(dim=1)
        first_agent = hit_pp.to(torch.int64).argmax(dim=1)
        poison_counts = torch.zeros((B, A), dtype=self.dtype, device=self.device)
        poison_counts.scatter_add_(1, first_agent, any_hit_poi.to(self.dtype))
        poi_mask3 = any_hit_poi.unsqueeze(-1)
        self.poi_pos.copy_(torch.where(poi_mask3, new_poi_pos, self.poi_pos))
        self.poi_vel.copy_(torch.where(poi_mask3, new_poi_vel, self.poi_vel))

        curr_overlap = dist_pe <= (self.r_p + self.r_e)
        entered = curr_overlap & (~self.prev_overlap_pe)
        exited = (~curr_overlap) & self.prev_overlap_pe
        n_ent = entered.sum(dim=1, dtype=torch.int32)
        n_ext = exited.sum(dim=1, dtype=torch.int32)
        cnt_after_begin = self.ev_counter + n_ent
        begin_cap_e = (n_ent > 0) & (cnt_after_begin >= self._g_n_coop)
        sep_cap_e = (n_ext > 0) & (cnt_after_begin >= self._g_n_coop)

        food_hits_agent = (
            (entered & begin_cap_e.unsqueeze(1)).to(self.dtype).sum(dim=2)
            + (exited & sep_cap_e.unsqueeze(1)).to(self.dtype).sum(dim=2)
        )
        ev_mask3 = sep_cap_e.unsqueeze(-1)
        active_overlap = curr_overlap & (~sep_cap_e.unsqueeze(1))
        touched_agent = active_overlap.to(self.dtype).sum(dim=2)
        self.food_obs.copy_((touched_agent >= 1.0).to(self.dtype))
        self.poison_obs.copy_((poison_counts >= 1.0).to(self.dtype))
        self.e_pos.copy_(torch.where(ev_mask3, new_e_pos, self.e_pos))
        self.e_vel.copy_(torch.where(ev_mask3, new_e_vel, self.e_vel))
        self.ev_counter.copy_(torch.where(sep_cap_e, 0, cnt_after_begin - n_ext))
        self.prev_overlap_pe.copy_(curr_overlap)

        behavior_rew = (
            self.food_reward * food_hits_agent
            + self.encounter_reward * touched_agent
            + self.poison_reward * poison_counts
        )
        local_rew = behavior_rew + ctrl_rewards
        rewards = local_rew * self.local_ratio + local_rew.mean(dim=-1, keepdim=True) * (
            1.0 - self.local_ratio
        )
        self._g_rewards.copy_(rewards)
        self.ep_team_ret.add_(rewards.sum(dim=-1))
        self.ep_food_events.add_(food_hits_agent.sum(dim=-1))
        self.ep_poison_hits.add_(poison_counts.sum(dim=-1))
        self._g_next_obs.copy_(self.observe())

    def set_n_coop(self, n_coop: int) -> None:
        """Dynamically update cooperative capture threshold without invalidating CUDA Graph."""
        self.n_coop = int(n_coop)
        self._g_n_coop.fill_(self.n_coop)

    @torch.no_grad()
    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, float]] | None]:
        """Step all batched environments on device with optional CUDA Graph acceleration."""
        if self._bank_idx >= self.RESPAWN_BANK_STEPS:
            self._refill_respawn_bank()

        self._g_actions.copy_(actions)
        self._g_bank_pos.copy_(self._bank_pos[self._bank_idx])
        self._g_bank_vel.copy_(self._bank_vel[self._bank_idx])
        self._bank_idx += 1

        if self.use_cuda_graph and self._step_graph is None:
            if self._warmup_count < 2:
                self._step_core_inplace()
                self._warmup_count += 1
            else:
                # Save pre-capture state, capture graph on side stream without perturbing state twice.
                saved_tensors = [
                    t.clone()
                    for t in (
                        self.p_pos,
                        self.p_vel,
                        self.e_pos,
                        self.e_vel,
                        self.poi_pos,
                        self.poi_vel,
                        self.food_obs,
                        self.poison_obs,
                        self.prev_overlap_pe,
                        self.ev_counter,
                        self.ep_team_ret,
                        self.ep_food_events,
                        self.ep_poison_hits,
                    )
                ]
                s = torch.cuda.Stream(device=self.device)
                s.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(s):
                    self._step_core_inplace()
                torch.cuda.current_stream(self.device).wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._step_core_inplace()
                self._step_graph = g
                # Restore saved state and execute exactly one replay for the actual step.
                for dst, src in zip(
                    (
                        self.p_pos,
                        self.p_vel,
                        self.e_pos,
                        self.e_vel,
                        self.poi_pos,
                        self.poi_vel,
                        self.food_obs,
                        self.poison_obs,
                        self.prev_overlap_pe,
                        self.ev_counter,
                        self.ep_team_ret,
                        self.ep_food_events,
                        self.ep_poison_hits,
                    ),
                    saved_tensors,
                ):
                    dst.copy_(src)
                self._step_graph.replay()
        elif self._step_graph is not None:
            self._step_graph.replay()
        else:
            self._step_core_inplace()

        next_obs = self._g_next_obs.clone()
        rewards = self._g_rewards.clone()

        self._env_cycles += 1
        done_np = self._env_cycles >= self.max_cycles
        dones = torch.from_numpy(done_np).to(device=self.device)

        finished_infos: list[dict[str, float]] | None = None
        if bool(done_np.any()):
            done_indices = np.flatnonzero(done_np)
            rets_np = self.ep_team_ret[dones].cpu().numpy()
            foods_np = self.ep_food_events[dones].cpu().numpy()
            pois_np = self.ep_poison_hits[dones].cpu().numpy()
            finished_infos = [
                {
                    "team_return": float(rets_np[k]),
                    "mean_agent_return": float(rets_np[k]) / max(self.n_pursuers, 1),
                    "food_events": float(foods_np[k]),
                    "poison_hits": float(pois_np[k]),
                }
                for k in range(len(done_indices))
            ]
            self._reset_indices(dones)
            self._env_cycles[done_indices] = 0
            next_obs = torch.where(
                dones.view(self.batch_size, 1, 1), self.observe(), next_obs
            )

        return next_obs, rewards, dones, finished_infos
