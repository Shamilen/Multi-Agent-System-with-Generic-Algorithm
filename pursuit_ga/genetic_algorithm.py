"""Genetic Algorithm engine for Deep Multi-Agent Neuroevolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pursuit_ga.config import GAConfig
from pursuit_ga.networks import ParamSlice, PursuitPolicyNet


@dataclass
class GenerationStats:
    """Summary statistics for a single evolutionary generation."""

    generation: int
    best_fitness: float
    elite_validated_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float
    best_return: float
    mean_return: float
    best_evaders_caught: float
    mean_evaders_caught: float
    mean_cycles: float
    mutation_sigma: float
    stagnation_counter: int
    elapsed_sec: float


class GeneticAlgorithm:
    """Population-based Genetic Algorithm with elitism, tournament selection,
    structure-aware crossover, adaptive Gaussian mutation, and stagnation rescue.
    """

    def __init__(
        self,
        policy_factory: Callable[[], PursuitPolicyNet],
        ga_cfg: GAConfig,
        seed: int = 42,
    ) -> None:
        self.cfg = ga_cfg
        self.rng = np.random.default_rng(seed)

        template = policy_factory()
        self.num_params = template.num_params
        self.param_slices: list[ParamSlice] = template.param_slices
        self.mutation_scale_vec = self._build_mutation_scale_vector()

        self.population = np.empty((self.cfg.population_size, self.num_params), dtype=np.float32)
        for i in range(self.cfg.population_size):
            torch_seed = int(self.rng.integers(0, 2**31 - 1))
            import torch

            torch.manual_seed(torch_seed)
            net = policy_factory()
            self.population[i] = net.get_genome()

        self.policy_factory = policy_factory
        self.sigma = float(self.cfg.mutation_sigma)
        self.best_ever_fitness = -float("inf")
        self.best_ever_genome: np.ndarray | None = None
        self.consensus_genome: np.ndarray = self.population[0].copy()
        self._consensus_initialized = False
        self.stagnation_counter = 0

    def _build_mutation_scale_vector(self) -> np.ndarray:
        vec = np.ones(self.num_params, dtype=np.float32)
        if not self.cfg.layer_scaled_mutation:
            return vec
        for sl in self.param_slices:
            vec[sl.start : sl.end] = sl.scale
        return vec

    def _tournament_select(self, fitnesses: np.ndarray) -> int:
        k = min(self.cfg.tournament_size, len(fitnesses))
        candidates = self.rng.choice(len(fitnesses), size=k, replace=False)
        best_idx = candidates[int(np.argmax(fitnesses[candidates]))]
        return int(best_idx)

    def _crossover_neuron(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Layer/filter-aligned crossover preserving intact neurons & conv kernels."""
        child = parent_a.copy()
        for sl in self.param_slices:
            if len(sl.shape) >= 2:
                out_dim = sl.shape[0]
                mask_rows = self.rng.random(out_dim) < 0.5
                a_view = parent_a[sl.start : sl.end].reshape(sl.shape)
                b_view = parent_b[sl.start : sl.end].reshape(sl.shape)
                c_view = child[sl.start : sl.end].reshape(sl.shape)
                c_view[mask_rows] = b_view[mask_rows]
                if not np.any(mask_rows):
                    c_view[:] = a_view[:]
            else:
                mask = self.rng.random(sl.end - sl.start) < 0.5
                child[sl.start : sl.end][mask] = parent_b[sl.start : sl.end][mask]
        return child

    def _crossover_arithmetic(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Convex interpolation between two parent genomes."""
        alpha = float(self.rng.uniform(0.15, 0.85))
        return (alpha * parent_a + (1.0 - alpha) * parent_b).astype(np.float32)

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        mode = self.cfg.crossover_type.lower()
        if mode == "none" or self.rng.random() >= self.cfg.crossover_rate:
            return parent_a.copy()
        if mode == "neuron":
            return self._crossover_neuron(parent_a, parent_b)
        if mode == "arithmetic":
            return self._crossover_arithmetic(parent_a, parent_b)
        if mode == "hybrid":
            if self.rng.random() < 0.5:
                return self._crossover_neuron(parent_a, parent_b)
            return self._crossover_arithmetic(parent_a, parent_b)
        raise ValueError(f"Unknown crossover_type: {self.cfg.crossover_type!r}")

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.cfg.mutation_rate:
            return genome
        noise = self.rng.normal(0.0, 1.0, size=self.num_params).astype(np.float32)
        if self.rng.random() < 0.25:
            sparse_mask = (self.rng.random(self.num_params) < 0.12).astype(np.float32)
            noise = noise * sparse_mask * 2.0
        mutated = genome + (self.sigma * self.mutation_scale_vec) * noise
        return mutated.astype(np.float32)

    def _fresh_immigrant(self) -> np.ndarray:
        import torch

        torch.manual_seed(int(self.rng.integers(0, 2**31 - 1)))
        return self.policy_factory().get_genome()

    def step(
        self,
        fitnesses: np.ndarray,
        elite_override_order: np.ndarray | None = None,
        tracked_val_fitness: float | None = None,
        validated_best_genome: np.ndarray | None = None,
    ) -> np.ndarray:
        """Produce the next generation population given evaluated fitness scores."""
        if len(fitnesses) != self.cfg.population_size:
            raise ValueError(
                f"Expected {self.cfg.population_size} fitnesses, got {len(fitnesses)}"
            )

        sorted_indices = (
            elite_override_order
            if elite_override_order is not None
            else np.argsort(fitnesses)[::-1]
        )
        gen_best_idx = int(sorted_indices[0])
        gen_best_fit = (
            float(tracked_val_fitness)
            if tracked_val_fitness is not None
            else float(fitnesses[gen_best_idx])
        )

        if gen_best_fit > self.best_ever_fitness + 1e-6:
            self.best_ever_fitness = gen_best_fit
            self.best_ever_genome = (
                validated_best_genome.copy()
                if validated_best_genome is not None
                else self.population[gen_best_idx].copy()
            )
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        next_pop = np.empty_like(self.population)
        n_elites = min(self.cfg.elite_count, self.cfg.population_size)
        for slot in range(n_elites):
            next_pop[slot] = self.population[int(sorted_indices[slot])].copy()

        # Always keep the best-validated genome in elite slot 0
        if self.best_ever_genome is not None and n_elites >= 1:
            if not np.any([np.allclose(next_pop[s], self.best_ever_genome) for s in range(n_elites)]):
                next_pop[n_elites - 1] = self.best_ever_genome.copy()

        n_immigrants = min(
            self.cfg.immigrant_count,
            max(0, self.cfg.population_size - n_elites),
        )
        immigrant_start = self.cfg.population_size - n_immigrants

        if self.cfg.ga_mode == "guided" and self.cfg.population_size >= 8:
            raw_order = np.argsort(fitnesses)[::-1]
            if not self._consensus_initialized:
                # Start consensus at best individual from initial random population
                self.consensus_genome = self.population[int(raw_order[0])].copy()
                self._consensus_initialized = True
            else:
                # Multi-parent rank-weighted crossover across top mu survivors
                mu = max(3, self.cfg.population_size // 3)
                top_mu = raw_order[:mu]
                ranks = np.arange(1, mu + 1, dtype=np.float32)
                weights = np.log(mu + 0.5) - np.log(ranks)
                weights /= np.sum(weights)

                elite_center = np.sum(
                    weights[:, None] * self.population[top_mu], axis=0
                ).astype(np.float32)
                step_vec = elite_center - self.consensus_genome
                self.consensus_genome = (
                    self.consensus_genome + self.cfg.consensus_lr * step_vec
                ).astype(np.float32)

            if n_elites >= 3:
                next_pop[n_elites - 2] = self.consensus_genome.copy()
            elif n_elites >= 2:
                next_pop[1] = self.consensus_genome.copy()

            # Allocate ~70% of remaining slots to antithetic mirrored offspring around consensus
            # and ~30% to tournament selection + crossover + mutation
            n_offspring = immigrant_start - n_elites
            n_mirrored_pairs = max(1, int(0.35 * n_offspring))
            cursor = n_elites
            for _ in range(n_mirrored_pairs):
                if cursor + 1 >= immigrant_start:
                    break
                z = self.rng.normal(0.0, 1.0, size=self.num_params).astype(np.float32)
                delta = (self.sigma * self.mutation_scale_vec) * z
                next_pop[cursor] = (self.consensus_genome + delta).astype(np.float32)
                next_pop[cursor + 1] = (self.consensus_genome - delta).astype(np.float32)
                cursor += 2

            while cursor < immigrant_start:
                idx_a = self._tournament_select(fitnesses)
                idx_b = self._tournament_select(fitnesses)
                child = self._crossover(self.population[idx_a], self.population[idx_b])
                next_pop[cursor] = self._mutate(child)
                cursor += 1
        else:
            for slot in range(n_elites, immigrant_start):
                idx_a = self._tournament_select(fitnesses)
                idx_b = self._tournament_select(fitnesses)
                child = self._crossover(self.population[idx_a], self.population[idx_b])
                next_pop[slot] = self._mutate(child)

        for slot in range(immigrant_start, self.cfg.population_size):
            next_pop[slot] = self._fresh_immigrant()

        self.population = next_pop

        # Adaptive sigma update (capped below initial sigma to avoid destructive resets)
        if (
            self.cfg.stagnation_patience > 0
            and self.stagnation_counter >= self.cfg.stagnation_patience
        ):
            self.sigma = min(
                self.cfg.mutation_sigma,
                max(self.sigma * self.cfg.stagnation_boost, self.cfg.mutation_sigma_min * 2.0),
            )
            self.stagnation_counter = 0
        else:
            self.sigma = max(self.cfg.mutation_sigma_min, self.sigma * self.cfg.mutation_decay)

        return self.population
