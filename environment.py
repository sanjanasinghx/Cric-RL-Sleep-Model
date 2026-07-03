"""Hopfield brain environment for the sleep as optimal policy experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_NAMES = ("AWAKE", "NREM", "REM")
AWAKE, NREM, REM = 0, 1, 2


@dataclass(frozen=True)
class Config:
    """All hyperparameters for the experiment in one seeded place."""

    seed: int = 0

    n_neurons: int = 100
    k_sweeps: int = 10
    hebb_scale: float = 1.0 / 100.0
    p_eval: int = 8
    m_probe: int = 64
    theta_match: float = 0.95
    theta_recall: float = 0.95
    recall_noise_frac: float = 0.15

    # The weight matrix is split into a consolidated store for the protected eval
    # memories and a labile store for transient experience. AWAKE writes the labile
    # store, REM unlearns it, the eval memories are never touched. This keeps the
    # repeated learn and unlearn cycle stable.
    w_eval: float = 2.0
    w_exp: float = 1.5
    rho_rem: float = 0.3

    e_max: float = 100.0
    c_awake: float = 6.0
    r_nrem: float = 12.0
    c_rem: float = 2.0
    b_unlearn: int = 6
    epsilon: float = 0.08 / 100.0

    r_wake: float = 1.0
    lambda_overload: float = 1.0
    c_idle: float = 0.1
    terminal_penalty: float = 100.0
    s2_max: float = 0.9

    max_steps: int = 300

    hidden_sizes: tuple[int, int] = (128, 128)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    n_steps: int = 20

    total_env_steps: int = 600_000

    # Scripted schedule for the mechanism demonstration figure only.
    k_load: int = 15
    load_per_awake: int = 1
    k_rem: int = 15

    @property
    def n_actions(self) -> int:
        return len(ACTION_NAMES)


def sign(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, 1.0, -1.0).astype(np.float32)


class SleepBrainEnv(gym.Env):
    """Battery plus Hopfield memory managed by AWAKE, NREM, and REM actions."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.n = cfg.n_neurons
        self.action_space = spaces.Discrete(cfg.n_actions)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)

        # Fixed banks so overload and recall stay comparable across the episode.
        self.eval_patterns = self._random_patterns(np.random.default_rng(cfg.seed + 10_000), cfg.p_eval)
        self.probe_states = self._random_patterns(np.random.default_rng(cfg.seed + 20_000), cfg.m_probe)
        flip_rng = np.random.default_rng(cfg.seed + 30_000)
        n_flip = int(round(cfg.recall_noise_frac * self.n))
        self.recall_flip_idx = np.stack(
            [flip_rng.choice(self.n, size=n_flip, replace=False) for _ in range(cfg.p_eval)]
        )

        self.t_consolidated = np.zeros((self.n, self.n), dtype=np.float32)
        self.t_labile = np.zeros((self.n, self.n), dtype=np.float32)
        self.s1 = cfg.e_max
        self.s2 = 0.0
        self.num_experiences = 0
        self.steps = 0
        self.rng = np.random.default_rng(cfg.seed)

    def _random_patterns(self, rng: np.random.Generator, count: int) -> np.ndarray:
        return sign(rng.integers(0, 2, size=(count, self.n)) * 2 - 1)

    def _weights(self) -> np.ndarray:
        return self.t_consolidated + self.t_labile

    def _relax(self, states: np.ndarray) -> np.ndarray:
        weights = self._weights()
        s = states.astype(np.float32, copy=True)
        for _ in range(self.cfg.k_sweeps):
            s = sign(s @ weights)
        return s

    def _measure_overload(self) -> float:
        # A probe is spurious if no consolidated memory, or its mirror image, is
        # within theta_match of where the probe settles. The fraction of spurious
        # probes is the overload s2.
        fixed_points = self._relax(self.probe_states)
        best = (np.abs(fixed_points @ self.eval_patterns.T) / self.n).max(axis=1)
        return float((best < self.cfg.theta_match).mean())

    def _measure_recall(self) -> float:
        cues = self.eval_patterns.copy()
        rows = np.arange(self.cfg.p_eval)[:, None]
        cues[rows, self.recall_flip_idx] *= -1.0
        recovered = self._relax(cues)
        overlaps = (recovered * self.eval_patterns).sum(axis=1) / self.n
        return float((overlaps > self.cfg.theta_recall).mean())

    def _obs(self) -> np.ndarray:
        return np.array([self.s1 / self.cfg.e_max, self.s2], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        self.t_consolidated = np.zeros((self.n, self.n), dtype=np.float32)
        for mu in self.eval_patterns:
            self.t_consolidated += self.cfg.w_eval * self.cfg.hebb_scale * np.outer(mu, mu).astype(np.float32)
        np.fill_diagonal(self.t_consolidated, 0.0)
        self.t_labile = np.zeros((self.n, self.n), dtype=np.float32)
        self.s1 = self.cfg.e_max
        self.num_experiences = 0
        self.steps = 0
        self.s2 = self._measure_overload()
        return self._obs(), self._info(None, self._measure_recall())

    def step(self, action: int):
        cfg = self.cfg
        action = int(action)
        if action == AWAKE:
            self.imprint_experience()
            self.s1 -= cfg.c_awake
        elif action == NREM:
            self.s1 = min(self.s1 + cfg.r_nrem, cfg.e_max)
        elif action == REM:
            self.unlearn()
            self.s1 -= cfg.c_rem
        else:
            raise ValueError(f"invalid action {action}")

        self.s2 = self._measure_overload()
        recall = self._measure_recall()
        self.steps += 1

        reward = cfg.r_wake if action == AWAKE else 0.0
        reward -= cfg.lambda_overload * self.s2
        if action in (NREM, REM):
            reward -= cfg.c_idle

        dead_energy = self.s1 <= 0.0
        dead_overload = self.s2 >= cfg.s2_max
        terminated = bool(dead_energy or dead_overload)
        if terminated:
            reward -= cfg.terminal_penalty
        truncated = bool(self.steps >= cfg.max_steps)

        info = self._info(action, recall)
        if dead_energy:
            info["terminal_cause"] = "energy"
        elif dead_overload:
            info["terminal_cause"] = "overload"
        elif truncated:
            info["terminal_cause"] = "survived"
        return self._obs(), float(reward), terminated, truncated, info

    def imprint_experience(self) -> None:
        """Hebbian write of one random experience into the labile store."""
        mu = self._random_patterns(self.rng, 1)[0]
        self.t_labile += self.cfg.w_exp * self.cfg.hebb_scale * np.outer(mu, mu).astype(np.float32)
        np.fill_diagonal(self.t_labile, 0.0)
        self.num_experiences += 1

    def unlearn(self) -> None:
        """REM reverse learning: relax random noise into the broad spurious basins,
        then raise their energy with a reverse Hebbian update. The labile decay keeps
        the repeated unlearning bounded."""
        cfg = self.cfg
        x = self._relax(self._random_patterns(self.rng, cfg.b_unlearn))
        self.t_labile = (1.0 - cfg.rho_rem) * self.t_labile
        self.t_labile -= cfg.epsilon * (x.T @ x).astype(np.float32)
        np.fill_diagonal(self.t_labile, 0.0)

    def _info(self, action: int | None, recall: float) -> dict[str, Any]:
        return {
            "s1": float(self.s1),
            "s2": float(self.s2),
            "recall_accuracy": float(recall),
            "num_patterns_stored": int(self.cfg.p_eval + self.num_experiences),
            "action_name": None if action is None else ACTION_NAMES[action],
        }
