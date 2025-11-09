"""
Custom Gymnasium environment for multi‑channel ad budget allocation.

The goal of the agent is to allocate a fixed daily budget across multiple
channels in order to maximise a composite reward signal. The environment
simulates hidden response curves, cross‑channel interactions and pacing
penalties. The state includes remaining budget, past spending and ROI
history and the current day fraction. Actions are unconstrained logits
which are converted to allocation fractions via a softmax.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium.spaces import Box

from .reward import compute_reward

class AdBudgetEnv(gym.Env):
    """A simple vectorised advertising budget environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        n_channels: int,
        episode_len: int,
        total_budget: float,
        drift_std: float,
        attr_noise_std: float,
        delay: int,
        smooth_penalty: float,
        overspend_penalty: float,
        history: int,
        seed: int,
        reward_params: dict,
        cross_matrix: Optional[np.ndarray] = None,
        channel_eff: Optional[np.ndarray] = None,
        action_low: float = -20.0,
        action_high: float = 20.0
    ) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.n, self.T = n_channels, episode_len
        self.total_budget = total_budget
        self.drift_std = drift_std
        self.attr_noise_std = attr_noise_std
        self.delay = delay
        self.smooth_penalty = smooth_penalty
        self.overspend_penalty = overspend_penalty
        self.H = history
        self.reward_params = reward_params.copy()

        # hidden response parameters and baseline efficiencies
        self.a = self.rng.uniform(0.5, 1.5, size=self.n)
        self.b = self.rng.uniform(1.0, 4.0, size=self.n)
        self.base_channel_eff = channel_eff.copy() if channel_eff is not None else None

        # cross‑channel matrix
        if cross_matrix is None:
            C = np.zeros((self.n, self.n), float)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    v = self.rng.normal(0.03, 0.01)
                    C[i, j] = C[j, i] = v
            self.cross_matrix = C
        else:
            self.cross_matrix = np.asarray(cross_matrix, float)

        # action and observation spaces
        self.action_space: gym.Space = Box(low=action_low, high=action_high, shape=(self.n,), dtype=np.float32)
        d_obs = 1 + 2 * self.H * self.n + 1
        self.observation_space: gym.Space = Box(low=-np.inf, high=np.inf, shape=(d_obs,), dtype=np.float32)

        # dynamic state variables initialised in reset()
        self.reset()

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        e = np.exp(z)
        return e / np.clip(e.sum(), 1e-8, np.inf)

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            np.array([self.remaining / (self.total_budget * self.T)], dtype=np.float32),
            np.concatenate(self.spend_hist[-self.H:]).astype(np.float32),
            np.concatenate(self.roi_hist[-self.H:]).astype(np.float32),
            np.array([self.t / self.T], dtype=np.float32),
        ])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.a = self.rng.uniform(0.5, 1.5, size=self.n)
        self.b = self.rng.uniform(1.0, 4.0, size=self.n)
        self.channel_eff = (
            self.base_channel_eff if self.base_channel_eff is not None else
            self.rng.uniform(0.5, 1.2, size=self.n)
        )
        self.t = 0
        self.remaining = self.total_budget * self.T
        self.prev_alloc = np.ones(self.n) / self.n
        self.spend_hist = [np.zeros(self.n) for _ in range(self.H)]
        self.roi_hist = [np.zeros(self.n) for _ in range(self.H)]
        self.adstock_state = np.zeros(self.n, dtype=float)
        self.delay_buf = deque([0.0] * max(1, self.delay), maxlen=max(1, self.delay))
        return self._obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        logits = np.asarray(action, dtype=np.float32)
        alloc = self._softmax(logits)

        daily_cap = self.total_budget / self.T
        step_cap = min(daily_cap, self.remaining)
        spend = alloc * step_cap
        total_spend = float(spend.sum())

        # drift hidden curves
        self.a = np.clip(self.a + self.rng.normal(0.0, self.drift_std, size=self.n), 0.2, 3.0)
        self.b = np.clip(self.b + self.rng.normal(0.0, self.drift_std, size=self.n), 0.5, 8.0)
        eff = np.clip(
            self.channel_eff * (1.0 + 0.15 * (self.a - 1.0)) / (1.0 + 0.05 * (self.b - 2.0)),
            0.2, 3.0,
        )

        remaining_frac = self.remaining / (self.total_budget * self.T)
        r_now, adstock_next, roi_ch = compute_reward(
            spend=spend,
            adstock_state=self.adstock_state,
            channel_eff=eff,
            cross_matrix=self.cross_matrix,
            remaining_budget_frac=remaining_frac,
            day=self.t,
            max_days=self.T,
            params=self.reward_params,
            rng=self.rng,
        )
        self.adstock_state = adstock_next

        smooth = float(np.linalg.norm(alloc - self.prev_alloc, ord=2))
        overspend = max(0.0, total_spend - step_cap)
        penalty = self.overspend_penalty * overspend + self.smooth_penalty * smooth
        r_now = float(r_now - penalty)

        observed_roi = roi_ch + self.rng.normal(0.0, self.attr_noise_std, size=self.n)
        self.prev_alloc = alloc
        self.spend_hist.append(spend.copy()); self.spend_hist = self.spend_hist[-self.H:]
        self.roi_hist.append(observed_roi.copy()); self.roi_hist = self.roi_hist[-self.H:]

        self.remaining = max(0.0, self.remaining - total_spend)
        self.t += 1
        terminated = self.t >= self.T or self.remaining <= 1e-6
        truncated = False

        if self.delay <= 0:
            reward = r_now
        else:
            self.delay_buf.append(r_now)
            reward = float(self.delay_buf.popleft())
            if terminated or truncated:
                reward += float(sum(self.delay_buf))
                self.delay_buf.clear()

        return self._obs(), reward, terminated, truncated, {}
