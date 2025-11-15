"""
adbudget_env.py

A self-contained Gymnasium environment for multi-channel advertising
budget allocation, plus configuration dictionaries for environment
and training.

High-level idea
---------------
We simulate a single "campaign month" of T days and N channels.
Each day:

- The agent outputs a vector of REAL-VALUED LOGITS, one per channel.
- We apply softmax to get allocation fractions across channels.
- We multiply these by a fixed daily budget cap (total_budget / T).
  => The agent controls ONLY how to split spend across channels,
     not how much total to spend per day.
- Spend feeds into:
    * Adstock (carry-over), modelling that past spend still has effect.
    * A Hill (S-shaped) response curve with diminishing returns.
    * A cross-channel interaction matrix for synergy / cannibalisation.
    * Budget pacing penalty: reward encourages staying close to
      a linear "ideal burn" line.
    * Smoothness penalty: discourage very abrupt allocation changes.
    * Optional reward delay and observation noise.

This gives a realistic but tractable continuous-control RL setting
for training PPO / RecurrentPPO agents.
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any

import numpy as np
from collections import deque

import gymnasium as gym
from gymnasium.spaces import Box


# ---------------------------------------------------------------------------
# Configuration dictionaries
# ---------------------------------------------------------------------------

#: Default environment settings.
ENV: Dict[str, Any] = {
    "n_channels": 4,            # number of advertising channels
    "episode_len": 30,          # number of days in an episode
    "total_budget": 1.0,        # total monthly budget (normalised)
    "drift_std": 0.002,         # std dev for hidden parameter drift (a, b)
    "attr_noise_std": 0.05,     # observation noise on ROI estimates
    "delay": 0,                 # reward delay in days (0 = immediate)
    "smooth_penalty": 0.2,      # penalty weight for allocation changes
    "overspend_penalty": 2.0,   # penalty weight for exceeding daily cap
    "history": 3,               # how many past spend/ROI observations in state
    "seed": 0,                  # base random seed
    "reward_params": {
        # Hill-curve parameters:
        # ROI ~ eff * A^n / (k^n + A^n)
        "hill_k": 0.03,
        "hill_n": 2.2,

        # Adstock (carry-over) decay factor, 0 < lambda < 1.
        "adstock_lambda": 0.6,

        # Pacing penalty strength for deviating from ideal linear burn.
        "budget_penalty_weight": 0.005,

        # Reward noise magnitude.
        "noise_sigma": 0.02,
    },
    # If None, these are randomised inside the env.
    "cross_matrix": None,
    "channel_eff": None,

    # Action bounds for logits (softmax ignores exact bounds but SB3 needs a Box).
    "action_low": -20.0,
    "action_high": 20.0,
}

#: Default training settings; these are *not* used by the environment itself,
#: but are handy when constructing a PPO training script.
TRAIN: Dict[str, Any] = {
    "num_envs": 4,
    "device": "auto",              # "cpu", "cuda", or "auto"
    "n_steps_mult": 1,             # n_steps = n_steps_mult * episode_len
    "batch_size": 512,
    "total_timesteps": 100_000,
    "learning_rate": (3e-4, 3e-5), # linear decay (start, end)
    "clip_range": (0.2, 0.1),      # linear decay (start, end)
    "ent_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 1.0,
    "target_kl": 0.02,
    "max_grad_norm": 0.5,
    "net_arch": [128, 128],
    "ortho_init": False,
    "seed": 0,
}

#: Simple evaluation config: how many episodes to average over.
EVAL: Dict[str, Any] = {
    "episodes": 10,
}


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(
    spend: np.ndarray,
    adstock_state: np.ndarray,
    channel_eff: np.ndarray,
    cross_matrix: np.ndarray,
    remaining_budget_frac: float,
    day: int,
    max_days: int,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute reward and next adstock state.

    Theory
    ------
    1. Adstock:
       Models carry-over of advertising effects.
       A_{t+1} = lambda * A_t + spend_t

    2. Hill (S-shaped) diminishing returns:
       ROI_i = eff_i * A_i^n / (k^n + A_i^n)
       - k sets the half-saturation point
       - n controls steepness (n > 1 gives sharp saturation)

    3. Cross-channel interactions:
       cross_term = spend^T * C * spend
       where C is a symmetric matrix. Positive off-diagonals model synergy;
       negative values model cannibalisation.

    4. Budget pacing:
       We compare remaining budget fraction to a linear "ideal" remaining curve:
         ideal_remaining = 1 - day / max_days
       and penalise squared deviation.

    5. Reward noise:
       Adds Gaussian noise to reflect measurement uncertainty.
    """

    # 1) Adstock update
    lam = params["adstock_lambda"]
    adstock_next = lam * adstock_state + spend

    # 2) Hill-curve diminishing returns
    k = params["hill_k"]
    n = params["hill_n"]
    roi_ch = channel_eff * (adstock_next ** n) / (k ** n + adstock_next ** n + 1e-12)

    # 3) Cross-channel interaction
    cross_term = float(spend @ (cross_matrix @ spend))

    base_reward = float(roi_ch.sum() + cross_term)

    # 4) Budget pacing penalty
    ideal_remaining = 1.0 - (day / max_days)
    pacing_pen = -params["budget_penalty_weight"] * (remaining_budget_frac - ideal_remaining) ** 2

    # 5) Reward noise
    noise = float(rng.normal(0.0, params["noise_sigma"]))

    reward = base_reward + pacing_pen + noise
    return reward, adstock_next, roi_ch


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AdBudgetEnv(gym.Env):
    """
    Multi-channel advertising budget allocation environment.

    State (observation)
    -------------------
    A 1D float32 vector with:
    [0]      remaining_budget_fraction  = remaining / (total_budget * T)
    [1:1+H*N] flattened spend history   (most recent first)
    [1+H*N:1+2*H*N] flattened ROI history (noisy observations)
    [-1]     time_fraction              = t / T

    Here:
      - N = number of channels
      - H = history length (number of past days stored)

    Action
    ------
    A vector of logits of length N:
        a_t in R^N
    We apply softmax to get allocation fractions across channels:
        alloc = softmax(a_t)
    Daily spend is then:
        daily_cap = total_budget / episode_len
        spend_t = alloc * daily_cap
    So:
      * total spend per day = daily_cap (while budget is available),
      * the agent only controls the SPLIT across channels.

    Episode dynamics
    ----------------
    - Hidden parameters a_i(t), b_i(t) drift over time, modulating channel
      effectiveness. This simulates non-stationary markets (fatigue, seasonality).
    - Adstock, Hill-curve ROI and cross-channel matrix produce a per-day reward.
    - Additional penalties encourage smooth allocations and good budget pacing.
    - Optional reward delay can be added via a FIFO buffer.

    Termination
    -----------
    Episode ends when:
      - t >= T  (we reach the final day), or
      - remaining budget is effectively zero.
    """

    metadata = {"render_modes": []}

    def __init__(self, **env_overrides: Any) -> None:
        super().__init__()

        # Merge defaults with user overrides
        cfg = {**ENV, **env_overrides}

        self.n: int = cfg["n_channels"]
        self.T: int = cfg["episode_len"]
        self.total_budget: float = cfg["total_budget"]
        self.drift_std: float = cfg["drift_std"]
        self.attr_noise_std: float = cfg["attr_noise_std"]
        self.delay: int = cfg["delay"]
        self.smooth_penalty: float = cfg["smooth_penalty"]
        self.overspend_penalty: float = cfg["overspend_penalty"]
        self.H: int = cfg["history"]
        self.reward_params: Dict[str, float] = cfg["reward_params"].copy()
        self.base_channel_eff = cfg["channel_eff"]
        self.cross_matrix = cfg["cross_matrix"]
        self.action_low = cfg["action_low"]
        self.action_high = cfg["action_high"]
        self.seed_val: int = cfg["seed"]

        # Random generator
        self.rng = np.random.default_rng(self.seed_val)

        # Cross-channel matrix construction
        if self.cross_matrix is None:
            # Small random symmetric matrix with weak positive interactions
            C = np.zeros((self.n, self.n), float)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    v = self.rng.normal(0.03, 0.01)
                    C[i, j] = C[j, i] = v
            self.cross_matrix = C
        else:
            self.cross_matrix = np.asarray(self.cross_matrix, float)

        # Action and observation spaces
        self.action_space = Box(
            low=self.action_low,
            high=self.action_high,
            shape=(self.n,),
            dtype=np.float32,
        )
        d_obs = 1 + 2 * self.H * self.n + 1  # remaining, H*spend, H*ROI, time
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(d_obs,),
            dtype=np.float32,
        )

        # Dynamic state vars will be initialised in reset()
        self.reset()

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over a 1D vector."""
        z = x - np.max(x)
        e = np.exp(z)
        return e / np.clip(e.sum(), 1e-8, np.inf)

    def _obs(self) -> np.ndarray:
        """Construct observation vector from internal state."""
        return np.concatenate(
            [
                # Remaining budget fraction over full campaign volume.
                np.array(
                    [self.remaining / (self.total_budget * self.T)],
                    dtype=np.float32,
                ),
                # Spend history (flattened)
                np.concatenate(self.spend_hist[-self.H:]).astype(np.float32),
                # ROI history (flattened, noisy)
                np.concatenate(self.roi_hist[-self.H:]).astype(np.float32),
                # Normalised time index
                np.array([self.t / self.T], dtype=np.float32),
            ]
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment state at the start of an episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Hidden drift parameters; they modulate channel effectiveness.
        self.a = self.rng.uniform(0.5, 1.5, size=self.n)
        self.b = self.rng.uniform(1.0, 4.0, size=self.n)

        # Channel base effectiveness
        if self.base_channel_eff is None:
            self.channel_eff = self.rng.uniform(0.5, 1.2, size=self.n)
        else:
            self.channel_eff = np.asarray(self.base_channel_eff, float)

        # Time and budget
        self.t = 0
        self.remaining = self.total_budget * self.T

        # Historical buffers
        self.prev_alloc = np.ones(self.n) / self.n
        self.spend_hist = [np.zeros(self.n) for _ in range(self.H)]
        self.roi_hist = [np.zeros(self.n) for _ in range(self.H)]
        self.adstock_state = np.zeros(self.n, dtype=float)

        # Delay buffer for rewards
        self.delay_buf = deque(
            [0.0] * max(1, self.delay),
            maxlen=max(1, self.delay),
        )

        return self._obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment by one day with the given action (logits).

        We:
        - convert logits -> allocations via softmax,
        - derive spend under a fixed daily cap,
        - update hidden drift parameters, adstock and budget,
        - compute reward (including penalties),
        - return next observation and termination flags.
        """
        logits = np.asarray(action, dtype=np.float32)
        alloc = self._softmax(logits)

        # Fixed daily cap: the agent always spends daily_cap while budget remains,
        # only splitting across channels.
        daily_cap = self.total_budget / self.T
        step_cap = min(daily_cap, self.remaining)
        spend = alloc * step_cap
        total_spend = float(spend.sum())

        # Drift hidden "market" parameters a, b
        self.a = np.clip(
            self.a + self.rng.normal(0.0, self.drift_std, size=self.n),
            0.2,
            3.0,
        )
        self.b = np.clip(
            self.b + self.rng.normal(0.0, self.drift_std, size=self.n),
            0.5,
            8.0,
        )

        # Effective channel-level ROI scale modulated by a, b drift
        eff = np.clip(
            self.channel_eff
            * (1.0 + 0.15 * (self.a - 1.0))
            / (1.0 + 0.05 * (self.b - 2.0)),
            0.2,
            3.0,
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

        # Penalise rapid shifts in allocation and any overspend
        smooth = float(np.linalg.norm(alloc - self.prev_alloc, ord=2))
        overspend = max(0.0, total_spend - step_cap)
        penalty = self.overspend_penalty * overspend + self.smooth_penalty * smooth
        r_now = float(r_now - penalty)

        # Update histories with noisy ROI observations
        observed_roi = roi_ch + self.rng.normal(
            0.0, self.attr_noise_std, size=self.n
        )
        self.prev_alloc = alloc
        self.spend_hist.append(spend.copy())
        self.spend_hist = self.spend_hist[-self.H:]
        self.roi_hist.append(observed_roi.copy())
        self.roi_hist = self.roi_hist[-self.H:]

        # Budget and time update
        self.remaining = max(0.0, self.remaining - total_spend)
        self.t += 1
        terminated = self.t >= self.T or self.remaining <= 1e-6
        truncated = False

        # Handle delayed rewards if configured
        if self.delay <= 0:
            reward = r_now
        else:
            self.delay_buf.append(r_now)
            reward = float(self.delay_buf.popleft())
            if terminated or truncated:
                reward += float(sum(self.delay_buf))
                self.delay_buf.clear()

        return self._obs(), reward, terminated, truncated, {}
