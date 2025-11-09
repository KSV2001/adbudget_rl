"""
Reward computation for the ad budget environment.

The reward function models diminishing returns via a hill function on
accumulated adstock, cross channel interactions, pacing penalties and
additive noise. This is factored out of the environment class to aid
testing and reuse.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

def compute_reward(
    spend: np.ndarray,
    adstock_state: np.ndarray,
    channel_eff: np.ndarray,
    cross_matrix: np.ndarray,
    remaining_budget_frac: float,
    day: int,
    max_days: int,
    params: dict,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the environment reward and next adstock state."""
    lam = params["adstock_lambda"]
    adstock_next = lam * adstock_state + spend

    k = params["hill_k"]
    n = params["hill_n"]
    roi_ch = channel_eff * (adstock_next ** n) / (k ** n + adstock_next ** n + 1e-12)

    cross_term = float(spend @ (cross_matrix @ spend))
    base_reward = float(roi_ch.sum() + cross_term)

    ideal_remaining = 1.0 - (day / max_days)
    pacing_pen = -params["budget_penalty_weight"] * (remaining_budget_frac - ideal_remaining) ** 2

    noise = float(rng.normal(0.0, params["noise_sigma"]))

    reward = base_reward + pacing_pen + noise
    return reward, adstock_next, roi_ch
