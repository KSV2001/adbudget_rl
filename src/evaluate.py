"""
Evaluation helpers for the RL budget agent and baseline policies.

This module provides functions to compute expected returns of various
policies on the `AdBudgetEnv`. It can evaluate random policies,
equal‑allocation baselines and a trained PPO model. Each function returns
the mean and standard deviation of returns over a number of episodes.
"""

from __future__ import annotations
from typing import Callable, Tuple, Optional
import os
import json
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .env import AdBudgetEnv
from .baselines import equal_allocation, random_logits
from .config import ENV, EVAL

def _rollout(env: AdBudgetEnv, act_fn: Callable[[AdBudgetEnv], np.ndarray]) -> float:
    """Run one episode in `env` using `act_fn` to generate action logits."""
    obs, _ = env.reset()
    total = 0.0
    done = False
    truncated = False
    while not (done or truncated):
        action = act_fn(env)
        obs, reward, done, truncated, _ = env.step(action)
        total += float(reward)
    return total

def eval_random(episodes: int = EVAL["episodes"], seed: Optional[int] = None) -> Tuple[float, float]:
    """Evaluate a random policy on the environment."""
    returns = []
    for i in range(episodes):
        env = AdBudgetEnv(**{**ENV, "seed": (seed or 0) + i})
        returns.append(_rollout(env, lambda env: random_logits(env.n, env.rng)))
    return float(np.mean(returns)), float(np.std(returns))

def eval_equal(episodes: int = EVAL["episodes"], seed: Optional[int] = None) -> Tuple[float, float]:
    """Evaluate the equal allocation baseline policy."""
    returns = []
    for i in range(episodes):
        env = AdBudgetEnv(**{**ENV, "seed": (seed or 0) + i})
        returns.append(_rollout(env, lambda env: equal_allocation(env.n)))
        return float(np.mean(returns)), float(np.std(returns))

def eval_model(
    model_path: str,
    vec_path: str,
    env_cfg_path: str,
    episodes: int = EVAL["episodes"],
    device: str = "auto",
) -> Tuple[float, float]:
    """Evaluate a trained PPO model."""
    env_kwargs = json.load(open(env_cfg_path, "r"))
    results = []
    for ep in range(episodes):
        def _make_env() -> AdBudgetEnv:
            kwargs = {**env_kwargs, "seed": 1_000 + ep}
            return AdBudgetEnv(**kwargs)
        venv = DummyVecEnv([_make_env])
        vec = VecNormalize.load(vec_path, venv)
        vec.training = False
        vec.norm_reward = False
        model = PPO.load(model_path, device=device)

        obs = vec.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec.step(action)
            ep_return += float(reward[0])
        results.append(ep_return)
    return float(np.mean(results)), float(np.std(results))
