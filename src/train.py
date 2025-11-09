"""
Training utilities for the ad budget reinforcement learning agent.

This module defines a `train` function that uses the PPO algorithm from
Stable Baselines 3 to learn a policy for the `AdBudgetEnv`. The function
sets up a vectorised environment, applies normalisation, defines a linear
decay schedule for the learning rate and clip range and trains the model
for a configurable number of timesteps. Trained models and normaliser
statistics are saved to disk for later evaluation.
"""

from __future__ import annotations
import os
import json
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import BaseCallback

from .env import AdBudgetEnv
from .config import ENV, TRAIN

class EntropyAnnealCallback(BaseCallback):
    """Linearly anneal the entropy coefficient during training."""
    def __init__(self, start: float, end: float, total_timesteps: int) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.total = total_timesteps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / float(self.total))
        self.model.ent_coef = self.start + (self.end - self.start) * frac
        return True

def make_env(seed: int, env_kwargs: Optional[dict] = None):
    """Utility to create a monitored environment closure for DummyVecEnv."""
    kwargs = {**ENV, **(env_kwargs or {}), "seed": seed}
    def _init():
        env = AdBudgetEnv(**kwargs)
        return Monitor(env)
    return _init

def train(
    env_kwargs: Optional[dict] = None,
    train_kwargs: Optional[dict] = None,
    save_dir: str = "trained_models",
    model_name: str = "ppo_adbudget",
    vec_name: str = "vecnorm.pkl",
) -> None:
    """Train a PPO agent on the AdBudgetEnv."""
    env_cfg = {**ENV, **(env_kwargs or {})}
    train_cfg = {**TRAIN, **(train_kwargs or {})}

    num_envs = train_cfg["num_envs"]
    ep_len = env_cfg["episode_len"]
    n_steps = ep_len * num_envs * train_cfg["n_steps_mult"]

    venv = DummyVecEnv([make_env(i, env_cfg) for i in range(num_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    lr_schedule = get_linear_fn(*train_cfg["learning_rate"], end_fraction=1.0)
    clip_schedule = get_linear_fn(*train_cfg["clip_range"], end_fraction=1.0)

    policy_kwargs = dict(net_arch=train_cfg["net_arch"], ortho_init=train_cfg["ortho_init"])

    model = PPO(
        "MlpPolicy",
        venv,
        device=train_cfg["device"],
        n_steps=n_steps,
        batch_size=train_cfg["batch_size"],
        learning_rate=lr_schedule,
        clip_range=clip_schedule,
        ent_coef=train_cfg["ent_coef"],
        gamma=train_cfg["gamma"],
        gae_lambda=train_cfg["gae_lambda"],
        vf_coef=train_cfg["vf_coef"],
        target_kl=train_cfg["target_kl"],
        max_grad_norm=train_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=train_cfg["seed"],
    )

    cb = EntropyAnnealCallback(
        start=train_cfg["ent_coef"], end=0.001, total_timesteps=train_cfg["total_timesteps"]
    )

    model.learn(total_timesteps=train_cfg["total_timesteps"], callback=cb)

    os.makedirs(save_dir, exist_ok=True)
    venv.save(os.path.join(save_dir, vec_name))
    model.save(os.path.join(save_dir, model_name))

    with open(os.path.join(save_dir, "env_kwargs.json"), "w") as f:
        json.dump(env_cfg, f)


def train_short(steps: int = 5_000) -> str:
    """
    Minimal CPU training step for the UI.
    Trains PPO for a few timesteps and returns a short text.
    """
    try:
        train(train_kwargs={"total_timesteps": steps})
        return f"Trained for {steps} timesteps."
    except Exception as e:
        return f"Training failed: {e}"
