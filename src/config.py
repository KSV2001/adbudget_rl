"""
Configuration constants for the RL budget environment and training.

These dictionaries are used throughout the package to define the size of the
environment, training hyper‑parameters and evaluation settings. Adjust these
values to tune the simulation or learning behaviour.

Note: All values are intentionally small and simple to promote rapid
prototyping. Feel free to customise them for your particular application.
"""

# Environment settings
ENV = {
    "n_channels": 4,            # number of advertising channels
    "episode_len": 30,          # number of days in an episode
    "total_budget": 1.0,        # total monthly budget (normalised)
    "drift_std": 0.002,         # standard deviation for underlying dynamics drift
    "attr_noise_std": 0.05,     # observation noise on ROI estimates
    "delay": 0,                 # reward delay in days (0 means immediate)
    "smooth_penalty": 0.2,      # penalty weight for allocation changes between steps
    "overspend_penalty": 2.0,   # penalty weight for exceeding daily spend cap
    "history": 3,               # how many past spend/ROI observations to include in state
    "seed": 0,                  # random seed used to initialise the environment
    "reward_params": {
        "hill_k": 0.03,
        "hill_n": 2.2,
        "adstock_lambda": 0.6,
        "budget_penalty_weight": 0.005,
        "noise_sigma": 0.02,
    },
    "cross_matrix": None,       # let env randomise if None
    "channel_eff": None,
    "action_low": -20.0,
    "action_high": 20.0,
}

# Training settings for PPO
TRAIN = {
    "num_envs": 4,
    "device": "auto",              # "cpu" or "cuda"
    "n_steps_mult": 1,
    "batch_size": 512,
    "total_timesteps": 100_000,
    "learning_rate": (3e-4, 3e-5), # linear decay
    "clip_range": (0.2, 0.1),      # linear decay
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

# Evaluation settings
EVAL = {
    "episodes": 10,
}
