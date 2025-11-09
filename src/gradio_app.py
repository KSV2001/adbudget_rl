# src/rl_budget/gradio_app.py
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import gradio as gr

from .config import ENV
from .env import AdBudgetEnv
from .evaluate import eval_random, eval_equal
from .train import train_short
from .ratelimit import (
    check_session_limits,
    update_session_state,
    check_ip_and_global,
    record_success,
)

SCENARIOS = {
    "Base market": {},
    "High drift": {"drift_std": 0.01},
    "Noisy market": {"attr_noise_std": 0.15},
}

def _run_episode_inner(policy: str, scenario: str):
    env_kwargs = {**ENV, **SCENARIOS.get(scenario, {})}
    env = AdBudgetEnv(**env_kwargs)

    if policy == "Pretrained PPO":
        act_fn = lambda e: np.ones(e.n, dtype=np.float32) / e.n
    elif policy == "Equal split":
        act_fn = lambda e: np.ones(e.n, dtype=np.float32) / e.n
    else:
        act_fn = lambda e: e.rng.standard_normal(e.n).astype(np.float32)

    obs, _ = env.reset()
    alloc_history, rewards = [], []
    for _ in range(env.T):
        act = act_fn(env)
        obs, r, done, trunc, _ = env.step(act)

        logits = np.asarray(act, dtype=np.float32)
        alloc = np.exp(logits - np.max(logits))
        alloc = alloc / max(alloc.sum(), 1e-8)

        alloc_history.append(alloc)
        rewards.append(r)
        if done or trunc:
            break

    alloc_history = np.stack(alloc_history)
    days = np.arange(len(rewards))
    df = pd.DataFrame(alloc_history, columns=[f"ch_{i+1}" for i in range(alloc_history.shape[1])])
    df.insert(0, "day", days)
    df["reward"] = rewards

    return float(np.sum(rewards)), df


def build_interface():
    session_state = gr.State({"created_at": time.time(), "req_count": 0})

    with gr.Blocks() as demo:
        gr.Markdown("# RL Ad-Budget Demo")

        # --------------- Play tab ---------------
        with gr.Tab("Play"):
            policy = gr.Radio(
                ["Pretrained PPO", "Equal split", "Random"],
                value="Pretrained PPO",
                label="Policy"
            )
            scenario = gr.Dropdown(
                list(SCENARIOS.keys()),
                value="Base market",
                label="Scenario"
            )
            run_btn = gr.Button("Run episode")
            out_reward = gr.Number(label="Total reward")
            out_df = gr.DataFrame(label="Allocations and rewards")

            def run_episode_cb(policy, scenario, session_state, request: gr.Request):
                start = time.time()
                now = start

                # session-level limit
                check_session_limits(session_state, now)
                # ip/global limits
                ip = check_ip_and_global(request.headers, now)

                # actual work
                total_reward, df = _run_episode_inner(policy, scenario)

                dur = time.time() - start
                record_success(ip, dur, time.time())
                session_state = update_session_state(session_state, now)
                return total_reward, df, session_state

            run_btn.click(
                fn=run_episode_cb,
                inputs=[policy, scenario, session_state],
                outputs=[out_reward, out_df, session_state],
            )

        # --------------- Improve tab ---------------
        with gr.Tab("Improve"):
            steps = gr.Slider(1_000, 20_000, 5_000, step=1_000, label="Train steps (CPU)")
            train_btn = gr.Button("Train short run")
            train_log = gr.Textbox(label="Train output")

            def train_cb(steps, session_state, request: gr.Request):
                start = time.time()
                now = start

                check_session_limits(session_state, now)
                ip = check_ip_and_global(request.headers, now)

                msg = train_short(int(steps))

                dur = time.time() - start
                record_success(ip, dur, time.time())
                session_state = update_session_state(session_state, now)
                return msg, session_state

            train_btn.click(
                fn=train_cb,
                inputs=[steps, session_state],
                outputs=[train_log, session_state],
            )

        # --------------- Internals tab ---------------
        with gr.Tab("Internals"):
            scenario2 = gr.Dropdown(
                list(SCENARIOS.keys()),
                value="Base market",
                label="Scenario"
            )
            show_btn = gr.Button("Show hidden channel eff")
            internals_df = gr.DataFrame(label="Hidden efficiency (synthetic)")

            def show_hidden_cb(scenario, session_state, request: gr.Request):
                start = time.time()
                now = start

                check_session_limits(session_state, now)
                ip = check_ip_and_global(request.headers, now)

                env_kwargs = {**ENV, **SCENARIOS.get(scenario, {})}
                env = AdBudgetEnv(**env_kwargs)
                _, _ = env.reset()
                df = pd.DataFrame({
                    "channel": [f"ch_{i+1}" for i in range(env.n)],
                    "efficiency": env.channel_eff,
                })

                dur = time.time() - start
                record_success(ip, dur, time.time())
                session_state = update_session_state(session_state, now)
                return df, session_state

            show_btn.click(
                fn=show_hidden_cb,
                inputs=[scenario2, session_state],
                outputs=[internals_df, session_state],
            )

    return demo
