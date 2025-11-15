# AdBudget RL — Reinforcement Learning Environment for Multi-Channel Advertising Budget Allocation

A fully custom Gymnasium environment that models **real-world multi-channel advertising**, including adstock carry-over, saturation (Hill curves), cross-channel synergy, pacing constraints, non-stationary drift, and attribution noise.  
Supports **PPO**, **Recurrent PPO (LSTM)**, and baseline evaluation.  
Designed for benchmarking RL algorithms on **noisy, nonlinear, delayed, non-stationary** control problems.

---

# 🚀 Quick Usage

### 📘 Interactive Notebook (recommended)

All training and evaluation is demonstrated in: **src/adbudget_full_demo.ipynb**

This notebook includes:

- Equal allocation baseline  
- PPO training  
- Recurrent PPO (LSTM) training  
- Evaluation summary  
- Allocation & cumulative profit plots  

### 📦 Training Artifacts

All models and normalization statistics are automatically saved under: 
**src/assets/
vecnorm.pkl, 
vecnorm_lstm.pkl,
ppo_adbudget.zip,
rppo_adbudget_lstm.zip,
env_kwargs.json**


These are loaded directly by the notebook.

---

# 📄 Detailed Documentation

This section explains the full environment design, equations, modelling assumptions, and agent behavior.

---

# 1. Problem Overview

A marketer runs a **T-day** campaign across **N channels** (e.g. search, social, video).  
The total monthly budget $B$ is subdivided into a fixed daily cap:

$$B_{\text{day}} = \frac{B}{T}$$

The RL agent controls **only the split** of this daily budget, not the amount.

Action logits:

$$a_t \in \mathbb{R}^N$$

Softmax allocation:

$$\pi_t(i) = \frac{e^{a_{t,i}}}{\sum_j e^{a_{t,j}}}$$

Spend per channel:

$$s_{t,i} = B_{\text{day}}\pi_t(i)$$

---

# 2. State / Observation Space

State vector contains:

1. **Remaining budget fraction**  
   $$\frac{B_{\text{remaining}}}{B}$$

2. **Spend history** of last $H$ days  
3. **Noisy ROI history** of last $H$ days  
4. **Normalized time index**  $$\frac{t}{T}$$

Total dimension: $$1 + HN + HN + 1$$

---

# 3. Environment Dynamics

Below is the full system diagram:

            ┌────────────────────────────────────────────┐
            │                RL Agent                    │
            │        outputs logits a_t ∈ ℝ^N            │
            └────────────────────────────────────────────┘
                               │
                               ▼
                      Softmax Allocation
                               │
                               ▼
                  s_t = B_day * softmax(a_t)
                               │
                               ▼
     ┌─────────────────────────────────────────────────────────┐
     │                 Environment Dynamics                    │
     │                                                         │
     │  Adstock:        A_{t+1} = λ A_t + s_t                  │
     │  Saturation:     R_i = eff_i * A_i^n / (k^n + A_i^n)   │
     │  Synergy:        cross = s_tᵀ C s_t                    │
     │  Drift:          a_t, b_t evolve via random walk       │
     │  Noise:          ROI_obs = ROI_true + ε                │
     │  Pacing:         penalty vs ideal remaining budget     │
     │  Smoothness:     penalty on ||π_t - π_{t-1}||          │
     └─────────────────────────────────────────────────────────┘
                               │
                               ▼
                      Reward r_t returned
                               │
                               ▼
                   Next Observation o_{t+1}

Each component is described below.

---

# 3.1 Adstock (Carry-Over Effects)

Adstock models memory of past advertising:

$$A_{t+1,i} = \lambda A_{t,i} + s_{t,i}$$

**Citations**

- Broadbent (1979), "One Way TV Advert Works"  
- Danaher et al., "Advertising Models in Marketing", JMR  

---

# 3.2 Saturation (Hill Curve Response)

Real channels saturate — large spend produces diminishing returns.

$$R_{t,i} = \text{eff}_i \cdot \frac{A_{t,i}^n}{k^n + A_{t,i}^n}$$

- $k$ = half-saturation  
- $n > 1$ = steepness  
- $\text{eff}_i$ = channel effectiveness

**Citations**

- Microsoft / Atlas Research: "Why Advertising Response Curves Matter"  
- Hoban & Bucklin (2015), Journal of Marketing Research  

---

# 3.3 Cross-Channel Synergy

Channels interact synergistically or cannibalize:

$$\text{cross}_t = s_t^\top C s_t$$

- $C_{ij} > 0$: synergy  
- $C_{ij} < 0$: cannibalization  

**Citation**

- Naik & Raman (2003), "Understanding the Impact of Synergy in Multimedia Communications", *Marketing Science*

---

# 3.4 Market Drift (Non-stationarity)

Real markets drift due to competition, seasonality, fatigue.

Hidden parameters follow a random walk:

$$a_{t+1,i} = a_{t,i} + \varepsilon^a_{t,i}, \quad b_{t+1,i} = b_{t,i} + \varepsilon^b_{t,i}$$

These distort effectiveness:

$$\text{eff}_{t,i}^\star = \text{clip}\left(\text{eff}_i \cdot f(a_{t,i}, b_{t,i})\right)$$

Recurrent PPO specifically benefits from this temporal structure.

---

# 3.5 Noisy ROI Observations

The agent never sees true ROI; it sees noisy attribution-style values:

$$\hat{R}_{t,i} = R_{t,i} + \zeta_{t,i}, \qquad \zeta \sim \mathcal{N}(0, \sigma^2)$$

**Citation**

- Lewis & Rao (2015), "The Unfavorable Economics of Measuring the Returns to Advertising", Brookings  

---

# 3.6 Reward Delay

Optional FIFO buffer:

$$r_t = \text{delayed}(r)$$

This models lag between spend and realized ROI.

---

# 3.7 Pacing Penalty (Spend Curvature)

Ideal remaining budget curve:

$$B_{\text{ideal}}(t) = 1 - \frac{t}{T}$$

Penalty term:

$$\text{pacing{\_}pen}_t = -w_{\text{pace}} \left( \frac{B_{\text{remaining}}}{B} - B_{\text{ideal}}(t) \right)^2$$

**Citation**

- Meta (Facebook) Marketing Science: Budget pacing research.

---

# 3.8 Smoothness Penalty (Realistic Behavior)

Avoid abrupt allocation changes:

$$\text{smooth}_t = \|\pi_t - \pi_{t-1}\|_2$$

Reward subtracts:

$$-r_{\text{smooth}} \cdot \text{smooth}_t$$

---

# 4. Final Reward Function

All components combine to:

$$\text{Reward}_t = \underbrace{\sum_i R_{t,i}}_{\text{Channel ROI}} + \underbrace{s_t^\top C s_t}_{\text{Synergy}} + \underbrace{\text{pacing\_pen}_t}_{\text{Budget Pacing}} - \underbrace{\text{smooth\_pen}\,\|\pi_t-\pi_{t-1}\|_2}_{\text{Stability}} + \underbrace{\epsilon}_{\text{Noise}}$$

This produces a **realistic noisy, nonlinear, non-stationary, delayed-feedback RL problem**.

---

# 5. Agents Included

### **5.1 Equal Allocation Baseline**
$$\pi(i) = \frac{1}{N}$$

Converted to logits via:

$$a_i = \log \pi(i)$$

### **5.2 PPO (Feed-forward MLP)**  
Captures instantaneous patterns.

### **5.3 Recurrent PPO (LSTM)**  
Tracks temporal drift and partial observability.

---

# 6. Intended Use

This environment is suitable for:

- Research on RL for **non-stationary control** and **AI for marketing budget allocation** 
- Evaluation of recurrent vs feed-forward agents  
- Simulating realistic digital advertising constraints  
- Teaching pacing, synergy, diminishing returns, lagged feedback

---

# 7. Future Plans

I intend to add these in upcoming versions:
- Choosing the actual fraction of budget per day to allocate (instead of fixed daily budget), given a total budget for the episode
- Currently only efficiency and drift coeffecients are per-channel. I plan to add other coefficients like adstock lag, hill-curve exponent etc per channel as well
- Capturing more non-linear dependencies between channels instead of just linear correlation as done currently

Please support by rating my GitHub.

---

# 8. References (Mostly Non-Paywalled)

- Broadbent, S. (1979). *One Way TV Advert Works.*  
- Microsoft/Atlas Research: *Why Advertising Response Curves Matter.*  
- Hoban & Bucklin (2015). JMR: *Effects of Digital Ads.*  
- Naik & Raman (2003). *Synergy in Multimedia Communication*, Marketing Science.  
- Lewis & Rao (2015). *Economics of Measuring Returns to Advertising*, Brookings.  
- Meta Marketing Science: *Budget pacing publications*.
