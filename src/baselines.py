"""
Simple baseline policies for ad budget allocation.

These functions provide deterministic or stochastic allocations that can be used
for comparison against a learned agent. The baselines operate on the raw
channel count and do not have access to environment internals.
"""

from __future__ import annotations
import numpy as np

def equal_allocation(n: int) -> np.ndarray:
    """Split budget equally across n channels."""
    return np.ones(n, dtype=np.float32) / float(n)

def random_logits(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return random action logits for n channels from a standard normal distribution."""
    rng = rng or np.random.default_rng()
    return rng.standard_normal(n).astype(np.float32)

def greedy_myopic(a: np.ndarray, b: np.ndarray, c: np.ndarray, total_budget: float) -> np.ndarray:
    """
    Oracle baseline assuming true hill curves are known. Approximates the
    optimal Dirichlet split by sampling random allocations.
    """
    n = len(a)
    best_val, best_s = -1e9, None
    samples = np.random.dirichlet(np.ones(n), size=2048)
    for s in samples:
        spend = s * total_budget
        roi = a * (1 - np.exp(-b * spend))
        val = roi.sum() - (c * spend).sum()
        if val > best_val:
            best_val, best_s = val, s
    return best_s.astype(np.float32)
