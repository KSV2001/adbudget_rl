# src/rl_budget/ratelimit.py
from __future__ import annotations
import time
from typing import Dict, Any

# config
SESSION_MAX_REQ = 5
SESSION_MAX_AGE_SEC = 15 * 60  # 15 minutes

IP_MAX_PER_HOUR = 10
IP_MAX_PER_DAY = 20
IP_MAX_ACTIVE_SEC_PER_DAY = 3600  # 1 hour

GLOBAL_MAX_PER_HOUR = 100
GLOBAL_MAX_PER_DAY = 200
GLOBAL_MAX_ACTIVE_SEC_PER_DAY = 6 * 3600  # 6 hours

COST_PER_SEC = 0.001  # $/sec
DAILY_COST_CAP = 2.0
MONTHLY_COST_CAP = 10.0

# in-memory stores
_ip_stats: Dict[str, Dict[str, Any]] = {}
_global_stats: Dict[str, Any] = {
    "hour_count": 0,
    "hour_start": time.time(),
    "day_count": 0,
    "day_start": time.time(),
    "day_active_sec": 0.0,
    "month_cost": 0.0,
    "month_start": time.time(),
}
# per-session is better kept in gradio state; we give helper funcs below.


def _reset_if_window_expired(now: float):
    # hour
    if now - _global_stats["hour_start"] >= 3600.0:
        _global_stats["hour_start"] = now
        _global_stats["hour_count"] = 0
    # day
    if now - _global_stats["day_start"] >= 86400.0:
        _global_stats["day_start"] = now
        _global_stats["day_count"] = 0
        _global_stats["day_active_sec"] = 0.0
    # month (simple 30d window)
    if now - _global_stats["month_start"] >= 30 * 86400.0:
        _global_stats["month_start"] = now
        _global_stats["month_cost"] = 0.0


def _get_ip_from_headers(headers: dict) -> str:
    # try X-Forwarded-For first
    xff = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
    if xff:
        # take first IP
        return xff.split(",")[0].strip()
    # fallback
    return headers.get("host", "unknown")


def check_session_limits(session_state: dict, now: float) -> None:
    """Raise ValueError if session limit exceeded."""
    created = session_state.get("created_at", now)
    count = session_state.get("req_count", 0)
    if now - created > SESSION_MAX_AGE_SEC:
        raise ValueError("Session expired. Refresh to start a new session.")
    if count >= SESSION_MAX_REQ:
        raise ValueError("Session request limit reached.")
    # else ok


def update_session_state(session_state: dict, now: float) -> dict:
    if "created_at" not in session_state:
        session_state["created_at"] = now
    session_state["req_count"] = session_state.get("req_count", 0) + 1
    return session_state


def check_ip_and_global(headers: dict, now: float) -> str:
    """Raise ValueError if IP/global limits exceeded. Return ip string."""
    _reset_if_window_expired(now)
    ip = _get_ip_from_headers(headers or {})

    ip_info = _ip_stats.get(ip)
    if ip_info is None:
        ip_info = {
            "hour_count": 0,
            "hour_start": now,
            "day_count": 0,
            "day_start": now,
            "day_active_sec": 0.0,
        }
        _ip_stats[ip] = ip_info

    # reset ip hour
    if now - ip_info["hour_start"] >= 3600.0:
        ip_info["hour_start"] = now
        ip_info["hour_count"] = 0

    # reset ip day
    if now - ip_info["day_start"] >= 86400.0:
        ip_info["day_start"] = now
        ip_info["day_count"] = 0
        ip_info["day_active_sec"] = 0.0

    # check IP limits
    if ip_info["hour_count"] >= IP_MAX_PER_HOUR:
        raise ValueError("IP hourly limit reached.")
    if ip_info["day_count"] >= IP_MAX_PER_DAY:
        raise ValueError("IP daily limit reached.")
    if ip_info["day_active_sec"] >= IP_MAX_ACTIVE_SEC_PER_DAY:
        raise ValueError("IP daily active time limit reached.")

    # check global limits
    if _global_stats["hour_count"] >= GLOBAL_MAX_PER_HOUR:
        raise ValueError("Global hourly limit reached.")
    if _global_stats["day_count"] >= GLOBAL_MAX_PER_DAY:
        raise ValueError("Global daily limit reached.")
    if _global_stats["day_active_sec"] >= GLOBAL_MAX_ACTIVE_SEC_PER_DAY:
        raise ValueError("Global daily active time limit reached.")
    if _global_stats["month_cost"] >= MONTHLY_COST_CAP:
        raise ValueError("Monthly cost cap reached.")
    if ( _global_stats["day_active_sec"] * COST_PER_SEC ) >= DAILY_COST_CAP:
        raise ValueError("Daily cost cap reached.")

    return ip


def record_success(ip: str, duration_sec: float, now: float) -> None:
    # ip stats
    ip_info = _ip_stats[ip]
    ip_info["hour_count"] += 1
    ip_info["day_count"] += 1
    ip_info["day_active_sec"] += duration_sec

    # global stats
    _global_stats["hour_count"] += 1
    _global_stats["day_count"] += 1
    _global_stats["day_active_sec"] += duration_sec
    _global_stats["month_cost"] += duration_sec * COST_PER_SEC
