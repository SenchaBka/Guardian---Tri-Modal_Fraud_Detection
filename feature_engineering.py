"""
feature_engineering.py
-----------------------
Transforms raw transaction data into the feature vector consumed by the
XGBoost fraud classifier, and computes the four interpretable sub-signals
surfaced in the Numerical Stream API output.

Derived signals (numerical_stream_contract.docx §3.2):
  - amount_anomaly   : Z-score of amount vs 90-day history, sigmoid-normalized
  - velocity_risk    : Rolling 1h/24h/7d transaction count vs baseline
  - pattern_deviation: Isolation Forest anomaly score
  - geo_risk         : Country risk tier + travel distance + impossible-travel flag
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Country risk tiers  (simplified lookup — extend as needed)
# Higher tier = higher inherent risk.  Range mapped to [0, 1].
# ---------------------------------------------------------------------------
COUNTRY_RISK_TIERS: Dict[str, float] = {
    "US": 0.1, "CA": 0.1, "GB": 0.1, "AU": 0.1, "DE": 0.1,
    "FR": 0.15, "JP": 0.1, "NL": 0.1, "SE": 0.1, "CH": 0.1,
    "BR": 0.4,  "MX": 0.35, "NG": 0.7, "UA": 0.5, "RU": 0.65,
    "CN": 0.3,  "IN": 0.25,
}
DEFAULT_COUNTRY_RISK = 0.5  # Unknown country

# Approximate km-per-hour limit for legitimate travel (used for impossible-travel check)
MAX_REALISTIC_SPEED_KMH = 900.0  # Commercial flight upper bound

# Earth radius in km
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(transaction: Dict[str, Any]) -> np.ndarray:
    """
    Transform a raw transaction dict into a fixed-length feature vector
    suitable for XGBoost inference.

    Expected keys in `transaction`:
        amount           (float)   – transaction value
        timestamp        (str|datetime) – UTC ISO 8601
        channel          (str)     – online | phone | in_person | atm
        country          (str)     – ISO 3166-1 alpha-2
        merchant_category(str)     – MCC / label
        account_history  (list)    – list of past transaction dicts (optional)

    Returns
    -------
    np.ndarray of shape (N_FEATURES,) — dtype float32
    """
    amount = float(transaction.get("amount", 0.0))
    ts = _parse_timestamp(transaction.get("timestamp"))
    channel = _encode_channel(transaction.get("channel", ""))
    country_risk = COUNTRY_RISK_TIERS.get(
        str(transaction.get("country", "")).upper(), DEFAULT_COUNTRY_RISK
    )
    mcc_encoded = _encode_mcc(transaction.get("merchant_category", ""))
    history: List[dict] = transaction.get("account_history", [])

    # --- Amount features ---
    amount_log = math.log1p(amount)
    amount_zscore = _compute_amount_zscore(amount, history)
    amount_anomaly = _sigmoid(amount_zscore)

    # --- Velocity features ---
    vel_1h, vel_24h, vel_7d = _velocity_counts(ts, history)
    vel_baseline_1h, vel_baseline_24h, vel_baseline_7d = _velocity_baseline(history)
    velocity_risk = _compute_velocity_risk(
        vel_1h, vel_24h, vel_7d,
        vel_baseline_1h, vel_baseline_24h, vel_baseline_7d
    )

    # --- Time features ---
    hour_of_day = ts.hour / 23.0 if ts else 0.5
    day_of_week = ts.weekday() / 6.0 if ts else 0.5

    # --- Geo features ---
    geo_risk = _compute_geo_risk(transaction, history)

    feature_vector = np.array([
        amount_log,          # 0
        amount_zscore,       # 1
        amount_anomaly,      # 2 (also surfaced as signal)
        vel_1h,              # 3
        vel_24h,             # 4
        vel_7d,              # 5
        velocity_risk,       # 6 (also surfaced as signal)
        hour_of_day,         # 7
        day_of_week,         # 8
        country_risk,        # 9 (component of geo_risk signal)
        geo_risk,            # 10 (also surfaced as signal)
        mcc_encoded,         # 11
        *channel,            # 12-15 (one-hot, 4 dims)
    ], dtype=np.float32)

    return feature_vector


def compute_velocity_features(transaction: Dict[str, Any], history: List[dict]) -> dict:
    """
    Calculate transaction velocity signals over rolling 1h / 24h / 7d windows
    relative to the account's historical baseline.

    Parameters
    ----------
    transaction : current transaction dict (must contain 'timestamp')
    history     : list of past transaction dicts, each with a 'timestamp' key

    Returns
    -------
    dict with keys:
        count_1h, count_24h, count_7d     – raw counts in each window
        baseline_1h, baseline_24h, baseline_7d – historical averages
        velocity_risk                     – normalized composite risk score [0, 1]
    """
    ts = _parse_timestamp(transaction.get("timestamp"))
    count_1h, count_24h, count_7d = _velocity_counts(ts, history)
    baseline_1h, baseline_24h, baseline_7d = _velocity_baseline(history)
    velocity_risk = _compute_velocity_risk(
        count_1h, count_24h, count_7d,
        baseline_1h, baseline_24h, baseline_7d
    )
    return {
        "count_1h": count_1h,
        "count_24h": count_24h,
        "count_7d": count_7d,
        "baseline_1h": baseline_1h,
        "baseline_24h": baseline_24h,
        "baseline_7d": baseline_7d,
        "velocity_risk": velocity_risk,
    }


def compute_signal_scores(
    transaction: Dict[str, Any],
    history: Optional[List[dict]] = None,
    isolation_forest_score: float = 0.5,
) -> Tuple[float, float, float, float]:
    """
    Compute all four sub-signal scores returned in the API response.

    Parameters
    ----------
    transaction            : raw transaction dict
    history                : list of historical transaction dicts (optional)
    isolation_forest_score : pre-computed Isolation Forest anomaly score [0,1]
                             (caller passes this in after model inference)

    Returns
    -------
    (amount_anomaly, velocity_risk, pattern_deviation, geo_risk)
    Each value in [0, 1].
    """
    if history is None:
        history = []

    amount = float(transaction.get("amount", 0.0))
    ts = _parse_timestamp(transaction.get("timestamp"))

    # amount_anomaly
    zscore = _compute_amount_zscore(amount, history)
    amount_anomaly = float(np.clip(_sigmoid(zscore), 0.0, 1.0))

    # velocity_risk
    vel_1h, vel_24h, vel_7d = _velocity_counts(ts, history)
    bl_1h, bl_24h, bl_7d = _velocity_baseline(history)
    velocity_risk = float(np.clip(
        _compute_velocity_risk(vel_1h, vel_24h, vel_7d, bl_1h, bl_24h, bl_7d),
        0.0, 1.0
    ))

    # pattern_deviation — caller supplies Isolation Forest score
    pattern_deviation = float(np.clip(isolation_forest_score, 0.0, 1.0))

    # geo_risk
    geo_risk = float(np.clip(_compute_geo_risk(transaction, history), 0.0, 1.0))

    return amount_anomaly, velocity_risk, pattern_deviation, geo_risk


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: Any) -> Optional[datetime]:
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt
        except ValueError:
            return None
    return None


def _encode_channel(channel: str) -> List[float]:
    """One-hot encode channel into a 4-element list."""
    options = ["online", "phone", "in_person", "atm"]
    return [1.0 if channel.lower() == opt else 0.0 for opt in options]


def _encode_mcc(mcc: str) -> float:
    """
    Simple hash-based MCC encoding to a float in [0, 1].
    Replace with a proper embedding or category lookup in production.
    """
    return (hash(str(mcc)) % 1000) / 1000.0


def _compute_amount_zscore(amount: float, history: List[dict]) -> float:
    """Z-score of the current amount relative to 90-day account history."""
    if not history:
        return 0.0
    amounts = [float(h.get("amount", 0.0)) for h in history if "amount" in h]
    if len(amounts) < 2:
        return 0.0
    mean = np.mean(amounts)
    std = np.std(amounts)
    if std < 1e-6:
        return 0.0
    return (amount - mean) / std


def _sigmoid(x: float) -> float:
    """Sigmoid function mapping any real to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-float(x)))


def _velocity_counts(ts: Optional[datetime], history: List[dict]) -> Tuple[int, int, int]:
    """Count transactions in history within 1h, 24h, and 7d of `ts`."""
    if ts is None or not history:
        return 0, 0, 0

    count_1h = count_24h = count_7d = 0
    for h in history:
        h_ts = _parse_timestamp(h.get("timestamp"))
        if h_ts is None:
            continue
        diff_hours = (ts - h_ts).total_seconds() / 3600.0
        if 0 <= diff_hours <= 1:
            count_1h += 1
        if 0 <= diff_hours <= 24:
            count_24h += 1
        if 0 <= diff_hours <= 168:
            count_7d += 1

    return count_1h, count_24h, count_7d


def _velocity_baseline(history: List[dict]) -> Tuple[float, float, float]:
    """
    Compute average hourly, daily, and weekly transaction counts from history.
    Uses the full history span to derive per-period baselines.
    """
    if len(history) < 2:
        return 1.0, 8.0, 30.0  # safe fallback defaults

    timestamps = [_parse_timestamp(h.get("timestamp")) for h in history]
    timestamps = [t for t in timestamps if t is not None]
    if len(timestamps) < 2:
        return 1.0, 8.0, 30.0

    total_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600.0
    if total_hours < 1:
        total_hours = 1.0

    n = len(timestamps)
    baseline_1h = n / total_hours
    baseline_24h = baseline_1h * 24
    baseline_7d = baseline_1h * 168
    return baseline_1h, baseline_24h, baseline_7d


def _compute_velocity_risk(
    count_1h: int, count_24h: int, count_7d: int,
    baseline_1h: float, baseline_24h: float, baseline_7d: float,
) -> float:
    """
    Compute normalized velocity risk as the maximum deviation ratio
    across windows, capped and mapped to [0, 1].
    """
    def ratio(count: int, baseline: float) -> float:
        if baseline < 1e-6:
            return 0.0
        return count / baseline

    ratios = [
        ratio(count_1h, baseline_1h),
        ratio(count_24h, baseline_24h),
        ratio(count_7d, baseline_7d),
    ]
    max_ratio = max(ratios)
    # Normalise: ratio of 1.0 = baseline (risk 0.5); ratio of 5.0 → risk ~1.0
    risk = _sigmoid((max_ratio - 1.0) * 1.5)
    return float(np.clip(risk, 0.0, 1.0))


def _compute_geo_risk(transaction: Dict[str, Any], history: List[dict]) -> float:
    """
    Composite geographical risk score [0, 1]:
      - Country risk tier
      - Distance from last known transaction location
      - Impossible travel flag
    """
    country = str(transaction.get("country", "")).upper()
    country_risk = COUNTRY_RISK_TIERS.get(country, DEFAULT_COUNTRY_RISK)

    # Distance / impossible-travel (uses lat/lon if available)
    travel_risk = 0.0
    lat = transaction.get("latitude")
    lon = transaction.get("longitude")
    ts = _parse_timestamp(transaction.get("timestamp"))

    if lat is not None and lon is not None and history and ts:
        last = _last_geo_entry(history)
        if last:
            dist_km = _haversine(lat, lon, last["lat"], last["lon"])
            hours_elapsed = (ts - last["ts"]).total_seconds() / 3600.0
            if hours_elapsed > 0:
                speed = dist_km / hours_elapsed
                if speed > MAX_REALISTIC_SPEED_KMH:
                    travel_risk = 1.0  # Impossible travel
                else:
                    # Scale: 0 km = 0 risk, 10 000 km = high risk
                    travel_risk = float(np.clip(dist_km / 10_000.0, 0.0, 1.0))

    # Weighted composite
    geo_risk = 0.4 * country_risk + 0.6 * travel_risk
    return float(np.clip(geo_risk, 0.0, 1.0))


def _last_geo_entry(history: List[dict]) -> Optional[dict]:
    """Return the most recent history entry that has lat/lon."""
    entries = []
    for h in history:
        if h.get("latitude") is not None and h.get("longitude") is not None:
            ts = _parse_timestamp(h.get("timestamp"))
            if ts:
                entries.append({"lat": h["latitude"], "lon": h["longitude"], "ts": ts})
    if not entries:
        return None
    return max(entries, key=lambda e: e["ts"])


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
