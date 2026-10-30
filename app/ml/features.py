"""Stationary daily features shared by the ML model and the rule-based agents.

Every column is a return, a ratio to price or a bounded oscillator, so the same
model works across price levels and instruments. Values at row t use only data
up to and including bar t (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_5", "ret_20", "vol_20", "rsi_14", "macd_hist", "dist_sma50",
    "sma50_slope", "atr_pct", "range_pct", "vol_z20",
]

TRADING_DAYS = 252


def _rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(gain.notna())


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    ret1 = np.log(c).diff()
    sma50 = c.rolling(50).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    vol = df["volume"].astype(float)
    vstd = vol.rolling(20).std()
    out = pd.DataFrame(
        {
            "ret_1": ret1,
            "ret_5": np.log(c / c.shift(5)),
            "ret_20": np.log(c / c.shift(20)),
            "vol_20": ret1.rolling(20).std() * np.sqrt(TRADING_DAYS),
            "rsi_14": _rsi_wilder(c) / 100.0,
            "macd_hist": (macd - macd.ewm(span=9, adjust=False).mean()) / c,
            "dist_sma50": c / sma50 - 1,
            "sma50_slope": sma50 / sma50.shift(10) - 1,
            "atr_pct": _atr(df) / c,
            "range_pct": (df["high"] - df["low"]) / c,
            "vol_z20": ((vol - vol.rolling(20).mean()) / vstd.replace(0, np.nan)).fillna(0.0),
        },
        index=df.index,
    )
    out.loc[out.index[:60], :] = np.nan  # warm-up: longest window (sma50 + slope) must be complete
    return out[FEATURES]


def make_labels(df: pd.DataFrame, horizon: int, k: float = 0.5) -> pd.Series:
    """0 = SELL, 1 = HOLD, 2 = BUY by the forward `horizon`-day log return.

    The threshold scales with recent volatility (k * daily vol * sqrt(horizon)), so a
    'big move' means the same thing for gold and for bitcoin.
    """
    c = df["close"]
    fwd = np.log(c.shift(-horizon) / c)
    daily_vol = np.log(c).diff().rolling(20).std()
    thr = k * daily_vol * np.sqrt(horizon)
    lab = pd.Series(1.0, index=df.index)
    lab[fwd > thr] = 2.0
    lab[fwd < -thr] = 0.0
    lab[fwd.isna() | thr.isna()] = np.nan
    return lab


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    return np.log(df["close"].shift(-horizon) / df["close"])
