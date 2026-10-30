"""Daily OHLCV history for the instruments the decision agents cover.

Providers
  * YahooHistoryProvider  - live data from Yahoo Finance's chart endpoint (needs internet)
  * CsvHistoryProvider    - DATA_DIR/<INSTRUMENT>.csv with timestamp,open,high,low,close,volume
  * SyntheticHistoryProvider - random prices for tests and offline demos ONLY
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

INSTRUMENTS: dict[str, str] = {
    "XAU": "GC=F",       # gold futures
    "XAG": "SI=F",       # silver futures
    "BTC": "BTC-USD",
    "NIFTY": "^NSEI",
    "CRUDE": "CL=F",     # WTI crude futures
}
DOLLAR_INDEX = "DXY"
YAHOO_SYMBOLS = {**INSTRUMENTS, DOLLAR_INDEX: "DX-Y.NYB"}

COLUMNS = ["open", "high", "low", "close", "volume"]


class HistoryProvider(Protocol):
    def get(self, instrument: str) -> pd.DataFrame: ...


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise: DatetimeIndex (UTC, date-normalised), float OHLCV, sorted, deduped."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    ok = (
        df[["open", "high", "low", "close"]].notna().all(axis=1)
        & (df[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9)
        & (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9)
    )
    df = df[ok]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "timestamp"
    return df


class YahooHistoryProvider:
    def __init__(self, range_: str = "5y", interval: str = "1d", timeout: int = 20) -> None:
        self.range = range_
        self.interval = interval
        self.timeout = timeout

    def get(self, instrument: str) -> pd.DataFrame:
        symbol = YAHOO_SYMBOLS[instrument]
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{quote(symbol, safe='')}?range={self.range}&interval={self.interval}"
        )
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 MACE-Market-Regime-Analyzer/1.0"})
        with urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        result = payload["chart"]["result"][0]
        q = result["indicators"]["quote"][0]
        df = pd.DataFrame(
            {c: q.get(c) for c in COLUMNS},
            index=pd.to_datetime(result["timestamp"], unit="s", utc=True),
        )
        return clean_history(df)


class CsvHistoryProvider:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)

    def get(self, instrument: str) -> pd.DataFrame:
        path = self.data_dir / f"{instrument}.csv"
        if not path.exists():
            raise FileNotFoundError(f"no history file {path}")
        return clean_history(pd.read_csv(path, comment="#"))


def synthetic_history(days: int = 1500, seed: int = 0, start_price: float = 100.0,
                      end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Regime-switching random walk. Contains NO real market information."""
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC").normalize()
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    idx = pd.bdate_range(end=end_ts.tz_localize(None), periods=days, tz="UTC")
    price, drift, prev = start_price, 0.0, 0.0
    rows = []
    for _ in idx:
        if rng.random() < 0.02:
            drift = rng.choice([-0.0015, 0.0, 0.0015])
        r = drift + 0.1 * prev + rng.normal(0, 0.012)
        o = price * np.exp(rng.normal(0, 0.003))
        c = o * np.exp(r)
        h = max(o, c) * np.exp(abs(rng.normal(0, 0.006)))
        lo = min(o, c) * np.exp(-abs(rng.normal(0, 0.006)))
        rows.append((o, h, lo, c, float(rng.integers(1_000, 5_000))))
        price, prev = c, r
    df = pd.DataFrame(rows, columns=COLUMNS, index=idx)
    df.index.name = "timestamp"
    return df


class SyntheticHistoryProvider:
    """Deterministic per-instrument random data, clearly NOT real prices."""

    START = {"XAU": 2400.0, "XAG": 30.0, "BTC": 60000.0, "NIFTY": 24000.0, "CRUDE": 75.0, "DXY": 104.0}

    def __init__(self, days: int = 1500, seed: int = 0) -> None:
        self.days = days
        self.seed = seed

    def get(self, instrument: str) -> pd.DataFrame:
        offset = sorted(self.START).index(instrument) if instrument in self.START else 99
        return synthetic_history(self.days, self.seed + offset, self.START.get(instrument, 100.0))


def make_provider(source: str, data_dir: str = "data") -> HistoryProvider:
    if source == "yahoo":
        return YahooHistoryProvider()
    if source == "csv":
        return CsvHistoryProvider(data_dir)
    if source == "synthetic":
        return SyntheticHistoryProvider()
    raise ValueError(f"unknown DATA_SOURCE {source!r} (use yahoo, csv or synthetic)")
