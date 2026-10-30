import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    app_name: str = "MACE Market Regime Analyzer"
    app_version: str = "0.3.0"
    store_path: str = field(default_factory=lambda: _env("STORE_PATH", "store.json"))

    # Market data: yahoo (live, needs internet) | csv (DATA_DIR/<INSTRUMENT>.csv) | synthetic (tests/demo)
    data_source: str = field(default_factory=lambda: _env("DATA_SOURCE", "yahoo"))
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "data"))
    model_dir: str = field(default_factory=lambda: _env("MODEL_DIR", "models"))
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "sqlite:///decisions.db"))

    # Decision settings
    horizon_days: int = field(default_factory=lambda: int(_env("HORIZON_DAYS", "5")))
    risk_capital: float = field(default_factory=lambda: float(_env("RISK_CAPITAL", "100000")))
    risk_pct: float = field(default_factory=lambda: float(_env("RISK_PCT", "0.01")))
    cost_bps: float = field(default_factory=lambda: float(_env("COST_BPS", "5")))
    max_data_age_days: int = field(default_factory=lambda: int(_env("MAX_DATA_AGE_DAYS", "5")))


settings = Settings()
