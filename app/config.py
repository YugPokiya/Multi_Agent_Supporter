from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "MACE Market Regime Analyzer"
    app_version: str = "0.2.0"
    store_path: str = "store.json"


settings = Settings()
