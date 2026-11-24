"""Runtime configuration, read from environment variables only (12-factor: config)."""

import os
from dataclasses import dataclass, field

ALL_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "is_red",
]

# Features of the current champion (XGBoost trained on dataset v1, no is_red).
# Only used when the loaded model carries no input signature.
DEFAULT_MODEL_FEATURES = [f for f in ALL_FEATURES if f != "is_red"]


def _split(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    # "dummy" serves a deterministic placeholder model (local dev, CI smoke tests).
    # "mlflow" resolves the registry alias and loads the approved model.
    model_source: str = "dummy"
    model_name: str = "WineQualityPredictor"
    model_alias: str = "champion"
    model_features: list[str] = field(default_factory=lambda: list(DEFAULT_MODEL_FEATURES))
    database_path: str = "var/app.db"
    session_ttl_hours: int = 24
    admin_token: str | None = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_source=os.getenv("MODEL_SOURCE", "dummy").lower(),
            model_name=os.getenv("MLFLOW_MODEL_NAME", "WineQualityPredictor"),
            model_alias=os.getenv("MLFLOW_MODEL_ALIAS", "champion"),
            model_features=_split(os.getenv("MODEL_FEATURES"), DEFAULT_MODEL_FEATURES),
            database_path=os.getenv("DATABASE_PATH", "var/app.db"),
            session_ttl_hours=int(os.getenv("SESSION_TTL_HOURS", "24")),
            admin_token=os.getenv("ADMIN_TOKEN") or None,
            cors_origins=_split(os.getenv("CORS_ORIGINS"), ["*"]),
        )
