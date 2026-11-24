"""Model loading: resolves the registry alias to a concrete version and serves it.

Two sources:
- dummy  : deterministic placeholder, no network. Used locally and in CI.
- mlflow : resolves `champion` in the MLflow Model Registry (DagsHub) to a version
           number, loads that exact version, and checks it can predict.

The service is ready only after a model has loaded AND answered a reference
prediction. A failed reload keeps the previous model serving.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.config import ALL_FEATURES, Settings

logger = logging.getLogger(__name__)

# Reference wine used to check that a freshly loaded model can predict.
REFERENCE_SAMPLE = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.70,
    "citric_acid": 0.00,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "is_red": 1,
}


@dataclass
class LoadedModel:
    predict: Callable[[pd.DataFrame], Any]
    version: str
    features: list[str]
    source: str
    run_id: str | None = None
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelNotReadyError(RuntimeError):
    pass


# ---------------------------------------------------------------- loaders


class DummyLoader:
    """Placeholder model: a hand-written linear rule. Replace with mlflow in prod."""

    VERSION = "dummy-0"

    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def _predict(df: pd.DataFrame) -> np.ndarray:
        score = (
            5.8
            + 0.35 * (df["alcohol"] - 10.5)
            - 1.2 * (df["volatile_acidity"] - 0.34)
            + 0.6 * (df["sulphates"] - 0.53)
            - 0.002 * (df["total_sulfur_dioxide"] - 115)
        )
        return score.to_numpy(dtype=float)

    def __call__(self) -> LoadedModel:
        return LoadedModel(
            predict=self._predict,
            version=self.VERSION,
            features=list(self.settings.model_features),
            source="dummy",
        )


class MlflowChampionLoader:
    """Resolve `models:/<name>@<alias>` to a version number, then load that version.

    `client` and `load_model` are injectable so tests can use a fake registry.
    Credentials come from MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD env vars.
    """

    def __init__(self, settings: Settings, client=None, load_model=None):
        self.settings = settings
        self._client = client
        self._load_model = load_model

    def _features_from_signature(self, model) -> list[str] | None:
        try:
            schema = model.metadata.get_input_schema()
            return list(schema.input_names()) if schema else None
        except AttributeError:
            return None

    def __call__(self) -> LoadedModel:
        if self._client is None or self._load_model is None:
            import mlflow  # imported lazily: not needed for the dummy source

            self._client = self._client or mlflow.MlflowClient()
            self._load_model = self._load_model or mlflow.pyfunc.load_model

        name, alias = self.settings.model_name, self.settings.model_alias
        mv = self._client.get_model_version_by_alias(name, alias)
        version = str(mv.version)
        logger.info("Alias %s@%s resolved to version %s", name, alias, version)

        model = self._load_model(f"models:/{name}/{version}")
        features = self._features_from_signature(model) or list(self.settings.model_features)
        return LoadedModel(
            predict=model.predict,
            version=version,
            features=features,
            source="mlflow",
            run_id=getattr(mv, "run_id", None),
        )


def build_loader(settings: Settings):
    if settings.model_source == "mlflow":
        return MlflowChampionLoader(settings)
    if settings.model_source == "dummy":
        return DummyLoader(settings)
    raise ValueError(f"Unknown MODEL_SOURCE: {settings.model_source}")


# ---------------------------------------------------------------- manager


class ModelManager:
    def __init__(self, settings: Settings, loader: Callable[[], LoadedModel]):
        self.settings = settings
        self.loader = loader
        self.current: LoadedModel | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.current is not None

    def load(self) -> bool:
        """Load (or reload) the model. Returns True on success.

        On failure the previously loaded model, if any, keeps serving.
        """
        with self._lock:
            try:
                candidate = self.loader()
                unknown = set(candidate.features) - set(ALL_FEATURES)
                if unknown:
                    raise ValueError(f"Model expects unknown features: {sorted(unknown)}")
                self._predict_with(candidate, REFERENCE_SAMPLE)  # readiness check
            except Exception as exc:  # noqa: BLE001 - any failure means not loaded
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Model load failed: %s", self.last_error)
                return False
            self.current = candidate
            self.last_error = None
            logger.info("Serving model version %s (%s)", candidate.version, candidate.source)
            return True

    @staticmethod
    def _predict_with(model: LoadedModel, features: dict) -> float:
        frame = pd.DataFrame([{name: features[name] for name in model.features}])
        value = float(np.asarray(model.predict(frame)).ravel()[0])
        if not np.isfinite(value):
            raise ValueError("Model returned a non-finite prediction")
        return value

    def predict(self, features: dict) -> tuple[float, LoadedModel]:
        model = self.current
        if model is None:
            raise ModelNotReadyError(self.last_error or "Model not loaded")
        return self._predict_with(model, features), model

    def info(self) -> dict:
        model = self.current
        return {
            "ready": model is not None,
            "source": model.source if model else self.settings.model_source,
            "model_name": self.settings.model_name,
            "alias": self.settings.model_alias,
            "version": model.version if model else None,
            "run_id": model.run_id if model else None,
            "features": model.features if model else [],
            "loaded_at": model.loaded_at if model else None,
            "error": self.last_error,
        }
