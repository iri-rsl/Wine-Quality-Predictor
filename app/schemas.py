"""Request and response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _feature(ge: float, le: float, description: str):
    return Field(..., ge=ge, le=le, allow_inf_nan=False, description=description)


class WineFeatures(BaseModel):
    """Physicochemical measurements of one wine.

    Bounds are wider than the observed dataset ranges but reject physically
    impossible values. Strict mode rejects strings such as "7.4".
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "example": {
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
        },
    )

    fixed_acidity: float = _feature(3.0, 16.0, "Tartaric acid, g/L")
    volatile_acidity: float = _feature(0.0, 2.0, "Acetic acid, g/L")
    citric_acid: float = _feature(0.0, 2.0, "Citric acid, g/L")
    residual_sugar: float = _feature(0.0, 70.0, "Residual sugar, g/L")
    chlorides: float = _feature(0.0, 1.0, "Sodium chloride, g/L")
    free_sulfur_dioxide: float = _feature(0.0, 300.0, "Free SO2, mg/L")
    total_sulfur_dioxide: float = _feature(0.0, 450.0, "Total SO2, mg/L")
    density: float = _feature(0.98, 1.05, "Density, g/cm3")
    pH: float = _feature(2.5, 4.5, "pH")
    sulphates: float = _feature(0.0, 2.5, "Potassium sulphate, g/L")
    alcohol: float = _feature(7.0, 16.0, "Alcohol, % vol")
    is_red: int = Field(..., ge=0, le=1, description="1 for red wine, 0 for white")

    # Label only: stored in the history, never sent to the model.
    wine_name: str | None = Field(None, max_length=80, description="Optional bottle name, e.g. 'Château X 2019'")

    @field_validator("wine_name")
    @classmethod
    def blank_name_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())  # trim and collapse whitespace, drop newlines/tabs
        return value or None

    @model_validator(mode="after")
    def free_so2_within_total(self):
        if self.free_sulfur_dioxide > self.total_sulfur_dioxide:
            raise ValueError("free_sulfur_dioxide cannot exceed total_sulfur_dioxide")
        return self


class PredictionResponse(BaseModel):
    quality: float = Field(description="Predicted quality score on the 0-10 scale")
    quality_class: int = Field(description="Score rounded to the nearest integer")
    label: str
    model_name: str
    model_version: str
    prediction_id: int | None = None
    wine_name: str | None = None


class ModelInfo(BaseModel):
    ready: bool
    source: str
    model_name: str
    alias: str
    version: str | None
    run_id: str | None
    features: list[str]
    loaded_at: datetime | None
    error: str | None = None


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class HistoryItem(BaseModel):
    id: int
    created_at: datetime
    inputs: dict
    quality: float
    label: str
    model_version: str
    wine_name: str | None = None
