"""FastAPI entry point.

Run locally:  uvicorn app.main:app --reload --port 8000
Web interface on http://localhost:8000, API docs on http://localhost:8000/docs
"""

import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from app import auth
from app.config import Settings
from app.db import Database
from app.model_loader import ModelManager, ModelNotReadyError, build_loader
from app.schemas import (
    Credentials,
    HistoryItem,
    ModelInfo,
    PredictionResponse,
    TokenResponse,
    WineFeatures,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

bearer = HTTPBearer(auto_error=False)
STATIC_DIR = Path(__file__).resolve().parent / "static"


def quality_label(score: float) -> str:
    if score < 5:
        return "Poor"
    if score < 6:
        return "Average"
    if score < 7:
        return "Good"
    return "Excellent"


def create_app(settings: Settings | None = None, loader=None) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = ModelManager(settings, loader or build_loader(settings))
    db = Database(settings.database_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.load()  # failure leaves the app up but not ready (503 on /ready)
        yield

    app = FastAPI(title="Wine Quality Predictor API", version="0.1.0", lifespan=lifespan)
    app.state.settings, app.state.manager, app.state.db = settings, manager, db
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        # Do not echo raw input back: NaN/inf would make the error response itself invalid JSON.
        errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})

    # ---------------------------------------------------------- auth helpers

    def optional_user(creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        if creds is None:
            return None
        user = db.user_for_token(creds.credentials)
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
        return user

    def required_user(user=Depends(optional_user)):
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
        return user

    # ---------------------------------------------------------- web interface

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    # ---------------------------------------------------------- health

    @app.get("/health", tags=["health"])
    def health():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["health"])
    def ready():
        """Readiness: an approved model is loaded and has answered a reference prediction."""
        if not manager.ready:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, manager.last_error or "Loading")
        return {"status": "ready", "model_version": manager.current.version}

    @app.get("/model-info", response_model=ModelInfo, tags=["model"])
    def model_info():
        return manager.info()

    @app.post("/admin/reload-model", response_model=ModelInfo, tags=["model"])
    def reload_model(x_admin_token: Annotated[str | None, Header()] = None):
        """Re-resolve the alias and load the new version. Called by CT after a promotion."""
        if (
            not settings.admin_token
            or not x_admin_token
            or not secrets.compare_digest(x_admin_token, settings.admin_token)
        ):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Forbidden")
        if not manager.load():
            # previous model keeps serving; report the failure to the caller
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, jsonable_encoder(manager.info()))
        return manager.info()

    # ---------------------------------------------------------- prediction

    @app.post("/predict", response_model=PredictionResponse, tags=["model"])
    def predict(features: WineFeatures, user=Depends(optional_user)):
        inputs = features.model_dump()
        try:
            raw, model = manager.predict(inputs)
        except ModelNotReadyError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        score = round(min(max(raw, 0.0), 10.0), 2)
        label = quality_label(score)
        prediction_id = None
        if user is not None:
            prediction_id = db.add_prediction(user["id"], inputs, score, label, model.version)
        return PredictionResponse(
            quality=score,
            quality_class=int(round(score)),
            label=label,
            model_name=settings.model_name,
            model_version=model.version,
            prediction_id=prediction_id,
        )

    # ---------------------------------------------------------- accounts

    @app.post("/auth/signup", response_model=TokenResponse, status_code=201, tags=["auth"])
    def signup(creds: Credentials):
        user_id = db.create_user(creds.username, auth.hash_password(creds.password))
        if user_id is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "Username already taken")
        token = auth.new_token()
        db.create_session(token, user_id, settings.session_ttl_hours)
        return TokenResponse(access_token=token, username=creds.username)

    @app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
    def login(creds: Credentials):
        user = db.get_user(creds.username)
        if user is None or not auth.verify_password(creds.password, user["password_hash"]):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
        token = auth.new_token()
        db.create_session(token, user["id"], settings.session_ttl_hours)
        return TokenResponse(access_token=token, username=user["username"])

    @app.post("/auth/logout", status_code=204, tags=["auth"])
    def logout(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
        user=Depends(required_user),
    ):
        db.delete_session(creds.credentials)

    @app.get("/auth/me", tags=["auth"])
    def me(user=Depends(required_user)):
        return {"username": user["username"], "created_at": user["created_at"]}

    # ---------------------------------------------------------- history

    @app.get("/history", response_model=list[HistoryItem], tags=["history"])
    def history(user=Depends(required_user), limit: int = Query(50, ge=1, le=500)):
        return db.history(user["id"], limit)

    @app.delete("/history", tags=["history"])
    def clear_history(user=Depends(required_user)):
        return {"deleted": db.clear_history(user["id"])}

    return app


app = create_app()
