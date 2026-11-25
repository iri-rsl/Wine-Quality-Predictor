# Wine-Quality-Predictor
Web platform with which the user can interact by inserting inputs about their wine and receiving a quality score predicted by a ML model.

## Web application

`app/` is a FastAPI service that serves both the inference API and the web interface
(`app/static/`: plain HTML/CSS/JS, no build step). At startup it resolves the MLflow alias
`champion` to a concrete version, loads that version, runs a reference prediction, and only then
reports ready.

### Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                   # Windows: copy .env.example .env
uvicorn app.main:app --reload --env-file .env
```

Interface: http://localhost:8000 · API docs: http://localhost:8000/docs

`MODEL_SOURCE=dummy` serves a deterministic placeholder model (no network, used in CI).
`MODEL_SOURCE=mlflow` loads `models:/WineQualityPredictor@champion` from DagsHub using the
`MLFLOW_TRACKING_*` variables.

### Endpoints

- `GET /`: web interface (form, sample wines, result, account and history).
- `GET /health`: liveness, the process is up.
- `GET /ready`: 200 only once the approved model has loaded and answered a reference prediction, 503 otherwise.
- `GET /model-info`: loaded registry version, run ID, features, last load error.
- `POST /predict`: wine features in, quality score out. Stored in the user's history when a bearer token is sent.
- `POST /auth/signup`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`: accounts (bearer tokens).
- `GET /history`, `DELETE /history`: per-user prediction history.
- `POST /admin/reload-model`: re-resolves the alias after a promotion or rollback (`X-Admin-Token` header).
  If the new version fails to load or predict, the previous model keeps serving and the call returns 502.

### Tests

```bash
pytest
ruff check .
```

- `tests/test_predict.py`: valid inputs, output format, numeric output, boundaries, clipping to 0-10.
- `tests/test_validation.py`: missing features, wrong types, out-of-range values, unexpected fields, NaN, malformed JSON.
- `tests/test_model_loading.py`: champion loaded when a newer candidate exists, missing or broken champion, readiness on failure, reload after promotion, failed reload keeps the previous model. Uses a fake registry.
- `tests/test_promotion.py`: candidate passes, fails the quality floor, improves insufficiently, champion unchanged after rejection.
- `tests/test_auth_history.py`: accounts, bad credentials, tokens, history isolation between users.
- `tests/test_web.py`: interface and static files are served, form fields match the API schema.

None of the tests need DagsHub, Azure or a trained model.

---
To reformat the code automatically, run:
```bash
ruff check . --fix
ruff format .
```

## Branches and CI

- `feature/*` branches start from `dev` and merge into `dev` through a reviewed pull request.
- `dev` → `staging` → `main` are promoted through pull requests only.
- `.github/workflows/ci.yml` runs on every pull request into `dev`, `staging` and `main`: job `lint` (ruff) and job `test` (pytest). Both are required status checks on the three branches.
