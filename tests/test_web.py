"""Web interface served by FastAPI."""

import pytest


def test_index_serves_form(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    for marker in ('id="predict-form"', 'id="model-status"', 'id="auth-dialog"', "/static/app.js"):
        assert marker in resp.text


@pytest.mark.parametrize("path,content_type", [("/static/app.js", "javascript"), ("/static/style.css", "text/css")])
def test_static_assets(client, path, content_type):
    resp = client.get(path)
    assert resp.status_code == 200
    assert content_type in resp.headers["content-type"]


def test_frontend_fields_match_api_schema(client):
    """Every feature the API expects has an input in the form (built by app.js)."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]["WineFeatures"]["properties"]
    js = client.get("/static/app.js").text
    for feature in schema:
        assert f'"{feature}"' in js or f"{feature}:" in js, feature


@pytest.mark.parametrize("path", ["/static/missing.js", "/static/../main.py"])
def test_unknown_static_paths_are_404(client, path):
    assert client.get(path).status_code == 404
