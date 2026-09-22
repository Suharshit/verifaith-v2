import pytest
from fastapi.testclient import TestClient

from verifaith import Verifier
from verifaith.claims import ExtractionError, SentenceExtractor
from verifaith.client import VeriFaithClient, VeriFaithError
from verifaith.config import Settings
from verifaith.service.app import create_app
from verifaith.testing import KeywordNLI

CTX = ["The Eiffel Tower is in Paris. It stands 330 metres tall."]


@pytest.fixture
def client():
    s = Settings(api_keys=["secret"], max_answer_chars=200, max_batch_items=2, _env_file=None)
    app = create_app(Verifier(SentenceExtractor(), KeywordNLI()), s)
    with TestClient(app) as c:
        yield c


def test_requires_api_key(client):
    r = client.post("/v1/evaluate", json={"answer": "x", "contexts": ["y"]})
    assert r.status_code == 401


def test_evaluate_ok(client):
    r = client.post(
        "/v1/evaluate",
        headers={"X-API-Key": "secret"},
        json={"answer": "The Eiffel Tower is in Paris.", "contexts": CTX},
    )
    assert r.status_code == 200
    assert r.json()["verdict"] == "faithful"
    assert "X-Request-ID" in r.headers


def test_per_request_config_override(client):
    r = client.post(
        "/v1/evaluate",
        headers={"X-API-Key": "secret"},
        json={
            "answer": "The Eiffel Tower is in Paris.",
            "contexts": CTX,
            "config": {"support_threshold": 0.99},
        },
    )
    assert r.json()["claims"][0]["label"] == "unsupported"


def test_limits(client):
    r = client.post(
        "/v1/evaluate", headers={"X-API-Key": "secret"}, json={"answer": "x" * 500, "contexts": CTX}
    )
    assert r.status_code == 413


def test_validation(client):
    r = client.post(
        "/v1/evaluate", headers={"X-API-Key": "secret"}, json={"answer": "x", "contexts": []}
    )
    assert r.status_code == 422


def test_batch_partial_failure(client):
    items = [
        {"answer": "The Eiffel Tower is in Paris.", "contexts": CTX},
        {"answer": "y" * 500, "contexts": CTX},
    ]
    r = client.post("/v1/evaluate/batch", headers={"X-API-Key": "secret"}, json={"items": items})
    body = r.json()
    assert r.status_code == 200 and body[0]["ok"] and not body[1]["ok"]


def test_upstream_failure_is_502_without_leaking():
    class Broken:
        source = "llm"

        def extract(self, answer):
            raise ExtractionError("secret upstream details")

    s = Settings(api_keys=[], _env_file=None)
    with TestClient(create_app(Verifier(Broken(), KeywordNLI()), s)) as c:
        r = c.post("/v1/evaluate", json={"answer": "x", "contexts": ["y"]})
    assert r.status_code == 502 and "secret" not in r.text


def test_sdk_client(client):
    client.headers["X-API-Key"] = "secret"
    vf = VeriFaithClient("http://testserver", client=client)
    res = vf.evaluate("The Eiffel Tower is in Paris.", CTX)
    assert res.verdict == "faithful"
    client.headers["X-API-Key"] = "wrong"
    with pytest.raises(VeriFaithError) as e:
        vf.evaluate("x", CTX)
    assert e.value.status == 401
