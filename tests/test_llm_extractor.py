import json

import httpx
import pytest

from verifaith.claims import ExtractionError
from verifaith.claims.llm import LLMClaimExtractor


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://llm")


def _ok(claims):
    body = {"choices": [{"message": {"content": json.dumps({"claims": claims})}}]}
    return httpx.Response(200, json=body)


def test_parses_and_caches(monkeypatch):
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        return _ok(["A is B.", " ", "C is D."])

    ex = LLMClaimExtractor("http://llm", "", "m", client=_client(handler))
    assert ex.extract("A is B and C is D.") == ["A is B.", "C is D."]
    assert ex.extract("A is B and C is D.") == ["A is B.", "C is D."]
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0


def test_retries_on_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    responses = iter([httpx.Response(503), _ok(["X."])])
    ex = LLMClaimExtractor("http://llm", "", "m", client=_client(lambda r: next(responses)))
    assert ex.extract("X.") == ["X."]


def test_bad_output_raises_extraction_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    bad = {"choices": [{"message": {"content": "not json"}}]}
    ex = LLMClaimExtractor(
        "http://llm", "", "m", client=_client(lambda r: httpx.Response(200, json=bad))
    )
    with pytest.raises(ExtractionError):
        ex.extract("X.")


def test_does_not_retry_auth_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    n = []

    def handler(req):
        n.append(1)
        return httpx.Response(401)

    ex = LLMClaimExtractor("http://llm", "", "m", client=_client(handler))
    with pytest.raises(ExtractionError):
        ex.extract("X.")
    assert len(n) == 1
