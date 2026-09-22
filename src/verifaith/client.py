"""Thin SDK for the hosted service.

from verifaith.client import VeriFaithClient
vf = VeriFaithClient("https://verifaith.example.com", api_key="...")
result = vf.evaluate(answer, contexts)
if result.verdict != "faithful": ...
"""

from __future__ import annotations

import httpx

from verifaith.schemas import EvalResult


class VeriFaithError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class VeriFaithClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        client: httpx.Client | None = None,
    ):
        headers = {"X-API-Key": api_key} if api_key else {}
        self._http = client or httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s
        )

    def evaluate(self, answer: str, contexts: list[str], **config) -> EvalResult:
        body = {"answer": answer, "contexts": contexts}
        if config:
            body["config"] = config
        r = self._http.post("/v1/evaluate", json=body)
        if r.status_code != 200:
            detail = (
                r.json().get("detail", r.text)
                if r.headers.get("content-type", "").startswith("application/json")
                else r.text
            )
            raise VeriFaithError(r.status_code, str(detail))
        return EvalResult.model_validate(r.json())

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
