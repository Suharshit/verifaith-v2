"""LLM claim decomposition via any OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict

import httpx
from pydantic import BaseModel, ValidationError

from verifaith.claims.base import ExtractionError

SYSTEM_PROMPT = """You split a text into atomic factual claims for a fact-checking system.

Rules:
1. Copy facts EXACTLY as the text states them. Never correct, update or improve a fact, even if you
   believe it is false. Wrong dates, numbers, names and places must stay wrong.
2. Never add information that is not in the text.
3. One fact per claim. Split "X and Y" into two claims.
4. Make each claim stand alone: replace pronouns ("it", "they", "he") with the entity they refer
   to, using ONLY the text itself.
5. Skip opinions, questions, hedges with no factual content, and filler.

Respond with JSON only: {"claims": ["claim 1", "claim 2"]}"""


class _Claims(BaseModel):
    claims: list[str]


class LLMClaimExtractor:
    source = "llm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        cache_size: int = 1024,
        client: httpx.Client | None = None,
    ):
        if not api_key and client is None:
            raise ValueError("LLM claim extractor needs an API key (VERIFAITH_LLM_API_KEY).")
        self.model = model
        self.max_retries = max_retries
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )
        self._cache: OrderedDict[str, list[str]] = OrderedDict()
        self._cache_size = cache_size

    def extract(self, answer: str) -> list[str]:
        key = hashlib.sha256(f"{self.model}\x00{answer}".encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return list(self._cache[key])
        claims = self._call(answer)
        self._cache[key] = claims
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return list(claims)

    def _call(self, answer: str) -> list[str]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"TEXT:\n{answer}"},
            ],
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.post("/chat/completions", json=payload)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                parsed = _Claims.model_validate(json.loads(content))
                return [c.strip() for c in parsed.claims if c and c.strip()]
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
                ValidationError,
            ) as e:
                last_err = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and status < 500 and status != 429:
                    break  # 4xx other than rate limit: retrying won't help
                if attempt < self.max_retries:
                    time.sleep(0.5 * 2**attempt)
        raise ExtractionError(f"Claim extraction failed: {type(last_err).__name__}") from last_err
