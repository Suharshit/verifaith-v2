"""HTTP service. Run: uvicorn verifaith.service.app:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from verifaith.claims.base import ExtractionError
from verifaith.config import Settings, VerifierConfig
from verifaith.pipeline import VERSION, Verifier
from verifaith.schemas import EvalResult

log = logging.getLogger("verifaith")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class EvaluateRequest(BaseModel):
    answer: str = Field(min_length=1, description="The generated answer to check")
    contexts: list[str] = Field(min_length=1, description="The documents the answer should rely on")
    config: VerifierConfig | None = Field(None, description="Per-request threshold overrides")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "The Eiffel Tower was completed in 1889 and is 330 metres tall.",
                "contexts": [
                    "The Eiffel Tower is in Paris. It was constructed from 1887 to 1889. "
                    "It stands 330 metres tall."
                ],
            }
        }
    }


class BatchRequest(BaseModel):
    items: list[EvaluateRequest] = Field(min_length=1)


class BatchItemResult(BaseModel):
    ok: bool
    result: EvalResult | None = None
    error: str | None = None


def create_app(verifier: Verifier | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load models once, at startup, not at import time.
        app.state.verifier = verifier or Verifier.from_settings(settings)
        if not settings.api_keys:
            log.warning("VERIFAITH_API_KEYS is empty: authentication is DISABLED (dev only).")
        yield

    app = FastAPI(
        title="VeriFaith",
        version=VERSION,
        lifespan=lifespan,
        description="Claim-level faithfulness verification for RAG answers.",
    )

    def require_key(key: str | None = Security(_api_key_header)) -> str:
        if not settings.api_keys:
            return "anonymous"
        if key and any(hmac.compare_digest(key, k) for k in settings.api_keys):
            return key
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

    def check_limits(req: EvaluateRequest) -> None:
        if len(req.answer) > settings.max_answer_chars:
            raise HTTPException(413, f"answer exceeds {settings.max_answer_chars} characters")
        if sum(len(c) for c in req.contexts) > settings.max_context_chars:
            raise HTTPException(413, f"contexts exceed {settings.max_context_chars} characters")

    def run(v: Verifier, req: EvaluateRequest) -> EvalResult:
        if req.config is None:
            return v.evaluate(req.answer, req.contexts)
        scoped = Verifier(v.extractor, v.nli, v.retriever, req.config)
        return scoped.evaluate(req.answer, req.contexts)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info(
            "%s %s %s %.0fms rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
            rid,
        )
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": VERSION}

    @app.post("/v1/evaluate", response_model=EvalResult)
    def evaluate(req: EvaluateRequest, request: Request, _: str = Depends(require_key)):
        check_limits(req)
        try:
            return run(request.app.state.verifier, req)
        except ExtractionError:
            raise HTTPException(502, "Upstream claim extractor failed; retry later.") from None
        except Exception:
            log.exception("evaluation failed")
            raise HTTPException(500, "Internal error.") from None  # never leak details

    @app.post("/v1/evaluate/batch", response_model=list[BatchItemResult])
    def evaluate_batch(body: BatchRequest, request: Request, _: str = Depends(require_key)):
        if len(body.items) > settings.max_batch_items:
            raise HTTPException(413, f"batch exceeds {settings.max_batch_items} items")
        out = []
        for item in body.items:
            try:
                check_limits(item)
                out.append(BatchItemResult(ok=True, result=run(request.app.state.verifier, item)))
            except HTTPException as e:
                out.append(BatchItemResult(ok=False, error=str(e.detail)))
            except ExtractionError:
                out.append(BatchItemResult(ok=False, error="Upstream claim extractor failed."))
            except Exception:
                log.exception("batch item failed")
                out.append(BatchItemResult(ok=False, error="Internal error."))
        return out

    return app


def __getattr__(name: str):
    # Lazily build the default app so importing this module doesn't load models.
    if name == "app":
        return create_app()
    raise AttributeError(name)
