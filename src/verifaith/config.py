from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class VerifierConfig(BaseModel):
    """Decision thresholds. Defaults are placeholders: calibrate them on a dev set (see eval/)."""

    support_threshold: float = Field(0.6, ge=0, le=1)
    contradiction_threshold: float = Field(0.7, ge=0, le=1)
    window_size: int = Field(3, ge=1, description="Sentences per evidence window")
    max_candidates: int = Field(8, ge=1, description="Evidence windows checked per claim")
    faithful_threshold: float = Field(0.9, ge=0, le=1)
    unfaithful_threshold: float = Field(0.5, ge=0, le=1)
    extraction_guard_threshold: float = Field(
        0.5, ge=0, le=1, description="Min P(answer entails claim) to trust an extracted claim"
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VERIFAITH_", env_file=".env", extra="ignore")

    extractor: Literal["llm", "sentence"] = "llm"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_timeout_s: float = 30.0
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_batch_size: int = 16

    api_keys: Annotated[list[str], NoDecode] = []
    max_answer_chars: int = 8000
    max_context_chars: int = 60000
    max_batch_items: int = 20

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_keys(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v
