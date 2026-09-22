# VeriFaith

**Claim-level faithfulness checks for RAG answers — as a Python library or a drop-in HTTP service.**

Give VeriFaith an answer and the documents it was supposed to be based on. It splits the answer into
atomic claims, finds the passages that bear on each one, and tells you which claims are
**supported**, **contradicted**, or **unsupported** — with the evidence for every decision.

```json
{
  "faithfulness": 0.5,
  "contradiction_rate": 0.5,
  "verdict": "partial",
  "counts": {"supported": 1, "contradicted": 1, "unsupported": 0, "total": 2},
  "claims": [
    {"claim": {"id": 1, "text": "The Eiffel Tower was completed in 1889."},
     "label": "supported", "confidence": 0.97,
     "evidence": {"text": "It was constructed from 1887 to 1889.", "context_index": 0, "start_sentence": 1, "end_sentence": 1}},
    {"claim": {"id": 2, "text": "The Eiffel Tower is 500 metres tall."},
     "label": "contradicted", "confidence": 0.94,
     "evidence": {"text": "... It stands 330 metres tall.", "context_index": 0, "start_sentence": 1, "end_sentence": 2}}
  ],
  "warnings": [],
  "version": "0.1.0"
}
```

## What it's for

- **Guardrail:** check an answer before showing it to a user; block or annotate contradicted claims.
- **Offline evaluation:** score a RAG system over a test set and track faithfulness across releases.
- **Monitoring:** sample production traffic and alert when contradiction rate rises.

Faithfulness here means *every claim in the answer is backed by the given context*. It does **not**
measure completeness (an answer that says less than the sources is still faithful) or real-world
truth (a claim that matches a wrong source is still "supported").

## Quickstart

```bash
pip install -e ".[all,dev]"
cp .env.example .env        # add VERIFAITH_LLM_API_KEY (Groq, OpenAI, or any OpenAI-compatible API)
pytest                      # runs offline with fake models
```

### As a library

```python
from verifaith import Verifier

verifier = Verifier.from_settings()
result = verifier.evaluate(answer, contexts)
print(result.verdict, result.faithfulness)
```

No LLM available? Set `VERIFAITH_EXTRACTOR=sentence` to treat each sentence as a claim.

### As a service

```bash
uvicorn verifaith.service.app:app --port 8000
# or
docker build -t verifaith . && docker run -p 8000:8000 --env-file .env verifaith
```

```bash
curl -X POST localhost:8000/v1/evaluate \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"answer": "...", "contexts": ["..."]}'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/evaluate` | Check one answer. Optional `config` overrides thresholds per request. |
| `POST /v1/evaluate/batch` | Up to `VERIFAITH_MAX_BATCH_ITEMS` answers; per-item success/failure. |
| `GET /healthz` | Liveness. |
| `GET /docs` | Interactive OpenAPI docs. |

### From another project (SDK)

```python
from verifaith.client import VeriFaithClient

with VeriFaithClient("https://your-host", api_key="...") as vf:
    result = vf.evaluate(answer, contexts)
```

See `examples/rag_guardrail.py` for a full guardrail pattern. For your own tests, use
`verifaith.testing.KeywordNLI` and `StaticExtractor` to run the pipeline without models or keys.

## Accuracy

Measured numbers go here — and only measured numbers. Run the harness on a public benchmark:

```bash
pip install datasets
python eval/prepare_aggrefact.py --out data/aggrefact.jsonl
python eval/run_eval.py data/aggrefact.jsonl --extractor sentence
```

Thresholds are tuned on a dev split and reported on a held-out test split. Default thresholds in
`VerifierConfig` are placeholders until you calibrate them.

## Design

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pipeline works and why, and
[docs/ROADMAP.md](docs/ROADMAP.md) for what comes next.
