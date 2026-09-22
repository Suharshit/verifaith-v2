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

Measured with `cross-encoder/nli-deberta-v3-base` and `--extractor sentence`, on held-out test
splits. Reproduce with:

```bash
pip install datasets
python eval/prepare_ragtruth.py --per-task 200 --out data/ragtruth.jsonl   # real RAG answers
python eval/run_eval.py data/ragtruth.jsonl --extractor sentence --limit 300
python eval/prepare_vitaminc.py --out data/vitaminc.jsonl                  # single claims
python eval/run_eval.py data/vitaminc.jsonl --extractor sentence
```

| | RAGTruth (219 answers) | VitaminC (1,087 claims) |
|---|---|---|
| What it measures | whole pipeline on real RAG answers | the NLI model on one claim + one sentence |
| Balanced accuracy, shipped config | **0.568** | **0.741** |
| Balanced accuracy, dev-tuned threshold | 0.630 | 0.740 |
| AUROC (`support_score`) | 0.665 | 0.832 |
| Per task | Summary 0.734, QA 0.631, Data2txt 0.518 | real 0.705, synthetic 0.796 |

**Read the RAGTruth column before deploying this as a guardrail.** On multi-sentence answers
over real passages, 85% of genuinely faithful answers are flagged, because a faithful answer
averages 7.3 claims and 2.8 of them are scored `unsupported`. Most of that is missed support,
not false contradiction: only 5% of faithful answers contain a `contradicted` claim. But
`contradicted` also fires on just 9% of answers annotated as containing an evident conflict, so
today the three labels carry far less signal than the design intends. The binding constraint is
the entailment model (see [docs/ROADMAP.md](docs/ROADMAP.md) Phase 1), not the thresholds.

Single-claim benchmarks like VitaminC look much better because they exercise only the NLI step.
Do not quote that number as the system's accuracy.

Thresholds are tuned on a dev split and reported on a held-out test split. Defaults in
`VerifierConfig` are placeholders until you calibrate them on your own data; on RAGTruth,
requiring 69% of claims supported rather than 90% was worth about 6 points.

Latency, CPU-only, is 12 s median and 42 s p95 per answer. Use a GPU for runtime guardrail use.

## Design

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pipeline works and why, and
[docs/ROADMAP.md](docs/ROADMAP.md) for what comes next.
