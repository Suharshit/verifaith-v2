# VeriFaith — Project Brief

An orientation document for auditing this codebase. Covers what the project is, how it is put
together, and what every file does. ~1,500 lines of Python across 24 modules.

---

## 1. What this project is

**VeriFaith checks whether a generated answer is actually backed by the documents it was supposed
to be based on.**

You give it two things — an `answer` and a list of `contexts` (the source documents) — and it
returns a per-claim verdict with the exact evidence sentence behind each decision.

```
answer:   "The Eiffel Tower was completed in 1889 and is 500 metres tall."
contexts: ["The Eiffel Tower is in Paris. It was constructed from 1887 to 1889.
            It stands 330 metres tall."]

          |
          v

faithfulness: 0.5   verdict: "partial"
  [   supported] The Eiffel Tower was completed in 1889   <- "It was constructed from 1887 to 1889."
  [contradicted] The Eiffel Tower is 500 metres tall      <- "It stands 330 metres tall."
```

### The precise definition of "faithful"

This is the single most important thing to understand, and it is narrower than it sounds:

> **Faithful = every claim in the answer is supported by the given context.**

Three consequences that trip people up:

| It does NOT measure | Why |
|---|---|
| **Truth** | A claim matching a wrong source scores `supported`. VeriFaith checks grounding, not reality. |
| **Completeness** | An answer that says less than the sources is still perfectly faithful. |
| **Relevance** | Whether the answer addresses the question is a separate concern entirely. |

So VeriFaith answers exactly one question: *did the model make this up, or is it in the sources?*
That is the hallucination-detection problem for RAG systems.

### Three labels, not two

Note the distinction between `contradicted` and `unsupported` — it carries real product meaning:

- **`supported`** — some evidence window entails the claim.
- **`contradicted`** — evidence actively conflicts with it. *The model got it wrong.* Block it.
- **`unsupported`** — the context simply says nothing either way. *The model went off-source.*
  Usually flag, not block.

Collapsing these two into "bad" loses the signal you most want in production.

### Use cases

**1. Runtime guardrail.** Check an answer before it reaches the user; block or annotate
contradicted claims. See [examples/rag_guardrail.py](examples/rag_guardrail.py) for the pattern,
including the fail-open/fail-closed decision you have to make.

**2. Offline evaluation.** Score a RAG system across a test set and track faithfulness between
releases — a regression test for hallucination rate. See [eval/run_eval.py](eval/run_eval.py).

**3. Production monitoring.** Sample live traffic, alert when `contradiction_rate` climbs. Often
the first signal that a retrieval index has gone stale or a prompt change backfired.

**4. Model/prompt comparison.** Same contexts, two generators, compare faithfulness. Useful when
picking between models or deciding whether a cheaper model is good enough.

### How it is consumed

Three surfaces, one shared data contract ([schemas.py](src/verifaith/schemas.py)):

- **Library** — `from verifaith import Verifier`, in-process.
- **HTTP service** — FastAPI, so non-Python stacks can call it.
- **SDK** — `VeriFaithClient`, for calling the service from another Python project.

---

## 2. Architecture

### The pipeline

```
answer ---> 1. extract claims ---> 2. guard ----+
                                                +--> 4. select --> 5. NLI --> 6. decide --> 7. score
contexts --> 3. sentence windows ---------------+
```

**1. Extract claims** ([claims/](src/verifaith/claims/))
Split the answer into atomic, self-contained facts. Default is an LLM at temperature 0 with JSON
output; `SentenceExtractor` (one sentence = one claim) is the no-LLM fallback. Why atomic: "X was
built in 1889 and is 500m tall" is half right, and a single verdict on the whole sentence would
throw away that distinction.

**2. Guard** ([guard.py](src/verifaith/guard.py))
The subtle and important step. An LLM asked to extract claims will sometimes *silently correct*
them — the answer says 1950, the extractor writes 1889, and a hallucination scores as supported.
The guard catches this two ways: every number in a claim must appear in the answer, and the answer
must entail the claim. If any claim fails, the **entire** extraction is thrown out and the
pipeline falls back to the answer's raw sentences, with a warning. Regression test:
[tests/test_pipeline.py:19](tests/test_pipeline.py#L19).

**3. Sentence windows** ([text.py](src/verifaith/text.py))
Contexts are split into sentences, then grouped into overlapping windows of up to `window_size`
sentences *ending at* each sentence. This exists to solve pronouns: "It stands 330 metres tall"
entails nothing on its own, but paired with the preceding sentence naming the Eiffel Tower, it
does. Pure regex, no model download.

**4. Select candidates** ([retrieval.py](src/verifaith/retrieval.py))
A BM25-style lexical ranker picks the `max_candidates` most promising windows per claim — running
NLI over every window would be quadratic. It returns *all* windows when there are few enough, and
never pads, so short contexts cannot yield bogus evidence.

**5. NLI** ([nli/](src/verifaith/nli/))
The actual entailment scoring. Every (evidence, claim) pair for the whole request goes to the
model in **one batched call**. Uses a real sentence-pair encoding, and reads label order from the
model config rather than assuming `LABEL_0/1/2`.

**6. Decide** ([pipeline.py:115](src/verifaith/pipeline.py#L115))
Per claim, on full probabilities rather than argmax:

```
if   max P(entailment)    >= support_threshold       -> supported
elif max P(contradiction) >= contradiction_threshold -> contradicted
else                                                 -> unsupported
```

Support is tested **first, deliberately**: one spurious contradiction from an unrelated window
must not override clear support found elsewhere.

**7. Score** ([scoring.py](src/verifaith/scoring.py))
`faithfulness = supported / total`. `contradiction_rate` is reported separately rather than folded
in, because the two failure modes need different responses. The `faithful` verdict requires both
high support *and* zero contradictions.

### The organising idea: three swappable protocols

The architecture's one real structural decision. Three `typing.Protocol` seams:

| Protocol | Contract | Implementations |
|---|---|---|
| `ClaimExtractor` | `extract(answer) -> list[str]` | `LLMClaimExtractor`, `SentenceExtractor`, `StaticExtractor` (fake) |
| `Retriever` | `select(claim, windows, k) -> list[int]` | `LexicalRetriever` |
| `EntailmentModel` | `predict(pairs) -> list[NLIScores]` | `HFEntailmentModel`, `KeywordNLI` (fake) |

`Protocol` means structural typing — no base class to inherit, no registration. Anything with the
right method shape works. This is what lets the entire test suite run offline in 0.5s with no
models and no API keys, and what makes swapping the NLI model a one-line change.

### Two models, two very different roles

Worth internalising, because it drives both cost and accuracy:

| | Claim extraction | Entailment |
|---|---|---|
| Runs on | Remote LLM over HTTP | Local HuggingFace cross-encoder |
| Currently | `gemini-3.5-flash-lite` | `cross-encoder/nli-deberta-v3-base` |
| Calls per request | 1 | claims x candidates (batched) |
| Cost | API tokens | CPU/GPU time |

The entailment model is local by design — it is called many times per request, so a remote API
would be slow and expensive. **This is where your accuracy ceiling lives** (see section 4).

### Service design

- Models load once in the FastAPI **lifespan**, never at import. `service/app.py` uses a module
  `__getattr__` so `import verifaith.service.app` doesn't trigger a model download.
- `create_app(verifier, settings)` is an app *factory* — tests inject fakes, no globals.
- API keys via `X-API-Key`, compared with `hmac.compare_digest` (constant time, no timing leak).
  Empty key list = dev mode, logged loudly as a warning.
- Upstream LLM failure returns `502`; anything else returns `500` with a generic message.
  Exception text never reaches the client. Regression test:
  [tests/test_service.py:75](tests/test_service.py#L75).
- Every response carries `X-Request-ID`; every result embeds the version and config used, so any
  stored result is reproducible.

---

## 3. File-by-file reference

### Core library — `src/verifaith/`

| File | Lines | What it does |
|---|---|---|
| [`__init__.py`](src/verifaith/__init__.py) | 7 | Public surface: `Verifier`, `EvalResult`, `ClaimVerdict`, `Label`, `__version__`. |
| [`schemas.py`](src/verifaith/schemas.py) | 57 | **The data contract.** Pydantic models shared by library, API and SDK: `Label`, `Claim`, `Evidence`, `NLIScores`, `ClaimVerdict`, `EvalResult`. Change here = change everywhere. Start audits here. |
| [`config.py`](src/verifaith/config.py) | 44 | Two classes. `VerifierConfig` = per-request thresholds. `Settings` = env vars / `.env` (prefix `VERIFAITH_`). Note `api_keys` uses `Annotated[list[str], NoDecode]` so comma-separated values parse instead of being JSON-decoded. |
| [`pipeline.py`](src/verifaith/pipeline.py) | 150 | **The orchestrator.** `Verifier.evaluate()` runs all seven stages; `from_settings()` wires real models from env; `_decide()` holds the labelling rule; `_extract_claims()` holds the guard fallback. The most important file. |
| [`guard.py`](src/verifaith/guard.py) | 30 | `number_mismatches()` and `unfaithful_extractions()` — catches an extractor that rewrites facts. Small file, disproportionate importance. |
| [`text.py`](src/verifaith/text.py) | 89 | `split_sentences()` (regex + abbreviation list so "Dr." and "U.S." don't split) and `make_windows()` (overlapping evidence windows). Zero dependencies. |
| [`retrieval.py`](src/verifaith/retrieval.py) | 72 | `Retriever` protocol + `LexicalRetriever`, a hand-rolled BM25. Deterministic, no model. The extension point for embedding retrieval. |
| [`scoring.py`](src/verifaith/scoring.py) | 25 | `aggregate()` — counts verdicts, computes `faithfulness` and `contradiction_rate`, picks the overall verdict. Pure function, trivially auditable. |
| [`client.py`](src/verifaith/client.py) | 56 | `VeriFaithClient` HTTP SDK + `VeriFaithError`. Context-manager support. Parses responses back into `EvalResult`. |
| [`testing.py`](src/verifaith/testing.py) | 78 | `KeywordNLI` and `StaticExtractor` — deterministic fakes. Why the test suite needs no GPU, no network, no keys. Shipped so *your users* can test their integrations too. |

### Claim extraction — `src/verifaith/claims/`

| File | Lines | What it does |
|---|---|---|
| [`__init__.py`](src/verifaith/claims/__init__.py) | 4 | Re-exports `ClaimExtractor`, `ExtractionError`, `SentenceExtractor`. |
| [`base.py`](src/verifaith/claims/base.py) | 13 | The `ClaimExtractor` protocol and `ExtractionError`. |
| [`llm.py`](src/verifaith/claims/llm.py) | 100 | LLM extractor over any OpenAI-compatible `/chat/completions`. Contains `SYSTEM_PROMPT` — **read it**, its rules ("never correct a fact, even if you believe it is false") are load-bearing. Also: LRU cache keyed on model+answer, exponential backoff, and no retry on 4xx except 429. |
| [`sentence.py`](src/verifaith/claims/sentence.py) | 10 | One sentence = one claim. No LLM, no key, no cost. Also serves as the guard's fallback. |

### Entailment — `src/verifaith/nli/`

| File | Lines | What it does |
|---|---|---|
| [`__init__.py`](src/verifaith/nli/__init__.py) | 3 | Re-exports `EntailmentModel`. |
| [`base.py`](src/verifaith/nli/base.py) | 13 | The `EntailmentModel` protocol: `predict(pairs) -> list[NLIScores]`, one per pair, in order. |
| [`hf.py`](src/verifaith/nli/hf.py) | 58 | HuggingFace backend. Batching, CUDA auto-detect, `truncation="only_first"` (truncates evidence, never the claim), and validates that the model's `id2label` really is 3-way NLI instead of trusting position. |

### Service — `src/verifaith/service/`

| File | Lines | What it does |
|---|---|---|
| [`__init__.py`](src/verifaith/service/__init__.py) | 0 | Empty package marker. |
| [`app.py`](src/verifaith/service/app.py) | 145 | The whole HTTP layer: `create_app()` factory, `EvaluateRequest`/`BatchRequest`, auth dependency, size limits, request-ID middleware, `/healthz`, `/v1/evaluate`, `/v1/evaluate/batch`. Batch returns per-item ok/error rather than failing wholesale. |

### Tests — `tests/` (25 tests, ~0.5s, fully offline)

| File | Lines | What it covers |
|---|---|---|
| [`conftest.py`](tests/conftest.py) | 15 | Two fixtures: `nli` (`KeywordNLI`) and `verifier`. |
| [`test_pipeline.py`](tests/test_pipeline.py) | 73 | Core behaviour, and **two named v1 regressions**: pronoun evidence, and the extractor that "corrects" 1950 to 1889. Also: support-beats-noisy-contradiction, empty context, no claims, single batched NLI call. |
| [`test_service.py`](tests/test_service.py) | 96 | Auth, per-request config override, 413 limits, 422 validation, batch partial failure, no-leak 502, and the SDK against the live app. |
| [`test_llm_extractor.py`](tests/test_llm_extractor.py) | 61 | The HTTP extractor via `httpx.MockTransport`: parsing, caching, 5xx retry, unparseable output, and *no* retry on 401. |
| [`test_text_and_retrieval.py`](tests/test_text_and_retrieval.py) | 33 | Sentence splitting (abbreviations, whitespace) and BM25 ranking. |

### Evaluation — `eval/`

| File | Lines | What it does |
|---|---|---|
| [`run_eval.py`](eval/run_eval.py) | 116 | The harness. Hash-based dev/test split (stable across runs), AUROC, balanced accuracy, threshold search on dev only, per-group breakdown, explicit failure counting. Writes timestamped JSON + records. **See section 4 for its limitation.** |
| [`prepare_aggrefact.py`](eval/prepare_aggrefact.py) | 44 | Converts `lytang/LLM-AggreFact` to the harness JSONL format. **The dataset is gated on HuggingFace** — needs terms accepted and `huggingface-cli login`. |
| [`prepare_vitaminc.py`](eval/prepare_vitaminc.py) | 60 | Same for `tals/vitaminc`, which is ungated. Added as a stand-in for AggreFact. Balanced sampling; `SUPPORTS` maps to 1, `REFUTES` and `NOT ENOUGH INFO` map to 0. |

### Examples — `examples/`

| File | Lines | What it does |
|---|---|---|
| [`library_usage.py`](examples/library_usage.py) | 16 | Smallest in-process usage. Good smoke test — expect `partial 0.5`. |
| [`rag_guardrail.py`](examples/rag_guardrail.py) | 19 | The production pattern: block on `contradicted`, annotate on `unsupported`, and an explicit fail-open choice if the checker is down. |

### Config, ops and docs

| File | What it does |
|---|---|
| [`pyproject.toml`](pyproject.toml) | Hatchling build, src-layout. Core deps are tiny (pydantic, httpx); torch/transformers sit behind the `nli` extra, FastAPI behind `service`. Also holds pytest and ruff config. |
| [`Dockerfile`](Dockerfile) | Python 3.11-slim. Installs CPU-only torch from the PyTorch index, then **bakes the NLI model into the image** so containers start fast and offline. Runs as non-root `app`. |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Push/PR: ruff + pytest on Python 3.10 and 3.12. Dev extras only — no model downloads in CI. |
| [`.env.example`](.env.example) | Template for the `VERIFAITH_*` vars. Copy to `.env`. |
| `.env` | **Your live config — holds the Gemini API key. Gitignored.** |
| [`.gitignore`](.gitignore) | Covers `.env`, `.venv/`, `eval/results/`, caches. Does **not** cover `data/`. |
| [`README.md`](README.md) | Public front door. Its "Accuracy" section is still a stub. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The canonical pipeline write-up — the source for section 2 above. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phased plan. Phase 0 is done except publishing a baseline. |
| `project-brief.md` | This file. |

### Generated, not source

| Path | What it is |
|---|---|
| `data/vitaminc.jsonl` | 1,500 balanced rows from VitaminC. Regenerate with `prepare_vitaminc.py`. Not gitignored — add `data/` before committing. |
| `eval/results/` | Timestamped eval summaries and per-item records. Gitignored. |
| `.venv/` | Local virtualenv (Python 3.12). Gitignored. |

---

## 4. Current state — what an auditor should know

**Works, measured:** 25 tests pass; full pipeline verified end to end on the Eiffel Tower example;
Gemini claim extraction confirmed live against the OpenAI-compatible endpoint.

**Measured baseline** (VitaminC, 1,087 held-out test rows, `--extractor sentence`, CPU):

| Metric | Value |
|---|---|
| Balanced accuracy | 0.740 |
| AUROC | 0.741 |
| Throughput | 0.16 s/item |
| Failures | 0 |

This is a **proxy benchmark**, not LLM-AggreFact, and is not comparable to published numbers.

**Three open issues, in priority order:**

1. **The eval harness does not calibrate the thresholds that matter.** `run_eval.py` only tunes
   `faithful_threshold`; `support_threshold` and `contradiction_threshold` — the parameters that
   actually label each claim — are never touched. Worse, on single-claim datasets like
   AggreFact and VitaminC, `faithfulness` can only be 0 or 1, making that tuning degenerate.

2. **Thresholds remain at their defaults, and that is the correct choice for now.** A manual sweep
   of `support_threshold` was flat across 0.05–0.95 (dev 0.709 to 0.747). Tuning picked 0.9 on dev
   and performed *worse* on test (0.734) than the default 0.6 (0.740) — dev-split noise, not
   signal.

3. **The accuracy ceiling is the NLI model, not the configuration.** At AUROC 0.74,
   `nli-deberta-v3-base` is the binding constraint. Swapping it (MiniCheck / AlignScore-style, per
   ROADMAP Phase 1) is the lever that moves the number. Threshold tuning is not.

**Known limitations** (from ARCHITECTURE.md, all still true): base-size NLI models are weak on
arithmetic, unit conversion and multi-hop reasoning; evidence spread beyond `window_size` can be
missed; the guard uses the same NLI model, so paraphrased-but-faithful claims can trigger a
spurious fallback.
