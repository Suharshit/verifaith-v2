# Architecture

## Pipeline

```
answer ──► 1. extract claims ──► 2. guard ──┐
                                             ├─► 4. select candidates ─► 5. NLI (one batch) ─► 6. decide ─► 7. score
contexts ─► 3. sentence windows ────────────┘
```

1. **Extract claims** (`claims/`). Default: LLM via any OpenAI-compatible endpoint, temperature 0,
   JSON output validated by Pydantic, retries with backoff, in-memory cache. The prompt forbids
   correcting or adding facts and resolves pronouns using only the answer. Alternative:
   `SentenceExtractor` (no LLM).
2. **Guard** (`guard.py`). Every extracted claim must (a) contain only numbers present in the answer
   and (b) be entailed by the answer. If any claim fails, the whole extraction is discarded and the
   answer's own sentences are used instead, with a warning. This exists because an LLM extractor can
   silently "fix" a hallucination (1950 → 1889), which would make a wrong answer look faithful.
3. **Windows** (`text.py`). Contexts are split into sentences (pure Python, no downloads) and turned
   into overlapping windows of up to `window_size` sentences ending at each sentence. A sentence
   like "It stands 330 m tall." is always checked together with the sentences before it.
4. **Select candidates** (`retrieval.py`). If a context has at most `max_candidates` windows, all of
   them are checked. Otherwise a BM25-style lexical ranker picks the top ones. It never pads, so
   short contexts can't produce duplicate or bogus evidence. Swap in an embedding retriever via the
   `Retriever` protocol.
5. **NLI** (`nli/`). All (evidence, claim) pairs for the request go to the model in one batched call,
   encoded as a real sentence pair. Label order is read from the model config. Any model can be
   plugged in through the `EntailmentModel` protocol.
6. **Decide** (`pipeline.py`). Per claim, using full probabilities, not argmax:
   - `supported` if max P(entailment) over candidates ≥ `support_threshold`
   - else `contradicted` if max P(contradiction) ≥ `contradiction_threshold`
   - else `unsupported`
   Support is checked first on purpose: a spurious contradiction from an unrelated window must not
   override clear support elsewhere.
7. **Score** (`scoring.py`). `faithfulness = supported / total`, `contradiction_rate` reported
   separately. `faithful` requires high support *and* zero contradictions.

## Package layout

```
src/verifaith/
  schemas.py      public data contract (library, API and SDK all use it)
  config.py       VerifierConfig (thresholds) + Settings (env vars)
  pipeline.py     Verifier: orchestration and decision rule
  claims/         extractors (LLM, sentence) behind a Protocol
  guard.py        extraction faithfulness checks
  text.py         sentence splitting + evidence windows
  retrieval.py    candidate selection behind a Protocol
  nli/            entailment backends behind a Protocol
  scoring.py      aggregation and verdict
  service/app.py  FastAPI app factory (auth, limits, batch, request IDs)
  client.py       HTTP SDK
  testing.py      deterministic fakes for offline tests
```

Notebooks, if you use them, `import verifaith` — they never hold their own copy of pipeline code.

## Service decisions

- Models load once in the FastAPI lifespan, never at import time.
- API keys via `X-API-Key`, compared in constant time. Empty key list = dev mode, logged loudly.
- Input size limits, batch size limits, per-item errors in batch responses.
- Upstream LLM failure → 502; anything else → 500 with a generic message. No exception text leaks.
- Every response carries `X-Request-ID`; every result carries the version and config used.

## Known limitations (be upfront about these)

- Base-size NLI models are weak on arithmetic, unit conversion and multi-hop reasoning.
- **Support recall is the dominant error.** On RAGTruth, 2.8 of the 7.3 claims in an average
  *faithful* answer score `unsupported`, so 85% of faithful answers are flagged. Balanced accuracy
  is 0.568 there against 0.741 on single-claim VitaminC.
- **`contradicted` rarely fires**: 9% of answers annotated with an evident conflict, against 5% of
  faithful answers. The label is precise enough to block on, but it catches little.
- Evidence spanning distant parts of a document (beyond `window_size`) can be missed — including a
  pronoun whose antecedent is further back than the window reaches.
- The guard relies on the same NLI model; paraphrased-but-faithful claims can trigger a fallback.
- Non-prose contexts (JSON, as in RAGTruth's Data2txt task) are near chance: 0.518.
- Contradiction detection needs lexical overlap with the claim (`contradiction_min_overlap`),
  because NLI models score unrelated text as contradiction. A conflict worded differently is
  reported as `unsupported` instead.
- Thresholds are uncalibrated until you run `eval/`. On RAGTruth, requiring 69% of claims
  supported instead of 90% was worth ~6 points of balanced accuracy.
