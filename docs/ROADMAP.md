# Roadmap

Each phase ends with numbers from `eval/`, not claims.

## Phase 0 — Foundation (this repo)
- [x] Package with protocols for extractor / retriever / NLI
- [x] Fixes for v1 failures, each with a regression test
- [x] FastAPI service, SDK, Dockerfile, CI
- [ ] Publish a baseline: run `eval/` on LLM-AggreFact with the default NLI model

## Phase 1 — Accuracy
- [ ] Calibrate thresholds on dev; per-domain threshold presets
- [ ] Try grounding-specific models (MiniCheck, AlignScore-style) as `EntailmentModel` backends
- [ ] Hybrid verifier: NLI for confident cases, LLM verifier only for the uncertain band
- [ ] Embedding retriever for long contexts; compare against lexical
- [ ] Honest baselines: real RAGAS, MiniCheck, and a prompted LLM judge on the same test split
- [ ] Add RAGTruth / HaluEval answer-level evaluation (answers with multiple claims)

## Phase 2 — Service
- [ ] Async endpoints + job queue for large batches; webhook on completion
- [ ] Per-key rate limits and usage metering (Redis)
- [ ] Persistent result store + `GET /v1/evaluations/{id}`
- [ ] Prometheus metrics (latency per stage, contradiction rate, extractor fallback rate)
- [ ] Integrations: LangChain / LlamaIndex callbacks, a GitHub Action for CI evals

## Phase 3 — Product
- [ ] Dashboard: faithfulness over time, worst claims, drill-down to evidence
- [ ] Human review loop: accept/reject verdicts → new labeled data → recalibration
- [ ] Multi-tenant auth, org-level configs, hosted offering
