<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 8 — Vision-Language Models](../section-08-vision-language-models/README.md) &nbsp;|&nbsp; [Section 10 — Mixture of Experts ➡️](../section-10-mixture-of-experts/README.md)

</div>

---

# Section 9 · Retrieval, RAG, and Long Context

<div align="center">

![Questions](https://img.shields.io/badge/questions-36-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F36-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-36%2F36-6c4fe0)

</div>

RAG and long-context are the two competing paradigms for grounding LLMs in external knowledge. This section covers **dense vs sparse retrieval**, **reranking**, **query rewriting**, **context extension** (YaRN, LongRoPE), **long-context evaluation** (RULER, ∞Bench), and the practical system design of large-scale RAG pipelines.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q9‑01 | What is RAG? Walk through the basic retrieval → augmentation → generation pipeline. | RAG · retrieval · augmentation · pipeline | 📝 |
| Q9‑02 | Compare dense retrieval (DPR, ColBERT) vs sparse retrieval (BM25). When does each win? | DPR · ColBERT · BM25 · dense-sparse | 📝 |
| Q9‑03 | What is a vector database? Why don't we just use brute-force search? | vector-database · ANN · HNSW · brute-force | 📝 |
| Q9‑04 | Explain ANN indexes — HNSW, IVF, ScaNN. What are the recall/latency tradeoffs? | HNSW · IVF · ScaNN · recall-latency | 📝 |
| Q9‑05 | What is chunking, and why is it deceptively important? | chunking · chunk-size · overlap · RAG | 📝 |
| Q9‑06 | Compare semantic search and keyword search. When do you want both (hybrid)? | semantic-search · BM25 · hybrid · keyword | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q9‑07 | Walk through ColBERT's late interaction. Why is it more expressive than single-vector retrieval, and what's the cost? | ColBERT · late-interaction · MaxSim · expressiveness | 📝 |
| Q9‑08 | Compare RAG, long-context (just stuff everything in), and fine-tuning for domain adaptation. When does each win? Reference Lost in the Middle, RULER, NIAH. | RAG · long-context · fine-tuning · Lost-in-Middle | 📝 |
| Q9‑09 | Discuss reranking. Why is a cross-encoder reranker over BM25 often competitive with dense retrieval + reranker? | reranking · cross-encoder · BM25 · pipeline | 📝 |
| Q9‑10 | Explain query rewriting, HyDE (Hypothetical Document Embeddings), and multi-query retrieval. What problems does each solve? | query-rewriting · HyDE · multi-query · retrieval | 📝 |
| Q9‑11 | What is GraphRAG? When does graph structure help vs hurt? | GraphRAG · knowledge-graph · graph-structure · RAG | 📝 |
| Q9‑12 | Discuss embedding model training (E5, BGE, GTE, NV-Embed). What objective and data choices matter? | embedding-training · E5 · BGE · contrastive | 📝 |
| Q9‑13 | What is the role of hard negatives in dense retrieval training? How are they mined? | hard-negatives · mining · dense-retrieval · training | 📝 |
| Q9‑14 | Compare bi-encoders, cross-encoders, and late-interaction models on accuracy-latency tradeoffs. | bi-encoder · cross-encoder · late-interaction · tradeoff | 📝 |
| Q9‑15 | Explain Matryoshka representation learning. Why does it help embedding-serving cost? | Matryoshka · MRL · variable-dimension · embedding | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q9‑16 | Long-context evaluation is broken — NIAH is too easy. Discuss RULER, LongBench, ∞Bench, and what a real long-context benchmark needs. | NIAH · RULER · LongBench · ∞Bench · evaluation | 📝 |
| Q9‑17 | Discuss the empirical finding that retrieval-augmented small models often beat long-context large models on factual tasks. What does this imply for memory architectures? | RAG-vs-long-context · factual-tasks · small-models | 📝 |
| Q9‑18 | Design a RAG system over 100M documents with sub-200ms p99 latency and freshness guarantees of 5 minutes. Walk through indexing, sharding, and the read/write path. | 100M-docs · 200ms · freshness · sharding | 📝 |
| Q9‑19 | Critique embedding models — they collapse semantics into a single vector. Discuss late-interaction (ColBERT), multi-vector (PLAID), and learned-sparse (SPLADE) as alternatives. | single-vector · PLAID · SPLADE · ColBERT | 📝 |
| Q9‑20 | Discuss agentic RAG / self-RAG. How does letting the model decide what to retrieve and when change the system design? | agentic-RAG · self-RAG · retrieval-decision · design | 📝 |
| Q9‑21 | Walk through the long-context training recipe. How do you extend a 4K-context model to 128K (or 1M+) without degrading short-context performance? | long-context-training · 128K · 1M · context-extension | 📝 |
| Q9‑22 | Discuss the role of position interpolation, YaRN, and LongRoPE in context extension. What's the theoretical and empirical comparison? | position-interpolation · YaRN · LongRoPE · context-extension | 📝 |
| Q9‑23 | Critically evaluate the claim that long context obviates RAG. What use cases favor each, and where do they coexist? | long-context · RAG · coexistence · use-cases | 📝 |
| Q9‑24 | Design an evaluation harness that distinguishes 'retrieval' (find the needle) from 'reasoning over long context' (multi-hop synthesis). | retrieval-vs-reasoning · evaluation · multi-hop · long-context | 📝 |
| Q9‑25 | Discuss RAG hallucination — when the retrieved context contradicts model priors, what does the model do? How do you measure and mitigate? | RAG-hallucination · context-conflict · prior · mitigation | 📝 |
| Q9‑26 | Walk through how you would implement prefix caching for a RAG system with overlapping retrieved chunks. What's the data structure and eviction policy? | prefix-caching · RAG · overlapping-chunks · eviction | 📝 |
| Q9‑27 | Discuss the interaction between RAG and reasoning. Does retrieval help or hurt for multi-step reasoning, and why? | RAG · multi-step-reasoning · retrieval-interaction | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q9‑28 | Your RAG system has 95% retrieval recall but only 70% answer correctness. Walk through where the gap is and how you'd close it. | retrieval-recall · answer-correctness · gap · debugging | 📝 |
| Q9‑29 | Walk through how you'd choose between adding RAG vs extending context length for an enterprise knowledge-base assistant. | RAG-vs-long-context · enterprise · decision · knowledge-base | 📝 |
| Q9‑30 | Your embedding model produces high cosine similarity for documents that are topically unrelated. Walk through your investigation and remediation. | cosine-similarity · false-positive · embedding · debugging | 📝 |
| Q9‑31 | Walk through how you'd build a freshness-guaranteed RAG over a corpus that updates every 30 seconds. | freshness · real-time · indexing · streaming | 📝 |
| Q9‑32 | Your reranker is helping on dev but hurting on production traffic. Walk through your investigation. | reranker · distribution-shift · dev-prod-gap · debugging | 📝 |
| Q9‑33 | You need to handle queries that span multiple retrieved documents requiring synthesis. Walk through your design beyond simple stuff-and-prompt. | multi-doc-synthesis · RAG · query-decomposition · fusion | 📝 |
| Q9‑34 | Walk through how you'd debug a RAG system where the model frequently ignores the retrieved context in favor of its priors. | context-ignoring · prior-bias · RAG · debugging | 📝 |
| Q9‑35 | You're given 1M documents and a $50K/month inference budget. Walk through your retrieval architecture and embedding-vs-index tradeoffs. | 1M-docs · budget · architecture · embedding-vs-index | 📝 |
| Q9‑36 | Walk through how you'd design RAG evaluation that separates retrieval quality from generation quality. | RAG-evaluation · retrieval-quality · generation-quality · separation | 📝 |

---

## Why these questions matter

RAG is the dominant production architecture for grounded AI systems. These questions test whether a candidate understands the **full RAG stack** — from ANN index design to embedding training to hallucination mitigation to evaluation.

| Theme | Key questions |
|-------|--------------|
| **Retrieval fundamentals** | Q9‑02, Q9‑04, Q9‑07, Q9‑13 — dense vs sparse, ColBERT, hard negatives |
| **System design** | Q9‑18, Q9‑26, Q9‑31, Q9‑35 — 100M docs, caching, freshness, budget |
| **Long context vs RAG** | Q9‑08, Q9‑17, Q9‑23, Q9‑29 — when each wins |
| **Evaluation** | Q9‑16, Q9‑24, Q9‑36 — NIAH/RULER, retrieval vs reasoning |

---

## Reading order suggestions

**New to RAG?** Q9‑01 → Q9‑02 → Q9‑03 → Q9‑05 → Q9‑09

**Interview in 48 hours?** Q9‑08 (RAG vs long context) · Q9‑07 (ColBERT) · Q9‑18 (system design) · Q9‑25 (hallucination) · Q9‑16 (eval)

**Full track:** Q9‑01 → Q9‑06 → Q9‑07 → Q9‑15 → Q9‑16 → Q9‑18 → Q9‑23

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 8 — Vision-Language Models](../section-08-vision-language-models/README.md) &nbsp;|&nbsp; [Section 10 — Mixture of Experts ➡️](../section-10-mixture-of-experts/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s9qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
