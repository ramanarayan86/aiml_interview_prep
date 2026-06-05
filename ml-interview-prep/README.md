<div align="center">

# 🧠 The GenAI Interview Question Bank

### First‑principles answers for senior LLM &amp; VLM interviews — built to read like a webpage, ship as a GitHub repo.

![Scope](https://img.shields.io/badge/scope-LLMs_&_VLMs-1f4fa3)
![Sections](https://img.shields.io/badge/sections-30-0e8a6e)
![Questions](https://img.shields.io/badge/questions-800%2B-6c4fe0)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-c77a12)
![Format](https://img.shields.io/badge/format-Markdown_+_SVG_+_LaTeX-16202e)

*Architecture · Pretraining · Post‑training · Reasoning · Inference · Quantization · VLMs · RAG · MoE · Agents · Safety · Diffusion · Systems*

</div>

---

## Why this exists

Most interview prep is a wall of bullet‑point "facts" you can't reconstruct under pressure. This bank is the opposite: **every answer is derived from first principles**, explained so plainly a beginner can follow it, yet deep enough to satisfy a panel at a frontier lab. Each page carries its own **figures, equations, pseudocode, runnable code, worked numerical examples, and references** — the things that make a concept *stick* and let you teach it back, which is the real test of understanding.

The goal: open one page, and walk out able to answer that question **like you invented the idea.**

---

## How to use this repo

- **Browse by section** in the [table of contents](#table-of-contents) below, then drill into a question page.
- Every question is a **standalone page** with consistent structure (the [answer template](./_TEMPLATE.md)), so you can read in any order.
- **Navigation is built in:** every page links back to its section index and to Home, plus previous/next — so you can move like a website, never hitting a dead end.
- Figures are committed **SVGs** (crisp at any zoom, diff‑able in git); math renders via GitHub's LaTeX support; diagrams use **Mermaid**.
- Want to contribute or extend it? See [CONTRIBUTING](./CONTRIBUTING.md) — adding a question is "copy the template, fill the blanks."

> [!TIP]
> **Suggested study loop:** read the *20‑second answer* → read the *worked example* → close the page and try to redraw Figure 1 and re‑derive the key equation from memory → only then read the follow‑up drills. If you can reproduce the figure, you own the concept.

---

## Repository layout

```
ml-interview-prep/
├── README.md                        ← you are here
├── _TEMPLATE.md                     ← answer-page template (copy to add a question)
├── CONTRIBUTING.md                  ← how to add / improve answers
├── assets/
│   ├── q01/  fig1-block-overview.svg  fig2-gradient-flow.svg  fig3-residual-stream.svg
│   ├── q02/  fig1-variance-explosion.svg  fig2-softmax-sharpening.svg
│   ├── q03/  fig1-batch-vs-layer-axes.svg  fig2-normalization-geometry.svg
│   ├── q04/  fig1-postnorm-gradient-vanishing.svg  fig2-prenorm-vs-postnorm-block.svg
│   ├── q05/  fig1-permutation-invariance.svg  fig2-encoding-comparison.svg  fig3-extrapolation.svg
│   ├── q06/  fig1-ffn-as-memory.svg  fig2-ffn-dimensions.svg
│   ├── q07/  fig1-teacher-forcing-vs-inference.svg  fig2-error-accumulation.svg
│   ├── q08/  fig1-attention-matrix-causal.svg  fig2-masking-mechanism.svg
│   ├── q09/  fig1-head-diversity.svg  fig2-single-vs-multi-head.svg
│   ├── q10/  fig1-architecture-comparison.svg  fig2-attention-patterns.svg
│   ├── q11/  fig1-complexity-scaling.svg  fig2-memory-hierarchy.svg
│   ├── q12/  fig1-head-types.svg  fig2-induction-circuit.svg
│   ├── q13/  fig1-attention-variants.svg  fig2-kv-cache-memory.svg
│   ├── q14/  fig1-norm-computation-steps.svg  fig2-invariance-properties.svg
│   ├── q15/  fig1-activation-functions.svg  fig2-swiglu-gating.svg
│   ├── q16/  fig1-weight-tying-diagram.svg  fig2-gradient-flow-tying.svg
│   ├── q17/  fig1-dropout-placement.svg  fig2-scale-vs-dropout.svg
│   ├── q18/  fig1-problem.svg  fig2-pipeline.svg  fig3-geometry.svg  fig4-distribution.svg
│   ├── q19/  fig1-extrapolation-comparison.svg  fig2-rope-aliasing.svg
│   ├── q20/  fig1-alibi-bias-matrix.svg  fig2-alibi-vs-rope-extrapolation.svg
│   ├── q21/  fig1-sequential-vs-parallel-block.svg  fig2-gemm-fusion.svg
│   ├── q22/  fig1-bias-ln-redundancy.svg  fig2-residual-stream-dc-offset.svg
│   ├── q23/  fig1-rope-rotation-geometry.svg  fig2-rope-long-context-scaling.svg
│   ├── q24/  fig1-flash-attention-tiling.svg  fig2-io-complexity.svg
│   ├── q25/  fig1-attention-sink-pattern.svg  fig2-streaming-llm-window.svg
│   ├── q26/  fig1-attention-complexity-landscape.svg  fig2-linear-attention-recurrence.svg
│   ├── q27/  fig1-ssm-recurrence-vs-convolution.svg  fig2-selective-scan-hardware.svg
│   └── q28/  fig1-expressivity-tasks.svg  fig2-state-size-tradeoff.svg
└── questions/
    └── section-01-transformer-architecture/
        ├── README.md                ← section index (all 42 questions)
        ├── q01-transformer-block.md
        ├── q02-scaled-attention.md
        ├── ...  (Q1–Q28 fully answered, Q29–Q42 scaffolded)
        ├── q22-no-bias.md
        ├── q23-rope-derivation.md
        ├── q24-flash-attention.md
        ├── q25-attention-sinks.md
        ├── q26-subquadratic-attention.md
        ├── q27-ssm-mamba.md
        └── q28-expressivity-gap.md
```

Each section is a folder with its own index; each question is one Markdown file; figures live under `assets/qNN/` — exactly 2–3 SVGs per question. Adding a new question never touches any existing file.

---

## ⭐ Answered so far — Section 1 (28 / 42)

All 28 answers follow the same first-principles template: 20-second answer → derivation → figures → PyTorch code → worked example → interview drills.

| Group | Questions answered |
|---|---|
| **Basic (Q1–Q10)** | [Q1 — Transformer block](./questions/section-01-transformer-architecture/q01-transformer-block.md) · [Q2 — √d_k scaling](./questions/section-01-transformer-architecture/q02-scaled-attention.md) · [Q3 — LayerNorm vs BatchNorm](./questions/section-01-transformer-architecture/q03-layernorm-batchnorm.md) · [Q4 — Pre/Post-norm](./questions/section-01-transformer-architecture/q04-prenorm-postnorm.md) · [Q5 — Positional encodings](./questions/section-01-transformer-architecture/q05-positional-encodings.md) · [Q6 — FFN](./questions/section-01-transformer-architecture/q06-ffn.md) · [Q7 — Teacher forcing](./questions/section-01-transformer-architecture/q07-teacher-forcing.md) · [Q8 — Causal masking](./questions/section-01-transformer-architecture/q08-causal-masking.md) · [Q9 — Multi-head attention](./questions/section-01-transformer-architecture/q09-multi-head-attention.md) · [Q10 — Encoder/Decoder](./questions/section-01-transformer-architecture/q10-encoder-decoder.md) |
| **Intermediate (Q11–Q22)** | [Q11 — Attention complexity O(n²)](./questions/section-01-transformer-architecture/q11-attention-complexity.md) · [Q12 — MHA & interpretability](./questions/section-01-transformer-architecture/q12-mha-interpretability.md) · [Q13 — GQA & MQA](./questions/section-01-transformer-architecture/q13-gqa-mqa.md) · [Q14 — RMSNorm vs LayerNorm](./questions/section-01-transformer-architecture/q14-rmsnorm-vs-layernorm.md) · [Q15 — SwiGLU & GeGLU](./questions/section-01-transformer-architecture/q15-swiglu-geglu.md) · [Q16 — Weight tying](./questions/section-01-transformer-architecture/q16-weight-tying.md) · [Q17 — Dropout](./questions/section-01-transformer-architecture/q17-dropout.md) · [**Q18 — QK‑norm ⭐**](./questions/section-01-transformer-architecture/q18-qk-norm.md) · [Q19 — RoPE extrapolation](./questions/section-01-transformer-architecture/q19-position-encodings.md) · [Q20 — ALiBi](./questions/section-01-transformer-architecture/q20-alibi.md) · [Q21 — Parallel Attn+FFN](./questions/section-01-transformer-architecture/q21-parallel-attention-ffn.md) · [Q22 — No-bias Transformers](./questions/section-01-transformer-architecture/q22-no-bias.md) |
| **Advanced (Q23–Q28)** | [Q23 — RoPE deep dive + NTK/YaRN/LongRoPE](./questions/section-01-transformer-architecture/q23-rope-derivation.md) · [Q24 — FlashAttention v1/v2/v3](./questions/section-01-transformer-architecture/q24-flash-attention.md) · [Q25 — Attention sinks & StreamingLLM](./questions/section-01-transformer-architecture/q25-attention-sinks.md) · [Q26 — Sub-quadratic attention](./questions/section-01-transformer-architecture/q26-subquadratic-attention.md) · [Q27 — SSMs & Mamba-2 / SSD](./questions/section-01-transformer-architecture/q27-ssm-mamba.md) · [Q28 — Expressivity gap: Transformers vs SSMs](./questions/section-01-transformer-architecture/q28-expressivity-gap.md) |

> **Reference standard:** [Q18 — QK‑norm](./questions/section-01-transformer-architecture/q18-qk-norm.md) — softmax temperature, attention‑logit explosion, entropy collapse, ℓ₂ vs RMSNorm variants, MLA incompatibility, and the σReparam / soft‑cap / μP family — with four original SVG figures.

---

## Table of contents

> **Legend** &nbsp; ✅ has published answers &nbsp;·&nbsp; 📝 scaffolded (template ready)

| # | Section | Focus | Status |
|---|---------|-------|:------:|
| 1 | [**Transformer Architecture Fundamentals**](./questions/section-01-transformer-architecture/README.md) | attention, norms, position, FFN, variants | ✅ |
| 2 | Tokenization and Embeddings | BPE/WordPiece/Unigram, byte‑level, glitch tokens | 📝 |
| 3 | Pretraining and Scaling Laws | Chinchilla, data/compute optima, emergence | 📝 |
| 4 | Post‑training: SFT, RLHF, DPO and Beyond | reward models, PPO, DPO/IPO/KTO | 📝 |
| 5 | Reasoning, CoT, and Inference‑Time Compute | CoT, self‑consistency, search, o‑series | 📝 |
| 6 | Efficient Inference and Serving | KV cache, batching, speculative decoding | 📝 |
| 7 | Quantization and Model Compression | INT8/FP8/FP4, GPTQ/AWQ, MXFP, distillation | 📝 |
| 8 | Vision‑Language Models (VLMs) | connectors, resolution, any‑res, evaluation | 📝 |
| 9 | Retrieval, RAG, and Long Context | chunking, rerankers, long‑context attention | 📝 |
| 10 | Mixture of Experts (MoE) | routing, load balancing, capacity factor | 📝 |
| 11 | Agents, Tool Use, and Code Generation | function calling, planning, sandboxes | 📝 |
| 12 | Evaluation, Safety, and Interpretability | benchmarks, red‑teaming, SAEs, probes | 📝 |
| 13 | Diffusion, DiT, and Generative Modeling | DDPM, score matching, DiT, flow matching | 📝 |
| 14 | Systems and Hardware | GPU memory hierarchy, MFU, kernels, NCCL | 📝 |
| 15 | Mathematical Foundations and Probability | linear algebra, information theory, optimization | 📝 |
| 16 | Distributed Training and Optimization | DP/TP/PP/FSDP, ZeRO, optimizer states | 📝 |
| 17 | Fine‑Tuning, PEFT, and Adaptation | LoRA/QLoRA, adapters, prefix tuning | 📝 |
| 18 | Speech, Audio, and Multimodal Beyond Vision | tokenizing audio, ASR/TTS, unified models | 📝 |
| 19 | Prompt Engineering, ICL, and Steering | in‑context learning, activation steering | 📝 |
| 20 | Data Engineering and Curation for LLMs | dedup, filtering, mixtures, synthetic data | 📝 |
| 21 | Frontier Topics and Research Directions | SSMs, world models, test‑time training | 📝 |
| 22 | Behavioral and Research Leadership | scoping, prioritization, technical leadership | 📝 |
| 23 | Vector Databases and Search Infrastructure | vector index types, ANN search, filtering, hybrid search, DB selection | 📝 |
| 24 | Document Digitization and Chunking Strategies | OCR pipelines, semantic chunking, tables/lists/charts, ideal chunk size | 📝 |
| 25 | Embedding Models and Semantic Search | embedding architectures, long vs short content, domain adaptation, benchmarking | 📝 |
| 26 | Production LLM Systems and Cost Optimization | latency/throughput tradeoffs, caching, batching, cost modelling, monitoring | 📝 |
| 27 | Hallucination: Causes, Detection, and Mitigation | taxonomy of hallucinations, grounding, self‑consistency checks, CAD | 📝 |
| 28 | Prompt Security and Adversarial Inputs | prompt injection, jailbreaks, indirect attacks, defence layers | 📝 |
| 29 | LLM Application Design Patterns and Case Studies | conversation memory, tool orchestration, multi‑agent pipelines, cost–quality dial | 📝 |
| 30 | LLM Evaluation in Practice | reference‑free metrics, LLM‑as‑judge, RAG‑specific metrics, evals at scale | 📝 |

---

## Section 1 at a glance

28 of 42 questions answered — all of Basic (Q1–Q10), all 12 Intermediate (Q11–Q22), and first 6 Advanced (Q23–Q28). [**Open Section 1 →**](./questions/section-01-transformer-architecture/README.md)

<div align="center">

| Basic | Intermediate | Advanced | Applied |
|:---:|:---:|:---:|:---:|
| Q1–Q10 ✅ | Q11–Q22 ✅ | Q23–Q28 ✅ · Q29–Q34 📝 | Q35–Q42 📝 |
| forward pass, √d_k, norms, position, FFN, masking, heads, architectures | complexity, GQA/MQA, RMSNorm, SwiGLU, weight tying, dropout, **QK‑norm**, RoPE, ALiBi, parallel block, no-bias | **RoPE derivation**, **FlashAttention v1/v2/v3**, **attention sinks**, **sub‑quadratic**, **Mamba/SSD**, **expressivity gap** · diff Transformer, RetNet, softmax-free, Turing completeness | MFU debugging, repetition, norm migration, 128K context, NaNs |

</div>

---

---

## New sections — question scaffolds

The eight new sections below extend the bank to cover the applied and production-facing interview topics that are heavily tested at ML engineering and applied scientist roles. Questions are ordered basic → advanced within each section.

---

### Section 23 · Vector Databases and Search Infrastructure

| # | Question | Level |
|---|----------|:-----:|
| 23‑01 | What is a vector index, and how does it differ from a vector database and a vector plugin? | Basic |
| 23‑02 | Walk through the trade-offs between exact nearest‑neighbour search and approximate nearest‑neighbour search — when is ANN unacceptable? | Basic |
| 23‑03 | Explain how HNSW works; what graph parameters control the quality‑speed tradeoff? | Intermediate |
| 23‑04 | How does IVF‑Flat clustering work, and when does it fail silently? | Intermediate |
| 23‑05 | Explain product quantisation (PQ) — what information does it discard and how does that affect recall? | Intermediate |
| 23‑06 | What is locality‑sensitive hashing and in which regimes does it outperform graph‑based indices? | Intermediate |
| 23‑07 | How does pre‑filtering differ from post‑filtering in vector search, and which is safer for recall? | Intermediate |
| 23‑08 | Compare cosine similarity, dot product, and Euclidean distance as retrieval metrics — when should you choose each? | Intermediate |
| 23‑09 | How would you decide which vector database to use for a given production system? What axes matter most? | Advanced |
| 23‑10 | A team reports that filtered vector search is returning far fewer results than expected. Walk through your diagnosis. | Applied |
| 23‑11 | How do you benchmark a vector database before committing to it in production? What does a rigorous eval look like? | Applied |
| 23‑12 | How would you design a vector search system that must handle 1 billion embeddings at p99 < 50 ms? | Applied |

---

### Section 24 · Document Digitization and Chunking Strategies

| # | Question | Level |
|---|----------|:-----:|
| 24‑01 | Why do we chunk documents before embedding, and what goes wrong if we skip it? | Basic |
| 24‑02 | What factors determine the ideal chunk size for a given retrieval task? | Basic |
| 24‑03 | Compare fixed‑size, sentence‑boundary, and semantic chunking — when does each break? | Intermediate |
| 24‑04 | How do you handle multi‑column PDFs, scanned images, and mixed-language documents in a digitization pipeline? | Intermediate |
| 24‑05 | What strategies exist for chunking tables without destroying their relational structure? | Intermediate |
| 24‑06 | How would you extract and represent charts and graphs for RAG retrieval? | Intermediate |
| 24‑07 | Describe a production‑grade document processing pipeline: ingestion → chunking → indexing. What are the failure modes at each stage? | Advanced |
| 24‑08 | How do you handle list items and hierarchical outlines so that parent context is not lost at retrieval time? | Advanced |
| 24‑09 | How would you empirically find the optimal chunk size for a specific corpus and set of user queries? | Applied |
| 24‑10 | A client's RAG system gives fragmented answers despite high retrieval accuracy. You suspect chunking. How do you diagnose and fix it? | Applied |

---

### Section 25 · Embedding Models and Semantic Search

| # | Question | Level |
|---|----------|:-----:|
| 25‑01 | What is a vector embedding and what property makes it useful for semantic search? | Basic |
| 25‑02 | How does a bi‑encoder differ from a cross‑encoder, and when should you use each? | Basic |
| 25‑03 | Why does embedding long documents with the same model used for short queries degrade retrieval? How do you address it? | Intermediate |
| 25‑04 | What is matryoshka representation learning (MRL) and why does it matter for production retrieval systems? | Intermediate |
| 25‑05 | How would you benchmark embedding models on a new domain before deploying them? | Intermediate |
| 25‑06 | Your embedding model has high recall@10 on general benchmarks but poor accuracy on your domain data. What steps would you take to improve it without full retraining? | Intermediate |
| 25‑07 | Walk through fine‑tuning a sentence‑transformer model for a specific retrieval task — data, loss function, negatives strategy. | Advanced |
| 25‑08 | How do late interaction models (ColBERT) differ from bi‑encoders, and what storage and latency trade‑off do they introduce? | Advanced |
| 25‑09 | How would you design an embedding pipeline that handles code, SQL, and natural language queries under a single index? | Applied |

---

### Section 26 · Production LLM Systems and Cost Optimisation

| # | Question | Level |
|---|----------|:-----:|
| 26‑01 | What are the primary cost drivers in a production LLM deployment — where does money actually go? | Basic |
| 26‑02 | How does the KV cache work, and what is its memory footprint at inference time? | Basic |
| 26‑03 | Explain continuous batching. How does it improve GPU utilisation over static batching? | Intermediate |
| 26‑04 | What is speculative decoding, and under what conditions does it give a meaningful speedup? | Intermediate |
| 26‑05 | How does prefix caching (prompt caching) work, and which workloads benefit most? | Intermediate |
| 26‑06 | What techniques can you use to reduce output latency without changing the model weights? | Intermediate |
| 26‑07 | Compare SaaS API costs versus self‑hosted open‑source for a 10 M token/day workload — what drives the cross‑over point? | Intermediate |
| 26‑08 | How would you monitor a production LLM for quality regressions after a silent model update? | Advanced |
| 26‑09 | Your LLM endpoint p99 latency spiked 3× overnight without a code deploy. Walk through your investigation. | Applied |
| 26‑10 | Design an LLM serving architecture for a chatbot that has 500 concurrent users and a hard 2‑second TTFT SLA. | Applied |

---

### Section 27 · Hallucination: Causes, Detection, and Mitigation

| # | Question | Level |
|---|----------|:-----:|
| 27‑01 | What is hallucination in LLMs, and what are the main types? | Basic |
| 27‑02 | Why do instruction‑tuned models sometimes hallucinate more confidently than base models? | Intermediate |
| 27‑03 | How does retrieval‑augmented generation reduce hallucination, and when does it fail to? | Intermediate |
| 27‑04 | What is the difference between intrinsic and extrinsic hallucination — give a concrete example of each. | Intermediate |
| 27‑05 | Explain the chain‑of‑verification (CoVe) approach and its core limitation. | Intermediate |
| 27‑06 | How would you prompt an LLM to produce an answer only when it has sufficient context, and refuse otherwise? | Intermediate |
| 27‑07 | What is contrastive decoding and how does it suppress hallucination at the token‑sampling level? | Advanced |
| 27‑08 | How can you detect hallucination post‑generation without a ground‑truth reference? | Advanced |
| 27‑09 | Design a multi‑layer hallucination control strategy for a medical QA system. What tradeoffs does each layer introduce? | Applied |

---

### Section 28 · Prompt Security and Adversarial Inputs

| # | Question | Level |
|---|----------|:-----:|
| 28‑01 | What is prompt injection, and how does it differ from a jailbreak? | Basic |
| 28‑02 | Explain indirect prompt injection — how can a user's query be hijacked by malicious content in retrieved documents? | Intermediate |
| 28‑03 | What are the main categories of jailbreak techniques and what makes them effective? | Intermediate |
| 28‑04 | What defence layers exist between a user's input and the LLM response, and what does each protect against? | Intermediate |
| 28‑05 | How do you test a production LLM application for prompt‑injection vulnerabilities before launch? | Advanced |
| 28‑06 | A deployed RAG system is returning confidential data from other tenants' documents. Diagnose the root cause and propose a fix. | Applied |

---

### Section 29 · LLM Application Design Patterns and Case Studies

| # | Question | Level |
|---|----------|:-----:|
| 29‑01 | How does a RAG system differ architecturally from a fine‑tuned LLM for a domain‑specific QA task? When do you choose each? | Basic |
| 29‑02 | What are the main architecture patterns for giving an LLM access to proprietary data? | Basic |
| 29‑03 | How do you implement and manage conversation memory in a long‑running chat assistant without blowing past the context window? | Intermediate |
| 29‑04 | What are the core components of a production‑grade RAG system and what can go wrong in each? | Intermediate |
| 29‑05 | How do you handle multi‑hop queries that require synthesising information from more than one retrieved passage? | Intermediate |
| 29‑06 | Explain the ReAct pattern — what problem does it solve and what are its failure modes? | Intermediate |
| 29‑07 | Compare function‑calling / tool‑use versus free‑form planning in agent systems. When does each break down? | Advanced |
| 29‑08 | How would you architect a multi‑agent pipeline for a task that requires parallel sub‑tasks with dependencies? | Advanced |
| 29‑09 | A client's LLM chat assistant is accurate but costs 10× the budget. Walk through your cost‑reduction plan without sacrificing answer quality. | Applied |
| 29‑10 | Design a document Q&A system over a 500‑page annual report with tables, charts, and footnotes. Cover ingestion, retrieval, generation, and evaluation. | Applied |

---

### Section 30 · LLM Evaluation in Practice

| # | Question | Level |
|---|----------|:-----:|
| 30‑01 | What are the standard metrics for evaluating a RAG system's retrieval quality, and when does each metric mislead you? | Basic |
| 30‑02 | Explain BLEU, ROUGE, and BERTScore — what does each capture and what does each miss? | Basic |
| 30‑03 | What is LLM‑as‑judge evaluation, and what biases does it introduce? | Intermediate |
| 30‑04 | How do you evaluate faithfulness vs. relevance in a RAG pipeline? What is the distinction? | Intermediate |
| 30‑05 | Your RAG system scores well on ROUGE but users are complaining about unhelpful answers. How do you diagnose and close this gap? | Intermediate |
| 30‑06 | Compare RAGAS, TruLens, and DeepEval as evaluation frameworks — what does each measure and where do they fall short? | Intermediate |
| 30‑07 | How would you build a reference‑free hallucination detection metric for a domain where no ground truth exists? | Advanced |
| 30‑08 | How do you design an offline evaluation suite that predicts online user satisfaction? What signals help bridge the gap? | Advanced |
| 30‑09 | Your team is launching a new LLM feature. Design an A/B testing strategy that safely measures quality without exposing all users to a potentially worse model. | Applied |

## License & attribution

Educational resource. Each answer cites its primary sources; please preserve attributions when reusing. Corrections and new answers welcome via PR — see [CONTRIBUTING](./CONTRIBUTING.md).

<div align="center">
<sub>Built as a go‑to, scalable template for ML interview preparation. One concept, one page, fully understood.</sub>
</div>
