<div align="center">

# 🧠 The GenAI Interview Question Bank

### First‑principles answers for senior LLM &amp; VLM interviews — built to read like a webpage, ship as a GitHub repo.

![Scope](https://img.shields.io/badge/scope-LLMs_&_VLMs-1f4fa3)
![Sections](https://img.shields.io/badge/sections-22-0e8a6e)
![Questions](https://img.shields.io/badge/questions-600%2B-6c4fe0)
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
├── README.md                      ← you are here (the question bank)
├── _TEMPLATE.md                   ← the answer-page template
├── CONTRIBUTING.md                ← how to add / improve answers
├── assets/
│   └── q18/                       ← per-question SVG figures
│       ├── fig1-problem.svg
│       ├── fig2-pipeline.svg
│       ├── fig3-geometry.svg
│       └── fig4-distribution.svg
└── questions/
    └── section-01-transformer-architecture/
        ├── README.md              ← section index (links every question)
        └── q18-qk-norm.md         ← a fully worked example answer
```

Each section is a folder with its own index; each question is one Markdown file; each question's images live under `assets/qNN/`. This keeps the repo **scalable** — adding Section 12 or Question 437 never touches anything else.

---

## ⭐ Answered so far — Section 1 (16 / 42)

All 16 answers follow the same first-principles template: 20-second answer → derivation → figures → PyTorch code → worked example → interview drills.

| Group | Questions answered |
|---|---|
| **Basic (Q1–Q10)** | [Q1 — Transformer block](./questions/section-01-transformer-architecture/q01-transformer-block.md) · [Q2 — √d_k scaling](./questions/section-01-transformer-architecture/q02-scaled-attention.md) · [Q3 — LayerNorm vs BatchNorm](./questions/section-01-transformer-architecture/q03-layernorm-batchnorm.md) · [Q4 — Pre/Post-norm](./questions/section-01-transformer-architecture/q04-prenorm-postnorm.md) · [Q5 — Positional encodings](./questions/section-01-transformer-architecture/q05-positional-encodings.md) · [Q6 — FFN](./questions/section-01-transformer-architecture/q06-ffn.md) · [Q7 — Teacher forcing](./questions/section-01-transformer-architecture/q07-teacher-forcing.md) · [Q8 — Causal masking](./questions/section-01-transformer-architecture/q08-causal-masking.md) · [Q9 — Multi-head attention](./questions/section-01-transformer-architecture/q09-multi-head-attention.md) · [Q10 — Encoder/Decoder](./questions/section-01-transformer-architecture/q10-encoder-decoder.md) |
| **Intermediate (Q11–Q18)** | [Q11 — Attention complexity O(n²)](./questions/section-01-transformer-architecture/q11-attention-complexity.md) · [Q12 — MHA & interpretability](./questions/section-01-transformer-architecture/q12-mha-interpretability.md) · [Q13 — GQA & MQA](./questions/section-01-transformer-architecture/q13-gqa-mqa.md) · [Q14 — RMSNorm vs LayerNorm](./questions/section-01-transformer-architecture/q14-rmsnorm-vs-layernorm.md) · [Q15 — SwiGLU & GeGLU](./questions/section-01-transformer-architecture/q15-swiglu-geglu.md) · [**Q18 — QK‑norm ⭐**](./questions/section-01-transformer-architecture/q18-qk-norm.md) |

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

---

## Section 1 at a glance

16 of 42 questions answered — all of Basic (Q1–Q10) and the first six Intermediate (Q11–Q15, Q18). [**Open Section 1 →**](./questions/section-01-transformer-architecture/README.md)

<div align="center">

| Basic | Intermediate | Advanced | Applied |
|:---:|:---:|:---:|:---:|
| Q1–Q10 ✅ | Q11–Q22 (6/12 ✅) | Q23–Q34 | Q35–Q42 |
| forward pass, √d_k, norms, position, FFN, masking, heads, architectures | complexity ✅, GQA/MQA ✅, RMSNorm ✅, SwiGLU ✅, **QK‑norm ✅**, interpretability ✅ | RoPE derivation, FlashAttention, sinks, sub‑quadratic, Mamba | MFU debugging, repetition, norm migration, 128K context, NaNs |

</div>

---

## License & attribution

Educational resource. Each answer cites its primary sources; please preserve attributions when reusing. Corrections and new answers welcome via PR — see [CONTRIBUTING](./CONTRIBUTING.md).

<div align="center">
<sub>Built as a go‑to, scalable template for ML interview preparation. One concept, one page, fully understood.</sub>
</div>
