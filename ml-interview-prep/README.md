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
│   └── q22/  fig1-bias-ln-redundancy.svg  fig2-residual-stream-dc-offset.svg
└── questions/
    └── section-01-transformer-architecture/
        ├── README.md                ← section index (all 42 questions)
        ├── q01-transformer-block.md
        ├── q02-scaled-attention.md
        ├── ...  (Q1–Q22 fully answered, Q23–Q42 scaffolded)
        └── q22-no-bias.md
```

Each section is a folder with its own index; each question is one Markdown file; figures live under `assets/qNN/` — exactly 2–3 SVGs per question. Adding a new question never touches any existing file.

---

## ⭐ Answered so far — Section 1 (22 / 42)

All 22 answers follow the same first-principles template: 20-second answer → derivation → figures → PyTorch code → worked example → interview drills.

| Group | Questions answered |
|---|---|
| **Basic (Q1–Q10)** | [Q1 — Transformer block](./questions/section-01-transformer-architecture/q01-transformer-block.md) · [Q2 — √d_k scaling](./questions/section-01-transformer-architecture/q02-scaled-attention.md) · [Q3 — LayerNorm vs BatchNorm](./questions/section-01-transformer-architecture/q03-layernorm-batchnorm.md) · [Q4 — Pre/Post-norm](./questions/section-01-transformer-architecture/q04-prenorm-postnorm.md) · [Q5 — Positional encodings](./questions/section-01-transformer-architecture/q05-positional-encodings.md) · [Q6 — FFN](./questions/section-01-transformer-architecture/q06-ffn.md) · [Q7 — Teacher forcing](./questions/section-01-transformer-architecture/q07-teacher-forcing.md) · [Q8 — Causal masking](./questions/section-01-transformer-architecture/q08-causal-masking.md) · [Q9 — Multi-head attention](./questions/section-01-transformer-architecture/q09-multi-head-attention.md) · [Q10 — Encoder/Decoder](./questions/section-01-transformer-architecture/q10-encoder-decoder.md) |
| **Intermediate (Q11–Q22)** | [Q11 — Attention complexity O(n²)](./questions/section-01-transformer-architecture/q11-attention-complexity.md) · [Q12 — MHA & interpretability](./questions/section-01-transformer-architecture/q12-mha-interpretability.md) · [Q13 — GQA & MQA](./questions/section-01-transformer-architecture/q13-gqa-mqa.md) · [Q14 — RMSNorm vs LayerNorm](./questions/section-01-transformer-architecture/q14-rmsnorm-vs-layernorm.md) · [Q15 — SwiGLU & GeGLU](./questions/section-01-transformer-architecture/q15-swiglu-geglu.md) · [Q16 — Weight tying](./questions/section-01-transformer-architecture/q16-weight-tying.md) · [Q17 — Dropout](./questions/section-01-transformer-architecture/q17-dropout.md) · [**Q18 — QK‑norm ⭐**](./questions/section-01-transformer-architecture/q18-qk-norm.md) · [Q19 — RoPE extrapolation](./questions/section-01-transformer-architecture/q19-position-encodings.md) · [Q20 — ALiBi](./questions/section-01-transformer-architecture/q20-alibi.md) · [Q21 — Parallel Attn+FFN](./questions/section-01-transformer-architecture/q21-parallel-attention-ffn.md) · [Q22 — No-bias Transformers](./questions/section-01-transformer-architecture/q22-no-bias.md) |

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

22 of 42 questions answered — all of Basic (Q1–Q10) and all 12 Intermediate (Q11–Q22). [**Open Section 1 →**](./questions/section-01-transformer-architecture/README.md)

<div align="center">

| Basic | Intermediate | Advanced | Applied |
|:---:|:---:|:---:|:---:|
| Q1–Q10 ✅ | Q11–Q22 ✅ | Q23–Q34 | Q35–Q42 |
| forward pass, √d_k, norms, position, FFN, masking, heads, architectures | complexity, GQA/MQA, RMSNorm, SwiGLU, weight tying, dropout, **QK‑norm**, RoPE, ALiBi, parallel block, no-bias | RoPE derivation, FlashAttention, sinks, sub‑quadratic, Mamba | MFU debugging, repetition, norm migration, 128K context, NaNs |

</div>

---

## License & attribution

Educational resource. Each answer cites its primary sources; please preserve attributions when reusing. Corrections and new answers welcome via PR — see [CONTRIBUTING](./CONTRIBUTING.md).

<div align="center">
<sub>Built as a go‑to, scalable template for ML interview preparation. One concept, one page, fully understood.</sub>
</div>
