<div align="center">

[🏠 Home](../../README.md)

</div>

---

# Section 1 · Transformer Architecture Fundamentals

<div align="center">

![Questions](https://img.shields.io/badge/questions-42-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Advanced→Applied-0e8a6e)
![Status](https://img.shields.io/badge/answered-28%2F42-c77a12)

</div>

The bedrock section. If you can teach every idea here from first principles, most of the rest of the bank becomes derivative. Questions are grouped by depth; each links to a standalone page built from the [shared answer template](../../_TEMPLATE.md).

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 planned (same template)

---

### 🟢 Basic

| # | Question | Status |
|---|----------|:------:|
| Q1 | [Forward pass of a Transformer block; what residual connections protect against](./q01-transformer-block.md) | ✅ |
| Q2 | [Why scaled dot‑product attention divides by √d_k](./q02-scaled-attention.md) | ✅ |
| Q3 | [LayerNorm vs BatchNorm; why LayerNorm for sequences](./q03-layernorm-batchnorm.md) | ✅ |
| Q4 | [Pre‑norm vs post‑norm; why the field shifted to pre‑norm](./q04-prenorm-postnorm.md) | ✅ |
| Q5 | [Why positional encodings; sinusoidal vs learned vs RoPE vs ALiBi](./q05-positional-encodings.md) | ✅ |
| Q6 | [Role of the FFN; why ~4× hidden dimension](./q06-ffn.md) | ✅ |
| Q7 | [Teacher forcing & exposure bias](./q07-teacher-forcing.md) | ✅ |
| Q8 | [Causal masking and how it's implemented](./q08-causal-masking.md) | ✅ |
| Q9 | [Why multiple attention heads instead of one big head](./q09-multi-head-attention.md) | ✅ |
| Q10 | [Encoder‑only vs decoder‑only vs encoder‑decoder; when to choose each](./q10-encoder-decoder.md) | ✅ |

### 🟡 Intermediate

| # | Question | Status |
|---|----------|:------:|
| Q11 | [Compute & memory complexity of attention; where O(n²) actually hurts](./q11-attention-complexity.md) | ✅ |
| Q12 | [Multi‑Head Attention & head specialization (induction/copy heads, sinks)](./q12-mha-interpretability.md) | ✅ |
| Q13 | [GQA & MQA; quality–throughput tradeoff](./q13-gqa-mqa.md) | ✅ |
| Q14 | [RMSNorm vs LayerNorm; why RMSNorm dominates](./q14-rmsnorm-vs-layernorm.md) | ✅ |
| Q15 | [SwiGLU & GeGLU; why they replaced ReLU/GELU](./q15-swiglu-geglu.md) | ✅ |
| Q16 | [Weight tying of input/output embeddings](./q16-weight-tying.md) | ✅ |
| Q17 | [Dropout in Transformers; why often off for large‑scale pretraining](./q17-dropout.md) | ✅ |
| **Q18** | **[What is the QK‑norm trick? What stability problem does it solve?](./q18-qk-norm.md)** | ✅ |
| Q19 | [Absolute vs relative vs rotary encodings; why RoPE extrapolates](./q19-position-encodings.md) | ✅ |
| Q20 | [ALiBi: linear bias on logits for length extrapolation](./q20-alibi.md) | ✅ |
| Q21 | [Parallel attention+FFN (GPT‑J, PaLM, Falcon); tradeoffs](./q21-parallel-attention-ffn.md) | ✅ |
| Q22 | [No‑bias Transformers: theoretical & empirical justification](./q22-no-bias.md) | ✅ |

### 🔴 Advanced

| # | Question | Status |
|---|----------|:------:|
| Q23 | [Derive RoPE from first principles; long‑context scaling (NTK, YaRN, LongRoPE)](./q23-rope-derivation.md) | ✅ |
| Q24 | [FlashAttention tiling & IO‑awareness; what changed in v2/v3](./q24-flash-attention.md) | ✅ |
| Q25 | [Attention sinks & StreamingLLM](./q25-attention-sinks.md) | ✅ |
| Q26 | [Sub‑quadratic attention: Linear Attention, Performer, Mamba, hybrids](./q26-subquadratic-attention.md) | ✅ |
| Q27 | [SSMs: selective scan; Mamba‑2 and State Space Duality](./q27-ssm-mamba.md) | ✅ |
| Q28 | [Expressivity gap between Transformers and SSMs](./q28-expressivity-gap.md) | ✅ |
| Q29 | Differential Transformer | 📝 |
| Q30 | Hyena, RetNet, RWKV complexity profiles | 📝 |
| Q31 | Role of softmax; is softmax‑free attention viable | 📝 |
| Q32 | Is attention Turing complete | 📝 |
| Q33 | Attention as a kernel method | 📝 |
| Q34 | Transformers on graphs (Graphormer, GraphGPS) | 📝 |

### 🔵 Practical / Applied

| # | Question | Status |
|---|----------|:------:|
| Q35 | Debugging 35% MFU on a 13B model on H100 | 📝 |
| Q36 | Great training loss but repetitive generation — diagnose | 📝 |
| Q37 | Migrating post‑norm → pre‑norm without full retrain | 📝 |
| Q38 | Experiments to require before swapping in sub‑quadratic attention | 📝 |
| Q39 | Extend 4K → 128K context in a 2‑week budget | 📝 |
| Q40 | Intermittent NaNs in attention kernels under concurrency | 📝 |
| Q41 | GQA‑8 vs GQA‑4 for a 70B model — how to decide | 📝 |
| Q42 | One layer = 40% of inference latency — investigate | 📝 |

---

## Reading order suggestions

**New to Transformers?** Q1 → Q2 → Q3 → Q4 → Q5 → Q6 → Q7 → Q8 → Q9 → Q10

**Interview in 48 hours?** [Q18 — QK‑norm](./q18-qk-norm.md) · [Q11 — Complexity](./q11-attention-complexity.md) · [Q13 — GQA/MQA](./q13-gqa-mqa.md) · [Q6 — FFN](./q06-ffn.md) · [Q12 — MHA Interpretability](./q12-mha-interpretability.md)

**Full Intermediate track:** Q11 → Q12 → Q13 → Q14 → Q15 → Q16 → Q17 → Q18 → Q19 → Q20 → Q21 → Q22

**Advanced track:** [Q23 — RoPE Deep Dive](./q23-rope-derivation.md) · [Q24 — FlashAttention](./q24-flash-attention.md) · [Q25 — Attention Sinks](./q25-attention-sinks.md) · [Q26 — Sub-quadratic Attention](./q26-subquadratic-attention.md) · [Q27 — SSMs / Mamba](./q27-ssm-mamba.md) · [Q28 — Expressivity Gap](./q28-expressivity-gap.md)

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [▶️ Start here: Q1 — Transformer Block](./q01-transformer-block.md) &nbsp;|&nbsp; [⭐ Reference answer: Q18 — QK‑norm](./q18-qk-norm.md)

</div>
