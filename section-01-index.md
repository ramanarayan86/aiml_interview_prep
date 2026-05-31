<div align="center">

[🏠 Home](../../README.md)

</div>

---

# Section 1 · Transformer Architecture Fundamentals

<div align="center">

![Questions](https://img.shields.io/badge/questions-42-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Advanced→Applied-0e8a6e)
![Status](https://img.shields.io/badge/answered-1%2F42-c77a12)

</div>

The bedrock section. If you can teach every idea here from first principles, most of the rest of the bank becomes derivative. Questions are grouped by depth; each links to a standalone page built from the [shared answer template](../../_TEMPLATE.md).

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 planned (same template)

---

### 🟢 Basic

| # | Question | Status |
|---|----------|:------:|
| Q1 | Forward pass of a Transformer block; what residual connections protect against | 📝 |
| Q2 | Why scaled dot‑product attention divides by √d_k | 📝 |
| Q3 | LayerNorm vs BatchNorm; why LayerNorm for sequences | 📝 |
| Q4 | Pre‑norm vs post‑norm; why the field shifted to pre‑norm | 📝 |
| Q5 | Why positional encodings; sinusoidal vs learned vs RoPE vs ALiBi | 📝 |
| Q6 | Role of the FFN; why ~4× hidden dimension | 📝 |
| Q7 | Teacher forcing & exposure bias | 📝 |
| Q8 | Causal masking and how it's implemented | 📝 |
| Q9 | Why multiple attention heads instead of one big head | 📝 |
| Q10 | Encoder‑only vs decoder‑only vs encoder‑decoder; when to choose each | 📝 |

### 🟡 Intermediate

| # | Question | Status |
|---|----------|:------:|
| Q11 | Compute & memory complexity of attention; where O(n²) actually hurts | 📝 |
| Q12 | Multi‑Head Attention & head specialization (induction/copy heads, sinks) | 📝 |
| Q13 | GQA & MQA; quality–throughput tradeoff | 📝 |
| Q14 | RMSNorm vs LayerNorm; why RMSNorm dominates | 📝 |
| Q15 | SwiGLU & GeGLU; why they replaced ReLU/GELU | 📝 |
| Q16 | Weight tying of input/output embeddings | 📝 |
| Q17 | Dropout in Transformers; why often off for large‑scale pretraining | 📝 |
| **Q18** | **What is the QK‑norm trick? What stability problem does it solve?** | [✅](./q18-qk-norm.md) |
| Q19 | Absolute vs relative vs rotary encodings; why RoPE extrapolates | 📝 |
| Q20 | ALiBi: linear bias on logits for length extrapolation | 📝 |
| Q21 | Parallel attention+FFN (GPT‑J, PaLM, Falcon); tradeoffs | 📝 |
| Q22 | No‑bias Transformers: theoretical & empirical justification | 📝 |

### 🔴 Advanced

| # | Question | Status |
|---|----------|:------:|
| Q23 | Derive RoPE from first principles; long‑context scaling (NTK, YaRN, LongRoPE) | 📝 |
| Q24 | FlashAttention tiling & IO‑awareness; what changed in v2/v3 | 📝 |
| Q25 | Attention sinks & StreamingLLM | 📝 |
| Q26 | Sub‑quadratic attention: Linear Attention, Performer, Mamba, hybrids | 📝 |
| Q27 | SSMs: selective scan; Mamba‑2 and State Space Duality | 📝 |
| Q28 | Expressivity gap between Transformers and SSMs | 📝 |
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

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [▶️ Start with Q18 — QK‑norm](./q18-qk-norm.md)

</div>
