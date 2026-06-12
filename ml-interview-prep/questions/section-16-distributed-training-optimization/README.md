<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 15 — Mathematical Foundations](../section-15-mathematical-foundations/README.md) &nbsp;|&nbsp; [Section 17 — Fine-Tuning, PEFT & Adaptation ➡️](../section-17-fine-tuning-peft-adaptation/README.md)

</div>

---

# Section 16 · Distributed Training and Optimization

<div align="center">

![Questions](https://img.shields.io/badge/questions-31-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F31-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-31%2F31-6c4fe0)

</div>

Distributed training is the engineering backbone of large-model development. This section covers **Adam and its variants** (AdamW, Lion, Adafactor, Shampoo), **FSDP and ZeRO**, **gradient accumulation and checkpointing**, **large-batch optimization**, and the full landscape of **optimizer design for trillion-parameter training**.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q16‑01 | What is the difference between synchronous and asynchronous SGD? Why is sync preferred for LLM training? | sync-SGD · async-SGD · staleness · consistency | 📝 |
| Q16‑02 | Explain Adam's update rule. What do β1, β2, and ε control? | Adam · β1 · β2 · ε · moment-estimates | 📝 |
| Q16‑03 | What is the role of the optimizer state in memory budgeting? Compare AdamW, Lion, Adafactor on this axis. | optimizer-state · AdamW · Lion · Adafactor · memory | 📝 |
| Q16‑04 | Discuss gradient accumulation. How does it differ from data parallelism with larger micro-batch? | gradient-accumulation · micro-batch · data-parallelism | 📝 |
| Q16‑05 | What is gradient checkpointing (activation checkpointing)? What's the compute-memory tradeoff? | gradient-checkpointing · activation · recompute · memory | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q16‑06 | Walk through FSDP (Fully Sharded Data Parallel). How does it differ from ZeRO-3 in implementation? | FSDP · ZeRO-3 · sharding · implementation | 📝 |
| Q16‑07 | Discuss communication-computation overlap. How does prefetching of weights in FSDP work? | communication-overlap · FSDP · prefetch · async | 📝 |
| Q16‑08 | Compare Adam, Adafactor, Lion, Sophia, and Shampoo. What does each trade off? | Adam · Adafactor · Lion · Sophia · Shampoo · comparison | 📝 |
| Q16‑09 | What is the role of EMA (exponential moving average) in training (BYOL, diffusion, RLHF)? Why does it stabilize? | EMA · BYOL · diffusion · stability | 📝 |
| Q16‑10 | Discuss large-batch optimization. Why does LAMB / LARS help at extreme batch sizes? | large-batch · LAMB · LARS · learning-rate-scaling | 📝 |
| Q16‑11 | Explain the linear scaling rule for learning rate with batch size. When does it break? | linear-scaling · learning-rate · batch-size · breakdown | 📝 |
| Q16‑12 | What is gradient noise scale (McCandlish et al.)? How is it used to estimate critical batch size? | gradient-noise-scale · critical-batch-size · McCandlish | 📝 |
| Q16‑13 | Discuss the role of stochastic weight averaging (SWA) and its connection to flat minima. | SWA · flat-minima · generalization · averaging | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q16‑14 | Walk through the design of an optimizer for trillion-parameter training. What memory, communication, and stability constraints dominate? | trillion-parameter · optimizer-design · memory · communication | 📝 |
| Q16‑15 | Discuss second-order optimizers (Shampoo, K-FAC, SOAP). Where do they win over Adam, and what's the wall-clock story? | Shampoo · K-FAC · SOAP · second-order · wall-clock | 📝 |
| Q16‑16 | Critically evaluate Sophia and its claimed Hessian-aware updates. Has it scaled to frontier-model training? | Sophia · Hessian · second-order · frontier-scale | 📝 |
| Q16‑17 | Walk through the design of a low-communication distributed optimizer (DiLoCo, Lo-fi). What's the convergence-vs-bandwidth tradeoff? | DiLoCo · Lo-fi · communication-efficient · convergence | 📝 |
| Q16‑18 | Discuss the role of learning rate scheduling at the end of training (decay-to-zero, cooldown). Why has WSD become popular? | WSD · cooldown · decay · learning-rate-schedule | 📝 |
| Q16‑19 | Walk through how you'd implement async checkpointing for a 1T-parameter MoE. What are the failure modes? | async-checkpointing · 1T-MoE · failure-modes | 📝 |
| Q16‑20 | Discuss the connection between SGD's implicit bias and generalization. What does the recent literature say about flat minima and edge-of-stability? | SGD · implicit-bias · flat-minima · edge-of-stability | 📝 |
| Q16‑21 | Critically evaluate gradient compression (PowerSGD, 1-bit Adam). Why hasn't it been adopted at frontier scale? | gradient-compression · PowerSGD · 1-bit-Adam · adoption | 📝 |
| Q16‑22 | Discuss the role of mixed-precision in optimizer state. How does Llama-3 / DeepSeek-V3 handle BF16 weights with FP32 optimizer states? | mixed-precision · optimizer-state · BF16 · FP32 | 📝 |
| Q16‑23 | Walk through how you would design an optimizer that is robust to loss spikes. What does the empirical evidence (PaLM, GLM-130B, OPT) suggest? | loss-spikes · optimizer-robustness · PaLM · OPT | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q16‑24 | Your FSDP training shows poor compute-communication overlap. Walk through your investigation and the kernel/sharding changes you'd try. | FSDP · overlap · investigation · sharding | 📝 |
| Q16‑25 | Walk through how you'd choose between FSDP, ZeRO-3, and 3D parallelism for a new 70B project. | FSDP · ZeRO-3 · 3D-parallelism · decision | 📝 |
| Q16‑26 | Your optimizer state corruption is causing intermittent loss spikes. Walk through your diagnostic and prevention plan. | optimizer-state · corruption · loss-spikes · diagnosis | 📝 |
| Q16‑27 | Walk through how you'd safely upgrade from Adam to a new optimizer (Lion, Shampoo) mid-training. | optimizer-swap · Lion · Shampoo · mid-training | 📝 |
| Q16‑28 | You inherit a training codebase where convergence is 30% slower than a published baseline at the same scale. Walk through your investigation. | convergence · baseline-gap · debugging · investigation | 📝 |
| Q16‑29 | Walk through how you'd debug why doubling your batch size doesn't double throughput. | batch-size · throughput · bottleneck · debugging | 📝 |
| Q16‑30 | Your team is hitting communication-bound limits on InfiniBand. Walk through what kernel-, algorithm-, and infra-level changes you'd consider. | InfiniBand · communication-bound · kernel · algorithm | 📝 |
| Q16‑31 | Walk through how you'd design async checkpointing that handles failures without corrupting state. | async-checkpointing · failure · corruption · design | 📝 |

---

## Why these questions matter

Distributed training knowledge is tested at every senior ML engineer role. These questions probe whether a candidate can navigate the **optimizer landscape, memory tradeoffs, and failure modes** of large-scale training.

| Theme | Key questions |
|-------|--------------|
| **Optimizers** | Q16‑02, Q16‑03, Q16‑08, Q16‑15 — Adam/AdamW, Lion, second-order |
| **Parallelism** | Q16‑06, Q16‑07, Q16‑25 — FSDP, ZeRO, overlap |
| **Large-batch & schedule** | Q16‑10, Q16‑11, Q16‑12, Q16‑18 — LAMB, scaling rule, WSD |
| **Stability & debugging** | Q16‑23, Q16‑26, Q16‑27, Q16‑28 — loss spikes, optimizer swap |

---

## Reading order suggestions

**New to the area?** Q16‑01 → Q16‑02 → Q16‑04 → Q16‑05 → Q16‑06

**Interview in 48 hours?** Q16‑06 (FSDP/ZeRO) · Q16‑08 (optimizer comparison) · Q16‑14 (trillion-param) · Q16‑18 (WSD) · Q16‑23 (loss spikes)

**Full track:** Q16‑01 → Q16‑05 → Q16‑06 → Q16‑13 → Q16‑14 → Q16‑17 → Q16‑21

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 15 — Mathematical Foundations](../section-15-mathematical-foundations/README.md) &nbsp;|&nbsp; [Section 17 — Fine-Tuning, PEFT & Adaptation ➡️](../section-17-fine-tuning-peft-adaptation/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s16qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
