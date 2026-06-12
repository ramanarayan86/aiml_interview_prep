<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 6 — Efficient Inference & Serving](../section-06-efficient-inference-and-serving/README.md) &nbsp;|&nbsp; [Section 8 — Vision-Language Models ➡️](../section-08-vision-language-models/README.md)

</div>

---

# Section 7 · Quantization and Model Compression

<div align="center">

![Questions](https://img.shields.io/badge/questions-46-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F46-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-46%2F46-6c4fe0)

</div>

Quantization is the primary lever for deploying large models on constrained hardware. This section covers everything from **basic INT8/INT4 concepts** to **GPTQ, AWQ, SmoothQuant**, the **FP4/MXFP4 frontier**, and advanced topics like **rotation-based outlier suppression**, **MoE quantization**, and **extreme 1-bit models**.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q7‑01 | Why does quantization work at all for LLMs? What property of weight distributions makes it possible? | quantization · weight-distributions · Gaussian · PTQ | 📝 |
| Q7‑02 | Explain symmetric vs asymmetric (zero-point) quantization. When does each shine? | symmetric · asymmetric · zero-point · quantization | 📝 |
| Q7‑03 | What is the difference between PTQ and QAT? When would you choose each? | PTQ · QAT · calibration · training | 📝 |
| Q7‑04 | What are outliers in LLM activations and why do they make quantization hard? | outliers · activations · quantization-difficulty · LLM.int8 | 📝 |
| Q7‑05 | Explain per-tensor vs per-channel vs per-group quantization. What are the tradeoffs? | per-tensor · per-channel · per-group · granularity | 📝 |
| Q7‑06 | What is the difference between weight-only and weight+activation quantization? | weight-only · W+A · activation-quantization · inference | 📝 |
| Q7‑07 | Define bits-per-weight (bpw) and explain why it's not the same as the nominal precision (e.g., 4-bit NF4 ≠ 4.0 bpw). | bpw · NF4 · nominal-precision · effective-bits | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q7‑08 | Walk through GPTQ. Why does the OBS (Optimal Brain Surgeon) formulation matter? What does the inverse Hessian capture? | GPTQ · OBS · Hessian · weight-quantization | 📝 |
| Q7‑09 | Explain AWQ. Why does activation-aware scaling outperform uniform quantization? | AWQ · activation-aware · scaling · per-channel | 📝 |
| Q7‑10 | Compare INT4, NF4, FP4 (E2M1), and MXFP4. Why has the field moved toward floating-point 4-bit? | INT4 · NF4 · FP4 · MXFP4 · comparison | 📝 |
| Q7‑11 | What is SmoothQuant? How does the activation-to-weight migration of quantization difficulty work? | SmoothQuant · activation-migration · per-channel · difficulty | 📝 |
| Q7‑12 | Discuss MXFP4 and the OCP Microscaling spec. What's the role of the shared scale (E8M0) and group size? | MXFP4 · Microscaling · E8M0 · shared-scale | 📝 |
| Q7‑13 | Compare GPTQ, AWQ, and RTN (round-to-nearest). When is the gap large vs small? | GPTQ · AWQ · RTN · comparison | 📝 |
| Q7‑14 | Explain the role of the calibration dataset in PTQ. What happens if it's distributionally mismatched with deployment? | calibration · PTQ · distribution-mismatch · quality | 📝 |
| Q7‑15 | Discuss QLoRA. How does it combine 4-bit base weights with LoRA adapters, and why does it work? | QLoRA · 4-bit · LoRA · fine-tuning | 📝 |
| Q7‑16 | Explain double quantization (used in QLoRA). What's quantized and what's the bpw saving? | double-quantization · QLoRA · bpw · memory | 📝 |
| Q7‑17 | Compare quantization for weights, activations, KV cache, and gradients. Why is each problem qualitatively different? | weights · activations · KV-cache · gradients · comparison | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q7‑18 | Derive the SQNR difference between INT4 and FP4 (E2M1) for a Gaussian weight distribution. Why does FP4 win for heavy-tailed distributions? | SQNR · INT4 · FP4 · Gaussian · heavy-tail | 📝 |
| Q7‑19 | Discuss NVFP4 vs MXFP4. What does the second-level FP8 scale buy you, and at what bpw cost? | NVFP4 · MXFP4 · FP8-scale · bpw | 📝 |
| Q7‑20 | Discuss sliding-window FP4 variants — how does an adaptive per-block codebook (NF4-style) compare to a fixed FP4 grid with a learned offset? | sliding-window · FP4 · codebook · adaptive | 📝 |
| Q7‑21 | Design a quantization scheme that handles attention KV cache, weights, and activations together. What's the joint optimization problem? | joint-quantization · KV-cache · weights · activations | 📝 |
| Q7‑22 | Walk through quantization for MoE models — what's special about expert weight distributions, and why does GPTQ struggle on rarely-activated experts? | MoE · expert-quantization · GPTQ · rare-experts | 📝 |
| Q7‑23 | Critique the FP4 frontier — at what point does quantization noise dominate the signal? Reference Apple's 'Scaling Laws for Precision' work and discuss whether sub-4-bit is fundamentally different. | FP4 · quantization-noise · BitNet · 1.58-bit | 📝 |
| Q7‑24 | Discuss rotation-based quantization (QuaRot, SpinQuant). Why do Hadamard/random orthogonal rotations make outliers tractable? What's the theoretical basis (incoherence)? | QuaRot · SpinQuant · Hadamard · incoherence · outliers | 📝 |
| Q7‑25 | Walk through RL-based per-layer codebook selection for FP4. What's the action space, reward, and credit assignment problem? | RL · codebook-selection · FP4 · per-layer | 📝 |
| Q7‑26 | Discuss the relationship between Hadamard rotations, JL sketching, and polar decomposition for outlier suppression. | Hadamard · JL-lemma · polar-decomposition · outlier-suppression | 📝 |
| Q7‑27 | Compare distribution-aware (E2M1 with offset, uE2M2) and distribution-agnostic (RTN, GPTQ) quantization. What does the empirical evidence say? | distribution-aware · distribution-agnostic · empirical | 📝 |
| Q7‑28 | Discuss quantization of diffusion transformers. Why is DiT quantization harder than LLM quantization? | DiT · diffusion · timestep-dependent · quantization | 📝 |
| Q7‑29 | Walk through 1-bit and 1.58-bit quantization (BitNet, BitNet b1.58). What architectural changes are required? | BitNet · 1-bit · 1.58-bit · extreme-quantization | 📝 |
| Q7‑30 | Critique the SQNR metric as a proxy for downstream task quality. Where does it correlate well, and where does it fail? | SQNR · metric · downstream-quality · reasoning | 📝 |
| Q7‑31 | Design an end-to-end quantization recipe for a frontier model targeting deployment on consumer GPUs (24GB VRAM). | consumer-GPU · 24GB · recipe · mixed-precision | 📝 |
| Q7‑32 | Discuss quantization-aware training vs post-training quantization at scale. Why has PTQ dominated despite QAT's theoretical advantages? | QAT · PTQ · scale · convergence | 📝 |
| Q7‑33 | Walk through pruning as an alternative/complement to quantization. Compare magnitude pruning, Wanda, and SparseGPT. When does sparsity actually translate to inference speedup? | pruning · Wanda · SparseGPT · sparsity · speedup | 📝 |
| Q7‑34 | Discuss the interaction between quantization and long context. Why does KV cache quantization become more sensitive at 100K+ context? | long-context · KV-quantization · 100K · sensitivity | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q7‑35 | Your INT4 quantized model loses 2.3 points on MMLU. Walk through your debugging tree — which layers do you investigate first, and when do you give up on INT4? | INT4 · MMLU · debugging · layer-sensitivity | 📝 |
| Q7‑36 | You're asked to compress a 70B model to fit on 24GB consumer GPUs. Walk through your bpw budget allocation across weights, KV cache, and activations. | 70B · 24GB · bpw-budget · consumer-GPU | 📝 |
| Q7‑37 | A new FP4 variant claims +1 dB SQNR over MXFP4 on synthetic weights but only +0.2 on real LLMs. Walk through how you'd diagnose the gap. | FP4 · SQNR-gap · synthetic-vs-real · diagnosis | 📝 |
| Q7‑38 | Your AWQ-quantized model has a 0.5-point perplexity regression on a specific eval split but matches everywhere else. Walk through your investigation. | AWQ · perplexity-regression · eval-split · debugging | 📝 |
| Q7‑39 | Walk through how you'd benchmark a new quantization scheme against NVFP4, MXFP4, and AWQ in a way that produces publishable comparisons. | benchmarking · NVFP4 · AWQ · publishable | 📝 |
| Q7‑40 | You're given a budget of 1 month to deliver a 4-bit serving recipe for a new model family. Walk through your milestones and critical decisions. | 4-bit · recipe · milestones · project-planning | 📝 |
| Q7‑41 | Walk through how you'd quantize attention KV cache without hurting long-context retrieval. What ablations would you run? | KV-quantization · long-context · retrieval · ablations | 📝 |
| Q7‑42 | A team proposes 2-bit quantization with QAT. Walk through what you'd measure to decide whether to fund a 3-month project. | 2-bit · QAT · feasibility · decision-criteria | 📝 |
| Q7‑43 | Your quantized model performs well on standard benchmarks but fails on a new reasoning benchmark. Walk through how you'd determine whether quantization or the base model is at fault. | reasoning · quantization-fault · attribution · debugging | 📝 |
| Q7‑44 | Walk through how you'd design A/B testing infrastructure to detect quantization-induced regressions that don't show up on offline benchmarks. | A/B-testing · regression · offline-benchmark · production | 📝 |
| Q7‑45 | You discover that quantization-induced errors disproportionately affect rare tokens. Walk through your detection method and mitigation options. | rare-tokens · quantization-error · detection · mitigation | 📝 |
| Q7‑46 | Walk through the engineering of a kernel for an FP4 GEMM with E8M0 shared scales. What are the dominant performance constraints on H100? | FP4-GEMM · E8M0 · H100 · kernel-engineering | 📝 |

---

## Why these questions matter

Quantization is a core skill at every inference-focused team. These questions test whether a candidate can navigate the **full stack** — from the math of SQNR and FP4 formats, to practical PTQ recipes, to production debugging.

| Theme | Key questions |
|-------|--------------|
| **Formats & theory** | Q7‑07, Q7‑10, Q7‑18, Q7‑30 — bpw, INT4 vs FP4, SQNR |
| **PTQ algorithms** | Q7‑08, Q7‑09, Q7‑11, Q7‑13 — GPTQ, AWQ, SmoothQuant |
| **Advanced schemes** | Q7‑24, Q7‑29, Q7‑32, Q7‑33 — rotation, BitNet, QAT, pruning |
| **Production** | Q7‑35, Q7‑36, Q7‑40, Q7‑44 — debugging, budget, recipes |

---

## Reading order suggestions

**New to quantization?** Q7‑01 → Q7‑02 → Q7‑03 → Q7‑05 → Q7‑08

**Interview in 48 hours?** Q7‑08 (GPTQ) · Q7‑09 (AWQ) · Q7‑10 (INT4 vs FP4) · Q7‑15 (QLoRA) · Q7‑24 (QuaRot)

**Full track:** Q7‑01 → Q7‑07 → Q7‑08 → Q7‑17 → Q7‑18 → Q7‑24 → Q7‑29

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 6 — Efficient Inference & Serving](../section-06-efficient-inference-and-serving/README.md) &nbsp;|&nbsp; [Section 8 — Vision-Language Models ➡️](../section-08-vision-language-models/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s7qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
