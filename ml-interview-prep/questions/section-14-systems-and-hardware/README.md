<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 13 — Diffusion, DiT & Generative Modeling](../section-13-diffusion-dit-generative-modeling/README.md) &nbsp;|&nbsp; [Section 15 — Mathematical Foundations ➡️](../section-15-mathematical-foundations/README.md)

</div>

---

# Section 14 · Systems and Hardware

<div align="center">

![Questions](https://img.shields.io/badge/questions-37-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F37-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-37%2F37-6c4fe0)

</div>

Systems and hardware knowledge distinguishes senior ML engineers from researchers. This section covers **GPU memory hierarchy** (HBM/SRAM/DRAM), **arithmetic intensity and the roofline model**, **5D parallelism** for large-scale training, **CUDA/Hopper-specific features**, **fault tolerance**, and the economics of AI compute.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q14‑01 | Walk through a single forward pass on a GPU at the hardware level — what hits HBM, what hits SRAM, where do tensor cores come in? | GPU · HBM · SRAM · tensor-cores | 📝 |
| Q14‑02 | What is the difference between TFLOPS and effective throughput? Why is utilization often <50%? | TFLOPS · throughput · MFU · utilization | 📝 |
| Q14‑03 | Explain arithmetic intensity and the roofline model. Why does it matter for LLM inference? | arithmetic-intensity · roofline · memory-bound · compute-bound | 📝 |
| Q14‑04 | Compare H100, B200, MI300X, and TPU v5p at a high level. What's each optimized for? | H100 · B200 · MI300X · TPU · comparison | 📝 |
| Q14‑05 | What does NCCL do, and why does it matter? | NCCL · all-reduce · collective · communication | 📝 |
| Q14‑06 | Explain HBM vs SRAM vs DRAM. What are the bandwidth and capacity tradeoffs? | HBM · SRAM · DRAM · bandwidth · capacity | 📝 |
| Q14‑07 | What is the difference between PCIe, NVLink, and InfiniBand? | PCIe · NVLink · InfiniBand · interconnect | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q14‑08 | Walk through tensor parallelism for an attention layer. Where does the all-reduce land, and why? | tensor-parallelism · all-reduce · attention · sharding | 📝 |
| Q14‑09 | Discuss the FP8 training story (Transformer Engine, DeepSeek-V3). Why is FP8 hard, and what does delayed scaling buy you? | FP8 · Transformer-Engine · DeepSeek-V3 · delayed-scaling | 📝 |
| Q14‑10 | Compare InfiniBand and NVLink for multi-node training. When does each dominate? | InfiniBand · NVLink · multi-node · bandwidth | 📝 |
| Q14‑11 | Discuss CPU offloading (DeepSpeed Zero-Infinity, FSDP CPU offload). When is the PCIe bandwidth not the bottleneck? | CPU-offload · Zero-Infinity · FSDP · PCIe | 📝 |
| Q14‑12 | What is a compiler stack for ML (XLA, TorchInductor, TVM)? Why does fusion matter so much? | XLA · TorchInductor · TVM · operator-fusion | 📝 |
| Q14‑13 | Explain Hopper-specific features (TMA, async copies, distributed shared memory). How do they change kernel design vs Ampere? | Hopper · TMA · async-copies · DSMEM | 📝 |
| Q14‑14 | Discuss the role of CUDA streams and graph capture in inference latency optimization. | CUDA-streams · graph-capture · latency · overlap | 📝 |
| Q14‑15 | What is sequence parallelism? Why is it complementary to tensor parallelism for long-context training? | sequence-parallelism · tensor-parallelism · long-context · CP | 📝 |
| Q14‑16 | Compare ring all-reduce and tree all-reduce. When does each win on a given topology? | ring-all-reduce · tree-all-reduce · topology · bandwidth | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q14‑17 | Design the training infrastructure for a 1T-parameter dense model on 16K H100s. Walk through 5D parallelism (DP × TP × PP × CP × EP), network topology, and NaN debugging. | 1T-model · 5D-parallelism · 16K-H100s · topology | 📝 |
| Q14‑18 | Discuss the economics of LLM inference. What's the bottleneck at scale — compute, HBM bandwidth, or networking? Where do TPUs and Groq-style architectures change the calculus? | inference-economics · HBM-bandwidth · TPU · Groq | 📝 |
| Q14‑19 | Walk through how Google's TPU pod topology (3D torus, OCS) differs from a GPU cluster. What workloads does each favor? | TPU · 3D-torus · OCS · GPU-cluster | 📝 |
| Q14‑20 | Critique the current hardware-software co-design landscape. Is the Transformer the right inductive bias for current hardware? What does a chip designed for SSMs/Mamba look like? | hardware-software · co-design · SSM-chip · Transformer-bias | 📝 |
| Q14‑21 | Discuss the design of distributed checkpointing at scale. How do async checkpointing, hierarchical storage, and in-memory checkpointing interact? | distributed-checkpointing · async · hierarchical · in-memory | 📝 |
| Q14‑22 | Walk through the failure modes of large-scale training (silent data corruption, link flaps, stragglers). What does production-grade fault tolerance look like? | fault-tolerance · SDC · stragglers · link-flaps | 📝 |
| Q14‑23 | Discuss memory-efficient training algorithms (activation checkpointing, selective recomputation, LOMO, GaLore). What's the wall-clock vs memory tradeoff? | activation-checkpointing · LOMO · GaLore · memory-efficiency | 📝 |
| Q14‑24 | Compare the systems implications of dense vs MoE vs hybrid (Mamba-Transformer) architectures for training and inference. | dense · MoE · hybrid · systems-comparison | 📝 |
| Q14‑25 | Walk through the design of a custom kernel for grouped matrix multiplication (used in MoE expert FFN). What are the key optimizations on H100? | grouped-matmul · MoE · H100 · kernel-design | 📝 |
| Q14‑26 | Discuss the role of compiler-level optimizations (operator fusion, kernel autotuning) vs hand-tuned kernels (CUTLASS, ThunderKittens). When does each pay off? | operator-fusion · autotuning · CUTLASS · ThunderKittens | 📝 |
| Q14‑27 | Critique current cluster designs for AI training. What would an AI-native datacenter look like (cooling, power, network)? | AI-datacenter · cooling · power · network-design | 📝 |
| Q14‑28 | Discuss reliability engineering for trillion-parameter training. What's the MTBF, and how do you achieve 95%+ effective utilization across weeks of training? | reliability · MTBF · utilization · trillion-parameter | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q14‑29 | Your large training run is hitting 38% MFU when the theoretical optimum is 55%. Walk through your investigation and the top 3 interventions. | MFU · 38-percent · debugging · interventions | 📝 |
| Q14‑30 | Walk through how you'd debug a NaN that appears once every 50K steps on a 1024-GPU training run. | NaN · 50K-steps · 1024-GPU · debugging | 📝 |
| Q14‑31 | You're asked to migrate a training run from H100 to B200 mid-project. Walk through what you'd test and the risks you'd flag. | H100-to-B200 · migration · risks · testing | 📝 |
| Q14‑32 | Your training run has a straggler that's 8% slower than peers. Walk through your investigation and mitigation. | straggler · 8-percent · investigation · mitigation | 📝 |
| Q14‑33 | Walk through how you'd design checkpointing for a 1T-parameter model with a 6-hour MTBF cluster. | checkpointing · 1T-model · 6-hour-MTBF · design | 📝 |
| Q14‑34 | Your team has a $50M budget for cluster expansion. Walk through how you'd choose between adding more H100s vs investing in B200s. | cluster-expansion · H100-vs-B200 · budget · ROI | 📝 |
| Q14‑35 | You discover that a recent NCCL upgrade has degraded all-reduce performance by 5%. Walk through your investigation. | NCCL · all-reduce · performance-regression · debugging | 📝 |
| Q14‑36 | Walk through how you'd architect inference serving across a mixed H100/A100 fleet to maximize cost efficiency. | mixed-fleet · H100 · A100 · cost-efficiency | 📝 |
| Q14‑37 | Your training run experiences a network partition that splits the cluster in half for 4 minutes. Walk through your recovery plan and prevention strategy. | network-partition · cluster · recovery · prevention | 📝 |

---

## Why these questions matter

Systems questions are weighted heavily at senior ML roles. These questions test whether a candidate can reason about **hardware constraints, parallelism strategies, and production reliability** — not just model architecture.

| Theme | Key questions |
|-------|--------------|
| **GPU fundamentals** | Q14‑01, Q14‑02, Q14‑03, Q14‑06 — HBM, roofline, MFU |
| **Parallelism** | Q14‑08, Q14‑15, Q14‑17, Q14‑19 — TP, SP, 5D, TPU topology |
| **Fault tolerance** | Q14‑21, Q14‑22, Q14‑30, Q14‑33 — checkpointing, NaN, MTBF |
| **Kernels & compilers** | Q14‑12, Q14‑13, Q14‑25, Q14‑26 — fusion, Hopper, CUTLASS |

---

## Reading order suggestions

**New to systems?** Q14‑01 → Q14‑02 → Q14‑03 → Q14‑06 → Q14‑07

**Interview in 48 hours?** Q14‑17 (5D parallelism) · Q14‑03 (roofline) · Q14‑13 (Hopper) · Q14‑18 (inference economics) · Q14‑22 (fault tolerance)

**Full track:** Q14‑01 → Q14‑07 → Q14‑08 → Q14‑16 → Q14‑17 → Q14‑22 → Q14‑28

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 13 — Diffusion, DiT & Generative Modeling](../section-13-diffusion-dit-generative-modeling/README.md) &nbsp;|&nbsp; [Section 15 — Mathematical Foundations ➡️](../section-15-mathematical-foundations/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s14qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
