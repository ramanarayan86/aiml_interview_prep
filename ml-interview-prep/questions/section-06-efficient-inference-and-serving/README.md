<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 5 — Reasoning, CoT & Inference-Time Compute](../section-05-reasoning-cot-inference-time-compute/README.md) &nbsp;|&nbsp; [Section 7 — Quantization & Model Compression ➡️](../section-07-quantization-and-model-compression/README.md)

</div>

---

# Section 6 · Efficient Inference and Serving

<div align="center">

![Questions](https://img.shields.io/badge/questions-40-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F40-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-40%2F40-6c4fe0)

</div>

Serving LLMs efficiently at scale is an engineering discipline in its own right. This section covers the **KV cache** (the dominant memory cost), **prefill vs decode phases**, **speculative decoding**, **continuous batching**, **PagedAttention**, and the full systems-design space from single-GPU to multi-node serving.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q6‑01 | What is the KV cache? Why does it dominate memory at inference time? | KV-cache · memory · attention · caching | 📝 |
| Q6‑02 | Explain prefill vs decode phases. Why are their compute profiles so different (compute-bound vs memory-bound)? | prefill · decode · compute-bound · memory-bound | 📝 |
| Q6‑03 | What is speculative decoding at a high level? | speculative-decoding · draft-model · acceptance · speedup | 📝 |
| Q6‑04 | Why does batch size affect throughput dramatically more during decode than prefill? | batch-size · decode · throughput · memory-bandwidth | 📝 |
| Q6‑05 | What is continuous batching (a la vLLM/Orca)? How does it improve GPU utilization? | continuous-batching · vLLM · Orca · utilization | 📝 |
| Q6‑06 | What is TTFT (time-to-first-token) and TPOT (time-per-output-token)? Why are they tracked separately? | TTFT · TPOT · latency · metrics | 📝 |
| Q6‑07 | Explain greedy, top-k, top-p (nucleus), and temperature sampling. When does each matter? | greedy · top-k · top-p · temperature · sampling | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q6‑08 | Derive the memory footprint of the KV cache for a Llama-3 70B model serving 1000 concurrent users at 8K context. | KV-cache · memory-footprint · Llama-3 · derivation | 📝 |
| Q6‑09 | Walk through PagedAttention. How does it borrow from OS virtual memory? | PagedAttention · virtual-memory · vLLM · paging | 📝 |
| Q6‑10 | Speculative decoding requires acceptance — derive the expected speedup as a function of draft model acceptance rate. When does it become net-negative? | speculative-decoding · acceptance-rate · speedup · derivation | 📝 |
| Q6‑11 | Compare Medusa, EAGLE, and Lookahead Decoding to vanilla speculative decoding with a draft model. | Medusa · EAGLE · Lookahead · speculative-decoding | 📝 |
| Q6‑12 | What is chunked prefill and why does it help mixed prefill/decode batches? | chunked-prefill · mixed-batching · latency · utilization | 📝 |
| Q6‑13 | Discuss prefix caching. How does it interact with multi-turn conversations and RAG? | prefix-caching · multi-turn · RAG · cache-hit | 📝 |
| Q6‑14 | Explain how a Transformer inference engine handles dynamic batching with variable sequence lengths. What are the masking and padding costs? | dynamic-batching · padding · masking · variable-length | 📝 |
| Q6‑15 | What is the role of CUDA graphs in low-latency LLM inference? When do they help and when do they hurt? | CUDA-graphs · latency · graph-capture · dynamic-shapes | 📝 |
| Q6‑16 | Discuss the design of an efficient softmax kernel. Why is online softmax (used in FlashAttention) preferable? | softmax-kernel · online-softmax · FlashAttention · numerical-stability | 📝 |
| Q6‑17 | Compare TensorRT-LLM, vLLM, SGLang, and TGI as serving frameworks. What does each optimize for? | TensorRT-LLM · vLLM · SGLang · TGI · comparison | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q6‑18 | Design an inference system serving a 405B model across 8 H100s with sub-second TTFT for 32K context. Walk through tensor parallel sharding, pipeline parallel tradeoffs, and KV cache placement. | 405B · tensor-parallel · pipeline-parallel · KV-placement | 📝 |
| Q6‑19 | KV cache compression — compare quantization (KIVI), eviction (H2O, StreamingLLM), and low-rank (MLA from DeepSeek-V2). What does Multi-Head Latent Attention buy you and what does it cost? | KV-compression · KIVI · H2O · MLA · DeepSeek-V2 | 📝 |
| Q6‑20 | Discuss disaggregated prefill/decode serving (Splitwise, DistServe). Why separate the phases onto different hardware? | disaggregated · Splitwise · DistServe · prefill-decode | 📝 |
| Q6‑21 | Walk through how you'd build a request scheduler that handles SLO-aware prioritization, fairness across users, and prefix caching. What are the key decisions? | scheduler · SLO · fairness · prefix-caching | 📝 |
| Q6‑22 | Critique current LLM serving — what's the theoretical efficiency ceiling, and what architectural innovations might break it? | efficiency-ceiling · hybrid-SSM · cross-layer-attention · innovation | 📝 |
| Q6‑23 | Discuss the tradeoffs of tensor parallelism vs pipeline parallelism at inference time. Why does PP rarely make sense for low-latency serving? | TP · PP · latency · inference-parallelism | 📝 |
| Q6‑24 | Walk through how you would design a multi-LoRA serving system (S-LoRA, Punica). What changes in the attention/FFN kernels? | multi-LoRA · S-LoRA · Punica · kernel-design | 📝 |
| Q6‑25 | Discuss the challenge of long-context inference. How do prefix caching, KV compression, and sparse attention interact? | long-context · prefix-caching · KV-compression · sparse-attention | 📝 |
| Q6‑26 | What is the role of CPU memory in LLM inference (offload, swap, DRAM-tiered KV cache)? When is the PCIe bandwidth not the bottleneck? | CPU-offload · DRAM-tiered · PCIe · memory-hierarchy | 📝 |
| Q6‑27 | Design an inference system that achieves p99 latency guarantees under bursty traffic. How do you handle head-of-line blocking? | p99 · bursty-traffic · SLO · head-of-line | 📝 |
| Q6‑28 | Discuss tree attention and parallel sampling. How can you batch multiple sequences sharing a common prefix efficiently? | tree-attention · parallel-sampling · shared-prefix · batching | 📝 |
| Q6‑29 | Walk through the inference economics of reasoning models (o1/R1). When does spending more on inference beat spending more on training? | inference-economics · reasoning · o1 · compute-tradeoff | 📝 |
| Q6‑30 | Critique the current state of LLM inference benchmarking. What does MLPerf miss, and what would a workload-realistic benchmark look like? | MLPerf · benchmarking · inference · workload-realistic | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q6‑31 | Your inference service has p99 TTFT of 3 seconds at peak. Walk through your investigation and the top 3 interventions you'd try. | TTFT · p99 · latency · debugging | 📝 |
| Q6‑32 | You're running a 70B model on 4×A100 and getting 40 tokens/sec single-stream. The team needs 80. Walk through what you'd profile and change. | throughput · A100 · profiling · optimization | 📝 |
| Q6‑33 | Memory pressure is causing KV cache evictions and request failures during peak hours. Walk through your short-term, medium-term, and long-term mitigation plan. | KV-eviction · memory-pressure · mitigation · serving | 📝 |
| Q6‑34 | You're asked to add speculative decoding to a serving stack. Walk through your draft model choice, acceptance-rate measurement, and rollout strategy. | speculative-decoding · draft-model · rollout · production | 📝 |
| Q6‑35 | Your serving system handles 80% short requests (≤1K tokens) and 20% long ones (≥32K tokens). Walk through your batching and scheduling design. | mixed-workload · batching · scheduling · long-context | 📝 |
| Q6‑36 | Walk through how you'd debug a sudden 30% throughput drop in production after a routine kernel update. | throughput-regression · kernel-update · debugging · profiling | 📝 |
| Q6‑37 | You need to serve 50 fine-tuned LoRA variants of the same base model. Walk through your architecture and the kernel/scheduler changes vs serving 50 separate models. | multi-LoRA · serving · S-LoRA · architecture | 📝 |
| Q6‑38 | A team wants to deploy a model with 'guaranteed' p99 TTFT under 500ms. Walk through what guarantees you can and can't make, and what infrastructure you'd require. | SLO · p99 · 500ms · guarantees | 📝 |
| Q6‑39 | Your serving cluster has 8 nodes; one is producing slightly higher-perplexity outputs than the others. Walk through how you'd isolate whether it's hardware, model state, or routing. | cluster · perplexity-mismatch · hardware · debugging | 📝 |

---

## Why these questions matter

Inference serving is the **production bottleneck** at every AI company. These questions test whether a candidate can reason about memory hierarchies, latency budgets, and system tradeoffs — not just model architecture.

| Theme | Key questions |
|-------|--------------|
| **KV cache & memory** | Q6‑01, Q6‑08, Q6‑19, Q6‑33 — footprint, compression, eviction |
| **Speculative decoding** | Q6‑03, Q6‑10, Q6‑11, Q6‑34 — mechanics, speedup, variants |
| **Batching & scheduling** | Q6‑04, Q6‑05, Q6‑12, Q6‑21 — continuous batching, SLO, chunked prefill |
| **Systems design** | Q6‑18, Q6‑20, Q6‑27, Q6‑38 — 405B serving, disaggregated, p99 guarantees |

---

## Reading order suggestions

**New to inference?** Q6‑01 → Q6‑02 → Q6‑04 → Q6‑05 → Q6‑09

**Interview in 48 hours?** Q6‑08 (KV footprint) · Q6‑10 (speculative decoding) · Q6‑09 (PagedAttention) · Q6‑18 (405B system design)

**Full track:** Q6‑01 → Q6‑07 → Q6‑08 → Q6‑17 → Q6‑18 → Q6‑19 → Q6‑20

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 5 — Reasoning, CoT & Inference-Time Compute](../section-05-reasoning-cot-inference-time-compute/README.md) &nbsp;|&nbsp; [Section 7 — Quantization & Model Compression ➡️](../section-07-quantization-and-model-compression/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s6qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
