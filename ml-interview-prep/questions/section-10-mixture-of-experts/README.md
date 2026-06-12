<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 9 — Retrieval, RAG & Long Context](../section-09-retrieval-rag-long-context/README.md) &nbsp;|&nbsp; [Section 11 — Agents, Tool Use & Code Generation ➡️](../section-11-agents-tool-use-code-generation/README.md)

</div>

---

# Section 10 · Mixture of Experts (MoE)

<div align="center">

![Questions](https://img.shields.io/badge/questions-30-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F30-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-30%2F30-6c4fe0)

</div>

Mixture of Experts decouples parameter count from compute, enabling trillion-parameter models at tractable training and inference cost. This section covers **sparse routing fundamentals**, the **Switch/Mixtral/DeepSeek-V3 lineage**, **load balancing**, **expert parallelism**, and the scaling and quantization properties unique to MoE.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q10‑01 | What is a sparse MoE layer? Why does it decouple parameters from FLOPs? | sparse-MoE · parameters · FLOPs · routing | 📝 |
| Q10‑02 | Explain top-k routing. Why is k=1 or k=2 typical? | top-k · routing · expert-selection · k=2 | 📝 |
| Q10‑03 | What is the load balancing loss? Why is it necessary? | load-balancing · auxiliary-loss · routing-collapse | 📝 |
| Q10‑04 | Compare dense models and MoE models at the same active-parameter count. Where does MoE win? | dense-vs-MoE · active-params · quality · inference-cost | 📝 |
| Q10‑05 | What is the difference between an expert and an FFN block? Why is the expert typically the FFN? | expert · FFN · sparse-activation · MoE | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q10‑06 | Walk through Switch Transformer, GShard, and Mixtral 8x7B. What evolved across these? | Switch · GShard · Mixtral · evolution | 📝 |
| Q10‑07 | Discuss expert collapse and routing instability. How do auxiliary losses, expert capacity, and noisy routing mitigate it? | expert-collapse · routing-instability · capacity · noise | 📝 |
| Q10‑08 | Compare token choice routing vs expert choice routing. When does each shine? | token-choice · expert-choice · routing · comparison | 📝 |
| Q10‑09 | Explain DeepSeek-V3's MoE design — fine-grained experts, shared experts, and auxiliary-loss-free load balancing. Why was this important? | DeepSeek-V3 · fine-grained · shared-experts · auxiliary-free | 📝 |
| Q10‑10 | What's the inference cost story for MoE? Why is memory the dominant constraint? | inference-cost · memory · expert-weights · MoE | 📝 |
| Q10‑11 | Discuss expert parallelism as a parallelism strategy. How does the all-to-all communication pattern affect training throughput? | expert-parallelism · all-to-all · communication · throughput | 📝 |
| Q10‑12 | Explain the soft MoE (Soft MoE, SMEAR) approach. Why does mixing experts rather than routing avoid load balancing issues? | soft-MoE · SMEAR · mixing · differentiable | 📝 |
| Q10‑13 | What is the role of router temperature and noise during training? | router-temperature · noise · exploration · routing | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q10‑14 | Design an inference system for a 1T-parameter MoE with 32B active parameters. How do you exploit expert sparsity for batched inference (expert parallelism, expert caching)? | 1T-MoE · expert-caching · batched-inference · parallelism | 📝 |
| Q10‑15 | Discuss MoE upcycling — converting a dense checkpoint into an MoE (Sparse Upcycling, MoE-Mamba). What are the benefits and pitfalls? | upcycling · Sparse-Upcycling · MoE-Mamba · dense-to-MoE | 📝 |
| Q10‑16 | Critique MoE as a path to scaling. What's the ceiling? Discuss communication costs, routing entropy collapse, and whether MoE generalizes worse than dense per active-param-flop. | MoE-ceiling · entropy-collapse · generalization · dense-comparison | 📝 |
| Q10‑17 | Walk through how you'd quantize an MoE model. What's different about expert weights vs shared weights? | MoE-quantization · expert-weights · GPTQ · rare-experts | 📝 |
| Q10‑18 | Discuss the theoretical underpinnings — is MoE doing implicit conditional computation, or is it closer to a learned ensemble? What does mechanistic interpretability of routers tell us? | conditional-computation · learned-ensemble · routing · interpretability | 📝 |
| Q10‑19 | Compare MoE scaling laws to dense scaling laws. Reference 'Scaling Laws for Fine-Grained MoE' and discuss the granularity-experts tradeoff. | MoE-scaling-laws · fine-grained · granularity · dense-comparison | 📝 |
| Q10‑20 | Discuss the engineering of training a 1T+ MoE. What are the dominant failure modes (expert imbalance, router NaN, dropped tokens)? | 1T-training · expert-imbalance · router-NaN · dropped-tokens | 📝 |
| Q10‑21 | Design a routing function that is robust to distribution shift at inference time. What does the recent literature suggest about routing brittleness? | routing-robustness · distribution-shift · brittleness · inference | 📝 |
| Q10‑22 | Walk through the design of MoE for VLMs. Should the vision encoder, projector, and LLM all be MoE, and why? | MoE-VLM · vision-encoder · projector · design | 📝 |
| Q10‑23 | Discuss continual learning in MoE — can you add experts incrementally without retraining? Reference Lifelong-MoE and follow-ups. | continual-learning · incremental-experts · Lifelong-MoE | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q10‑24 | Your MoE training shows one expert receiving 40% of routed tokens. Walk through your investigation and fix. | expert-imbalance · routing · investigation · fix | 📝 |
| Q10‑25 | You're choosing between a 70B dense and a 200B/30B-active MoE for the same compute budget. Walk through the tradeoffs you'd present to leadership. | dense-vs-MoE · compute-budget · tradeoffs · decision | 📝 |
| Q10‑26 | Walk through how you'd debug a sudden drop in inference throughput on an MoE model after expert quantization. | throughput-drop · expert-quantization · debugging | 📝 |
| Q10‑27 | Your MoE model achieves the same loss as the dense baseline but underperforms on instruction-following. Walk through your investigation. | instruction-following · MoE · dense-comparison · debugging | 📝 |
| Q10‑28 | Walk through how you'd design inference serving for a 1T-parameter MoE on a heterogeneous cluster. | 1T-serving · heterogeneous · expert-parallelism · design | 📝 |
| Q10‑29 | You inherit an MoE codebase where expert utilization is unstable. Walk through your stabilization plan. | utilization · instability · stabilization · MoE | 📝 |
| Q10‑30 | A team proposes upcycling a dense 7B into an MoE. Walk through the experiments you'd require and the risks you'd flag. | upcycling · 7B · experiments · risks | 📝 |

---

## Why these questions matter

MoE is the architecture behind GPT-4, Mixtral, and DeepSeek-V3. These questions test whether a candidate understands the **routing mechanics, training pitfalls, inference economics, and systems engineering** of sparse models.

| Theme | Key questions |
|-------|--------------|
| **Routing fundamentals** | Q10‑01, Q10‑02, Q10‑03, Q10‑13 — top-k, load balancing, temperature |
| **Architecture evolution** | Q10‑06, Q10‑09, Q10‑12 — Switch, Mixtral, DeepSeek-V3, soft MoE |
| **Systems & inference** | Q10‑10, Q10‑11, Q10‑14, Q10‑28 — memory, expert parallelism |
| **Failure modes** | Q10‑07, Q10‑20, Q10‑24, Q10‑29 — collapse, imbalance, NaN |

---

## Reading order suggestions

**New to MoE?** Q10‑01 → Q10‑02 → Q10‑03 → Q10‑05 → Q10‑06

**Interview in 48 hours?** Q10‑09 (DeepSeek-V3) · Q10‑06 (Switch/Mixtral) · Q10‑14 (1T inference) · Q10‑16 (MoE critique) · Q10‑19 (scaling laws)

**Full track:** Q10‑01 → Q10‑05 → Q10‑06 → Q10‑13 → Q10‑14 → Q10‑16 → Q10‑19

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 9 — Retrieval, RAG & Long Context](../section-09-retrieval-rag-long-context/README.md) &nbsp;|&nbsp; [Section 11 — Agents, Tool Use & Code Generation ➡️](../section-11-agents-tool-use-code-generation/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s10qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
