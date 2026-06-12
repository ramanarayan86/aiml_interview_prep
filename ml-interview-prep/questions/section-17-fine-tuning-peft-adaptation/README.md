<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 16 — Distributed Training & Optimization](../section-16-distributed-training-optimization/README.md) &nbsp;|&nbsp; [Section 18 — Speech, Audio & Multimodal ➡️](../section-18-speech-audio-multimodal/README.md)

</div>

---

# Section 17 · Fine-Tuning, PEFT, and Adaptation

<div align="center">

![Questions](https://img.shields.io/badge/questions-27-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F27-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-27%2F27-6c4fe0)

</div>

PEFT has made large-model fine-tuning accessible. This section covers **LoRA and its variants** (QLoRA, DoRA, LoRA+), **prefix and prompt tuning**, **adapter merging** (TIES, DARE), **multi-LoRA serving**, and the deep question of whether PEFT is a permanent paradigm or a transitional solution.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q17‑01 | What is the difference between full fine-tuning, adapter tuning, and prompt tuning? | full-FT · adapter · prompt-tuning · PEFT | 📝 |
| Q17‑02 | Explain LoRA. Why does the low-rank decomposition work? | LoRA · low-rank · decomposition · fine-tuning | 📝 |
| Q17‑03 | What is the role of the rank r and the scaling α in LoRA? How do you choose them? | LoRA · rank · alpha · hyperparameter | 📝 |
| Q17‑04 | Compare LoRA, QLoRA, DoRA, and LoRA+. What does each add? | LoRA · QLoRA · DoRA · LoRA+ · comparison | 📝 |
| Q17‑05 | What is catastrophic forgetting? How does PEFT mitigate it? | catastrophic-forgetting · PEFT · regularization · plasticity | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q17‑06 | Walk through prefix tuning and prompt tuning. Why does scale matter for prompt tuning to work? | prefix-tuning · prompt-tuning · scale · soft-prompts | 📝 |
| Q17‑07 | Discuss IA³ and BitFit. When are these extreme-PEFT methods competitive with LoRA? | IA3 · BitFit · extreme-PEFT · parameter-count | 📝 |
| Q17‑08 | Compare LoRA on attention layers only vs all linear layers. What's the empirical tradeoff? | LoRA · attention-layers · all-linear · empirical | 📝 |
| Q17‑09 | Explain merging adapters (LoRA Hub, TIES, DARE). Why does naive averaging often work? | adapter-merging · LoRA-Hub · TIES · DARE | 📝 |
| Q17‑10 | Discuss multi-LoRA serving. What changes in the kernel design (S-LoRA, Punica)? | multi-LoRA · S-LoRA · Punica · kernel | 📝 |
| Q17‑11 | What is the role of LoRA initialization (PiSSA, LoRA-GA)? Why does it affect final quality? | LoRA-init · PiSSA · LoRA-GA · initialization | 📝 |
| Q17‑12 | Compare LoRA and full fine-tuning on instruction-following tasks. When is the gap large vs small? | LoRA-vs-full-FT · instruction-following · gap | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q17‑13 | Discuss the theoretical justification for LoRA. What does the 'intrinsic dimension of fine-tuning' research suggest? | intrinsic-dimension · LoRA-theory · fine-tuning | 📝 |
| Q17‑14 | Critically evaluate PEFT methods on reasoning tasks. Does LoRA underperform full fine-tuning for math/code, and why? | PEFT · reasoning · math · code · LoRA-vs-full | 📝 |
| Q17‑15 | Design a PEFT recipe for adapting a 70B model to 10 domains simultaneously. How do you handle adapter selection at inference? | multi-domain · 70B · adapter-routing · inference | 📝 |
| Q17‑16 | Discuss model merging at scale (TIES, DARE, model soups). What's the theoretical basis, and what are the failure modes? | model-merging · TIES · DARE · model-soups · theory | 📝 |
| Q17‑17 | Walk through how you'd combine LoRA with quantization-aware training. What are the gradient propagation issues? | LoRA · QAT · gradient-propagation · combined | 📝 |
| Q17‑18 | Discuss continual learning with PEFT — can you add LoRA adapters indefinitely without forgetting? Reference O-LoRA and follow-ups. | continual-learning · O-LoRA · forgetting · incremental | 📝 |
| Q17‑19 | Critically evaluate whether PEFT is a stopgap or a permanent paradigm. When the cost of full FT drops, does PEFT survive? | PEFT-future · full-FT-cost · paradigm · permanent | 📝 |
| Q17‑20 | Walk through how you would adapt a base model into a multi-skill model using PEFT + routing. Compare to dense MoE. | multi-skill · PEFT-routing · dense-MoE · comparison | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q17‑21 | Your LoRA fine-tune matches full FT on win-rate but loses on reasoning benchmarks. Walk through your investigation. | LoRA · full-FT · reasoning-gap · investigation | 📝 |
| Q17‑22 | Walk through how you'd choose LoRA rank, alpha, and target modules for a new fine-tuning project. | LoRA-hyperparams · rank · alpha · target-modules | 📝 |
| Q17‑23 | You're asked to host 100 LoRA adapters on the same base model. Walk through your serving architecture. | 100-adapters · serving · S-LoRA · architecture | 📝 |
| Q17‑24 | Your customer reports that fine-tuning broke a previously-working capability. Walk through your catastrophic-forgetting investigation and fix. | catastrophic-forgetting · fine-tuning · capability · regression | 📝 |
| Q17‑25 | Walk through how you'd combine LoRA with quantization in a way that's both memory-efficient and quality-preserving. | LoRA · quantization · QLoRA · memory-quality | 📝 |
| Q17‑26 | Walk through how you'd merge 5 task-specific LoRA adapters into a single multi-task adapter with minimal regression. | adapter-merging · 5-tasks · TIES · regression | 📝 |
| Q17‑27 | Your team has 10K labeled examples and a 70B base model. Walk through how you'd decide between LoRA, full FT, and rejection-sampling fine-tuning. | 10K-examples · 70B · LoRA-vs-full · RSF | 📝 |

---

## Why these questions matter

PEFT is the dominant paradigm for LLM adaptation. These questions test whether a candidate understands the **theoretical foundations, practical tradeoffs, and production serving** of fine-tuned models.

| Theme | Key questions |
|-------|--------------|
| **LoRA fundamentals** | Q17‑02, Q17‑03, Q17‑04, Q17‑13 — mechanics, variants, theory |
| **PEFT variants** | Q17‑06, Q17‑07, Q17‑11 — prefix tuning, IA³, initialization |
| **Merging & serving** | Q17‑09, Q17‑10, Q17‑16, Q17‑23 — TIES, S-LoRA, 100 adapters |
| **Limitations** | Q17‑05, Q17‑14, Q17‑18, Q17‑19 — forgetting, reasoning gap, future |

---

## Reading order suggestions

**New to PEFT?** Q17‑01 → Q17‑02 → Q17‑03 → Q17‑05 → Q17‑12

**Interview in 48 hours?** Q17‑02 (LoRA) · Q17‑04 (QLoRA/DoRA) · Q17‑13 (theory) · Q17‑09 (merging) · Q17‑14 (reasoning gap)

**Full track:** Q17‑01 → Q17‑05 → Q17‑06 → Q17‑12 → Q17‑13 → Q17‑16 → Q17‑19

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 16 — Distributed Training & Optimization](../section-16-distributed-training-optimization/README.md) &nbsp;|&nbsp; [Section 18 — Speech, Audio & Multimodal ➡️](../section-18-speech-audio-multimodal/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s17qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
