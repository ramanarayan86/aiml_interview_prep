<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 18 — Speech, Audio & Multimodal](../section-18-speech-audio-multimodal/README.md) &nbsp;|&nbsp; [Section 20 — Data Engineering & Curation ➡️](../section-20-data-engineering-curation/README.md)

</div>

---

# Section 19 · Prompt Engineering, ICL, and Steering

<div align="center">

![Questions](https://img.shields.io/badge/questions-23-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F23-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-23%2F23-6c4fe0)

</div>

Prompting is both a practitioner skill and an active research area. This section covers **in-context learning theory**, **chain-of-thought variants**, **automated prompt optimization** (APE, OPRO, DSPy), **steering vectors** and **activation patching**, and the hard question of where prompting ends and fine-tuning should begin.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q19‑01 | What is in-context learning (ICL)? How does it differ from fine-tuning? | ICL · few-shot · fine-tuning · no-gradient | 📝 |
| Q19‑02 | Explain the difference between zero-shot, one-shot, and few-shot prompting. When does few-shot win? | zero-shot · one-shot · few-shot · prompting | 📝 |
| Q19‑03 | What is the difference between a system prompt and a user prompt? How does instruction-following training affect this? | system-prompt · user-prompt · instruction-following | 📝 |
| Q19‑04 | Why are LLMs sensitive to prompt phrasing? What are the main failure modes? | prompt-sensitivity · brittleness · surface-form · failures | 📝 |
| Q19‑05 | Compare CoT, self-consistency, and Tree-of-Thought prompting. When does each help most? | CoT · self-consistency · ToT · reasoning · prompting | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q19‑06 | What mechanisms explain ICL? Discuss implicit gradient descent, Bayesian inference, and task retrieval hypotheses. | ICL-mechanisms · implicit-GD · Bayesian · task-retrieval | 📝 |
| Q19‑07 | Walk through APE, OPRO, APO, and DSPy. What do automated prompt optimizers gain over human engineering? | APE · OPRO · APO · DSPy · automated-prompting | 📝 |
| Q19‑08 | What is the difference between hard prompts (discrete tokens) and soft prompts (continuous embeddings)? What does each buy? | hard-prompt · soft-prompt · continuous · discrete | 📝 |
| Q19‑09 | Explain steering vectors. How are they constructed, and how do they differ from prompt-based control? | steering-vectors · representation · control · activation | 📝 |
| Q19‑10 | What is activation patching / causal tracing? What has it revealed about factual recall and in-context learning? | activation-patching · causal-tracing · factual-recall · ICL | 📝 |
| Q19‑11 | Discuss the role of demonstrations in ICL. What matters: the labels, the format, the distribution, or all three? | demonstrations · labels · format · distribution · ICL | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q19‑12 | Critically evaluate ICL vs fine-tuning as adaptation strategies. What does recent empirical evidence say at scale? | ICL-vs-FT · scale · adaptation · empirical | 📝 |
| Q19‑13 | Discuss the two leading theories of ICL — Bayesian inference over implicit concepts and mesa-optimization / gradient descent in-context. Which best fits the empirical evidence? | ICL-theory · Bayesian · mesa-optimization · gradient-descent | 📝 |
| Q19‑14 | Walk through the design of a self-improving prompt optimization loop (OPRO/APO-style). What convergence guarantees exist? | self-improving · OPRO · APO · convergence · loop | 📝 |
| Q19‑15 | What are the fundamental limits of prompting? When does any amount of prompting fail? | prompting-limits · capability · reasoning · failure-modes | 📝 |
| Q19‑16 | Discuss steering-based safety. Can you use activation steering to suppress harmful behavior reliably? What are the failure modes? | steering-safety · activation · harmful-behavior · failure | 📝 |
| Q19‑17 | Walk through a full automated prompt optimization pipeline using OPRO/APO. What metrics do you optimize, and how do you avoid overfitting to the dev set? | automated-prompt · OPRO · APO · metrics · overfitting | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q19‑18 | Your product requires consistent JSON output from a model that sometimes fails to comply. Walk through your prompting and post-processing strategy. | JSON-output · structured-output · compliance · prompting | 📝 |
| Q19‑19 | Walk through how you'd choose between prompt engineering and fine-tuning for a new capability. What's your decision framework? | prompt-vs-finetune · decision-framework · capability | 📝 |
| Q19‑20 | Your CoT reasoning is occasionally wrong but confident. Walk through your investigation and what you'd change. | CoT · wrong-reasoning · confidence · debugging | 📝 |
| Q19‑21 | You want to use ICL to teach a model a new task with 20 demonstrations. Walk through how you'd select, format, and validate them. | ICL · 20-demonstrations · selection · format · validation | 📝 |
| Q19‑22 | Walk through how you'd systematically optimize a multi-turn conversational prompt using DSPy. | DSPy · multi-turn · optimization · systematic | 📝 |
| Q19‑23 | Your steering vector intervention reduces harmful outputs but also degrades general helpfulness. Walk through how you'd investigate the tradeoff and tune it. | steering-vectors · tradeoff · helpfulness · safety | 📝 |

---

## Why these questions matter

Prompting research sits at the intersection of interpretability, systems design, and practical ML engineering. These questions probe whether a candidate understands the **theory of ICL, the limits of prompting, and how to automate and scale prompt optimization**.

| Theme | Key questions |
|-------|--------------|
| **ICL theory** | Q19‑06, Q19‑11, Q19‑13 — mechanisms, demonstrations, Bayesian vs GD |
| **Automated optimization** | Q19‑07, Q19‑14, Q19‑17 — APE, OPRO, DSPy |
| **Steering & interpretability** | Q19‑09, Q19‑10, Q19‑16 — steering vectors, activation patching |
| **Limits & decisions** | Q19‑12, Q19‑15, Q19‑19 — ICL vs FT, fundamental limits |

---

## Reading order suggestions

**New to prompting?** Q19‑01 → Q19‑02 → Q19‑04 → Q19‑05 → Q19‑11

**Interview in 48 hours?** Q19‑05 (CoT/ToT) · Q19‑06 (ICL mechanisms) · Q19‑09 (steering) · Q19‑12 (ICL vs FT) · Q19‑15 (limits)

**Full track:** Q19‑01 → Q19‑05 → Q19‑06 → Q19‑09 → Q19‑12 → Q19‑13 → Q19‑15

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 18 — Speech, Audio & Multimodal](../section-18-speech-audio-multimodal/README.md) &nbsp;|&nbsp; [Section 20 — Data Engineering & Curation ➡️](../section-20-data-engineering-curation/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s19qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
