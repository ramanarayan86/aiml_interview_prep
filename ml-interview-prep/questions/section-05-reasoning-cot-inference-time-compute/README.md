<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 4 — Post-training: SFT, RLHF, DPO](../section-04-post-training-sft-rlhf-dpo/README.md) &nbsp;|&nbsp; [Section 6 — Efficient Inference & Serving ➡️](../section-06-efficient-inference-and-serving/README.md)

</div>

---

# Section 5 · Reasoning, Chain-of-Thought, and Inference-Time Compute

<div align="center">

![Questions](https://img.shields.io/badge/questions-34-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-6%2F34-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-34%2F34-6c4fe0)

</div>

Reasoning and inference-time compute have become the defining frontier of modern LLMs. This section covers **chain-of-thought prompting** and its variants, **test-time compute scaling** (o1/o3/R1 paradigm), **process supervision**, **search-based reasoning** (MCTS, verifiers), and the deep question of whether models are truly reasoning or pattern-matching.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| [Q5‑01 · Chain-of-thought prompting fundamentals](./q01-cot-prompting.md) | What is chain-of-thought prompting? Why does it work? | CoT · prompting · step-by-step · reasoning | ✅ |
| [Q5‑02 · CoT variants](./q02-cot-variants.md) | Differentiate few-shot CoT, zero-shot CoT, and self-consistency. | few-shot · zero-shot · self-consistency · CoT | ✅ |
| [Q5‑03 · Zero-shot CoT mechanism](./q03-zero-shot-cot-mechanism.md) | What is 'let's think step by step' doing mechanically? | zero-shot-CoT · elicitation · reasoning-trace | ✅ |
| [Q5‑04 · Self-consistency decoding](./q04-self-consistency.md) | Explain self-consistency decoding. When does it help vs hurt? | self-consistency · majority-voting · sampling · reasoning | ✅ |
| [Q5‑05 · Tree of thoughts](./q05-tree-of-thoughts.md) | What is tree-of-thoughts? How does it differ from chain-of-thought? | tree-of-thoughts · search · branching · CoT | ✅ |
| [Q5‑06 · System 1 vs system 2 reasoning](./q06-system1-system2.md) | What is the difference between system 1 and system 2 reasoning in the LLM context? | system-1 · system-2 · fast-slow · reasoning | ✅ |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q5‑07 | Discuss the test-time compute scaling laws from the recent DeepMind paper. When is it better to scale inference vs pretraining? | test-time-compute · scaling · inference-vs-training · DeepMind | 📝 |
| Q5‑08 | Walk through how a verifier-guided search (best-of-N, beam search with PRM) improves reasoning. Where does it break? | PRM · verifier · best-of-N · beam-search | 📝 |
| Q5‑09 | Explain Monte Carlo Tree Search applied to LLM reasoning. How are rollouts, value estimates, and selection adapted? | MCTS · rollouts · value-estimation · reasoning | 📝 |
| Q5‑10 | What is process supervision and why is it more sample-efficient than outcome supervision for reasoning tasks? | process-supervision · PRM · ORM · sample-efficiency | 📝 |
| Q5‑11 | Discuss the o1/o3/R1 paradigm. What's novel about training models to 'think' via long CoT with RL on verifiable rewards? | o1 · o3 · R1 · long-CoT · verifiable-rewards | 📝 |
| Q5‑12 | Compare Self-Refine, Reflexion, and Self-Correct. When does self-correction actually improve answers vs hurt them? | Self-Refine · Reflexion · self-correction · iterative | 📝 |
| Q5‑13 | Discuss the 'reasoning gap' between base models and reasoning-tuned models. What is gained, and what (if anything) is lost? | reasoning-gap · base-model · tuned-model · capability | 📝 |
| Q5‑14 | What is faithfulness in chain-of-thought? Reference the work showing that CoT explanations often don't reflect the model's actual computation. | faithfulness · CoT · post-hoc · explanation | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q5‑15 | DeepSeek-R1 used pure RL (GRPO) without SFT cold start and observed emergent reasoning. Discuss GRPO vs PPO — why does GRPO not need a value network? What are the stability tradeoffs? | GRPO · DeepSeek-R1 · emergent-reasoning · value-free | 📝 |
| Q5‑16 | How would you design a reward signal for non-verifiable domains (creative writing, open-ended dialogue)? Discuss generative reward models and rubric-based scoring. | non-verifiable · reward-design · generative-RM · rubric | 📝 |
| Q5‑17 | Critically evaluate whether long CoT reasoning is 'real reasoning' or sophisticated pattern matching. Reference Apple's GSM-Symbolic paper, the ARC challenge results, and François Chollet's critique. | real-reasoning · pattern-matching · GSM-Symbolic · ARC | 📝 |
| Q5‑18 | Design an inference-time compute allocation strategy: given a budget of K total forward passes, how do you split between sampling, verification, and revision? | compute-allocation · forward-passes · sampling · verification | 📝 |
| Q5‑19 | Discuss latent reasoning approaches (Coconut, Quiet-STaR) — reasoning in continuous space rather than tokens. What are the theoretical and practical advantages and limitations? | latent-reasoning · Coconut · Quiet-STaR · continuous-space | 📝 |
| Q5‑20 | What is the role of length in reasoning models? Why do o1-style models produce very long chains, and what does this say about test-time compute scaling? | chain-length · o1 · test-time-scaling · verbosity | 📝 |
| Q5‑21 | Discuss the difficulty of training reasoning models on tasks without verifiable rewards. How do you bootstrap from verifiable domains to non-verifiable ones? | non-verifiable · bootstrap · reward-generalization | 📝 |
| Q5‑22 | Walk through the design of a curriculum for training a reasoning model. Should easy problems come first, or hard ones? What does the recent literature suggest? | curriculum · difficulty · ordering · reasoning-training | 📝 |
| Q5‑23 | Discuss whether reasoning capability transfers across domains. If a model is trained on math reasoning, does it improve at code reasoning? What's the evidence? | transfer · cross-domain · math · code-reasoning | 📝 |
| Q5‑24 | Critique the use of GSM8K, MATH, and AIME as primary reasoning benchmarks. What's missing, and what would a better benchmark look like? | GSM8K · MATH · AIME · benchmark-critique | 📝 |
| Q5‑25 | Discuss inference-time speculative reasoning — methods that explore multiple reasoning paths in parallel and prune. How does this relate to classical search? | speculative-reasoning · parallel-paths · pruning · classical-search | 📝 |
| Q5‑26 | Walk through how you would distill an o1-style reasoning model into a smaller model. What's preserved, what's lost? | distillation · o1 · reasoning · knowledge-transfer | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q5‑27 | Your reasoning model wins on AIME but loses on a custom enterprise reasoning benchmark. Walk through how you'd investigate the gap. | benchmark-gap · reasoning · enterprise · investigation | 📝 |
| Q5‑28 | You need to ship a reasoning model with sub-2-second average latency. Walk through your design — model size, max reasoning length, early-exit strategy, verifier integration. | latency · reasoning · early-exit · production | 📝 |
| Q5‑29 | Best-of-N sampling improves accuracy from 60% to 78% at N=16. Your boss asks if it's worth it for production. How do you frame the tradeoff? | best-of-N · production · cost-accuracy · tradeoff | 📝 |
| Q5‑30 | Walk through how you'd build a verifier model for a domain where you can't easily generate ground-truth labels (e.g., legal reasoning). | verifier · weak-supervision · legal · label-free | 📝 |
| Q5‑31 | Your team wants to train an o1-style reasoning model. You have 2 months and 512 H100s. Walk through your plan and the riskiest decisions. | o1-training · resource-planning · RL · risk | 📝 |
| Q5‑32 | A reasoning model you trained shows excellent CoT on math but fabricates reasoning steps on factual questions. Walk through how you diagnose and fix. | CoT-fabrication · factuality · faithfulness · debugging | 📝 |
| Q5‑33 | You're given an inference budget of 32 forward passes per query. Walk through how you'd allocate them — single chain, parallel chains, verification, revision. | inference-budget · allocation · parallel · verification | 📝 |
| Q5‑34 | Walk through how you'd detect and prevent 'reasoning shortcuts' where the model arrives at correct answers via incorrect reasoning chains. | reasoning-shortcuts · spurious-CoT · detection · prevention | 📝 |

---

## Why these questions matter

Reasoning is the most-interviewed topic at frontier labs in 2024–2025. These questions test whether a candidate understands not just how CoT works but the **entire reasoning training pipeline** — from reward design to test-time compute allocation to honest evaluation.

| Theme | Key questions |
|-------|--------------|
| **CoT fundamentals** | Q5‑01, Q5‑02, Q5‑04, Q5‑14 — mechanics, variants, faithfulness |
| **Inference-time compute** | Q5‑07, Q5‑11, Q5‑18, Q5‑20 — scaling laws, o1 paradigm |
| **Training reasoning** | Q5‑10, Q5‑15, Q5‑22, Q5‑26 — PRMs, GRPO, curriculum, distillation |
| **Evaluation & limits** | Q5‑17, Q5‑24, Q5‑32, Q5‑34 — real reasoning vs pattern matching |

---

## Reading order suggestions

**New to reasoning?** Q5‑01 → Q5‑02 → Q5‑04 → Q5‑05 → Q5‑11

**Interview in 48 hours?** Q5‑11 (o1/R1) · Q5‑10 (PRMs) · Q5‑15 (GRPO) · Q5‑17 (real reasoning?) · Q5‑07 (scaling)

**Full track:** Q5‑01 → Q5‑14 → Q5‑08 → Q5‑10 → Q5‑11 → Q5‑15 → Q5‑17

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 4 — Post-training: SFT, RLHF, DPO](../section-04-post-training-sft-rlhf-dpo/README.md) &nbsp;|&nbsp; [Section 6 — Efficient Inference & Serving ➡️](../section-06-efficient-inference-and-serving/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s5qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
