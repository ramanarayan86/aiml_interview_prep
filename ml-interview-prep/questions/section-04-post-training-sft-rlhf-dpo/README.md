<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 3 — Pretraining & Scaling Laws](../section-03-pretraining-and-scaling-laws/README.md) &nbsp;|&nbsp; [Section 5 — Reasoning, CoT & Inference-Time Compute ➡️](../section-05-reasoning-cot-inference-time-compute/README.md)

</div>

---

# Section 4 · Post-training: SFT, RLHF, DPO and Beyond

<div align="center">

![Questions](https://img.shields.io/badge/questions-40-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-7%2F40-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-33%2F40-6c4fe0)

</div>

Post-training is where raw capability becomes useful, safe, and aligned behavior. This section covers the full pipeline: **supervised fine-tuning** (LIMA hypothesis, data quality), **RLHF** (reward models, PPO, KL penalty), **direct preference optimization** (DPO and its variants), and the frontier of **online RL, process rewards, and constitutional AI**.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q4‑01 | [What is supervised fine-tuning (SFT)? Why is data quality often more important than quantity here (LIMA hypothesis)?](./q01-sft-lima.md) | SFT · LIMA · instruction-tuning · data-quality | ✅ |
| Q4‑02 | [Explain RLHF at a high level. What are the three stages (SFT, RM, PPO)?](./q02-rlhf-overview.md) | RLHF · reward-model · PPO · three-stage | ✅ |
| Q4‑03 | [What is a reward model? How is it trained from preference pairs (Bradley-Terry model)?](./q03-reward-model.md) | reward-model · Bradley-Terry · preference-pairs | ✅ |
| Q4‑04 | [What is the KL penalty in RLHF for? What happens if you remove it?](./q04-kl-penalty.md) | KL-penalty · RLHF · policy-constraint · reward-hacking | ✅ |
| Q4‑05 | [What's the difference between on-policy and off-policy RL? Which category does PPO fall into?](./q05-on-off-policy.md) | on-policy · off-policy · PPO · RL | ✅ |
| Q4‑06 | [What is instruction tuning, and how does it differ from chat tuning?](./q06-instruction-tuning.md) | instruction-tuning · chat-tuning · SFT | ✅ |
| Q4‑07 | [Explain the role of a system prompt vs a user prompt during post-training.](./q07-system-user-prompt.md) | system-prompt · user-prompt · post-training | ✅ |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q4‑08 | Derive the PPO objective. What does the clipping ratio prevent? | PPO · clipping · policy-gradient · surrogate-objective | 📝 |
| Q4‑09 | What is DPO (Direct Preference Optimization)? Derive its loss from the RLHF objective — why does it eliminate the reward model? | DPO · preference-optimization · reward-free · derivation | 📝 |
| Q4‑10 | Compare DPO, IPO, KTO, ORPO, and SimPO. What problems does each address with vanilla DPO? | DPO · IPO · KTO · ORPO · SimPO · comparison | 📝 |
| Q4‑11 | What is reward hacking? Give concrete examples (length bias, sycophancy, format gaming) and mitigation strategies. | reward-hacking · Goodhart · length-bias · sycophancy | 📝 |
| Q4‑12 | Explain RLAIF and Constitutional AI. When does AI feedback substitute reliably for human feedback? | RLAIF · Constitutional-AI · AI-feedback | 📝 |
| Q4‑13 | Discuss the role of the reference model in DPO/RLHF. What happens if it drifts too far from the policy? | reference-model · DPO · policy-drift · KL | 📝 |
| Q4‑14 | What is rejection sampling fine-tuning (RFT / RAFT)? When is it preferable to RLHF? | rejection-sampling · RFT · RAFT · sampling-efficiency | 📝 |
| Q4‑15 | Compare offline preference learning (DPO) vs online preference learning (iterative DPO, online RLHF). What does the recent literature suggest about the gap? | offline · online · iterative-DPO · preference-learning | 📝 |
| Q4‑16 | What is the advantage in PPO and how is it estimated (GAE)? Why does it matter for stability? | GAE · advantage-estimation · PPO · stability | 📝 |
| Q4‑17 | Discuss best-of-N sampling as a baseline for RLHF. When does RLHF actually outperform BoN at matched inference compute? | best-of-N · BoN · inference-compute · RLHF-baseline | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q4‑18 | DPO has known failure modes — the chosen and rejected log-probs often both decrease. Why does this happen, and how do methods like DPO-Positive, RPO, or length-normalized DPO address it? | DPO · log-prob-collapse · DPO-Positive · RPO | 📝 |
| Q4‑19 | Derive the connection between RLHF and inverse reinforcement learning. What does the optimal policy look like in closed form (the soft-Q form)? | IRL · soft-Q · RLHF · optimal-policy | 📝 |
| Q4‑20 | Discuss process reward models (PRMs) vs outcome reward models (ORMs) for reasoning tasks. Reference Let's Verify Step by Step, Math-Shepherd, and OpenAI's o1 approach. | PRM · ORM · process-supervision · reasoning | 📝 |
| Q4‑21 | Design a post-training pipeline for a 70B model targeting math reasoning. How would you combine SFT, rejection sampling (STaR/V-STaR), DPO, and online RL? | post-training · math-reasoning · STaR · pipeline-design | 📝 |
| Q4‑22 | Critically evaluate the 'superficial alignment hypothesis' (LIMA, URIAL). Does post-training inject new knowledge or only elicit existing capabilities? | superficial-alignment · LIMA · URIAL · knowledge-elicitation | 📝 |
| Q4‑23 | Discuss GRPO (Group Relative Policy Optimization) used in DeepSeek-R1. How does it eliminate the value function and what are the stability tradeoffs? | GRPO · DeepSeek-R1 · value-free · policy-optimization | 📝 |
| Q4‑24 | What are the theoretical limits of RLHF? Discuss the impossibility results around social choice and preference aggregation. | RLHF-limits · social-choice · Arrow's-theorem · preference-aggregation | 📝 |
| Q4‑25 | Design a reward model that is robust to length bias, position bias, and sycophancy. What architectural and data choices help? | reward-model · robustness · length-bias · position-bias | 📝 |
| Q4‑26 | Discuss multi-objective RLHF — how do you train a model that is simultaneously helpful, harmless, and honest? Reference the constitutional approach and the Pareto-front view. | multi-objective · RLHF · Pareto · HHH | 📝 |
| Q4‑27 | Walk through Anthropic's Constitutional AI in detail. What is the SL-CAI vs RL-CAI distinction, and why does the critique-revise loop work? | Constitutional-AI · SL-CAI · RL-CAI · critique-revise | 📝 |
| Q4‑28 | Discuss model collapse from self-training on AI-generated data. What does 'The Curse of Recursion' suggest about RLAIF at scale? | model-collapse · self-training · RLAIF · recursion | 📝 |
| Q4‑29 | Critique the assumption that human preferences are transitive, consistent, and well-defined. What alternative frameworks (e.g., regret-based, constraint-based) have been proposed? | preference-theory · transitivity · regret-based · constraint-based | 📝 |
| Q4‑30 | Discuss the 'alignment tax' — does RLHF degrade base capabilities? What does the evidence say across model families? | alignment-tax · capability-regression · RLHF · evidence | 📝 |
| Q4‑31 | Walk through how you would design an evaluation suite to detect overoptimization in RLHF. Reference Gao, Schulman, Hilton on goodharting. | overoptimization · Goodhart · evaluation · goodharting | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q4‑32 | Your DPO run improves win-rate on Arena but tanks on MMLU and GSM8K. Walk through your diagnostic and mitigation plan. | DPO · capability-regression · debugging · win-rate | 📝 |
| Q4‑33 | You're given 50K preference pairs and 200K SFT examples. Walk through how you'd order and combine them in a post-training pipeline. | pipeline · preference-data · SFT · data-ordering | 📝 |
| Q4‑34 | Your RLHF model has started giving overly long, sycophantic answers. What's your debugging and fix plan? | sycophancy · length-bias · RLHF · debugging | 📝 |
| Q4‑35 | A new RM you trained scores well on a held-out preference set but PPO with it produces worse outputs than the previous RM. What's going on, and how do you investigate? | reward-hacking · overoptimization · RM · PPO | 📝 |
| Q4‑36 | Your team has 100K human annotations budget for the next quarter. How do you allocate them across SFT, preference data, red-teaming, and eval? | annotation-budget · SFT · preference-data · red-teaming | 📝 |
| Q4‑37 | You discover that 8% of your preference pairs come from annotators who disagree with the majority. Do you filter them, weight them, or keep them? Defend. | annotator-disagreement · data-quality · filtering · weighting | 📝 |
| Q4‑38 | Walk through how you'd run an online RLHF loop in production, including data collection, RM retraining cadence, policy update frequency, and rollback strategy. | online-RLHF · production · rollback · retraining-cadence | 📝 |
| Q4‑39 | Your DPO checkpoint regresses on safety after a single epoch. Walk through how to balance helpfulness gains with safety preservation in the recipe. | DPO · safety-regression · helpfulness · balance | 📝 |
| Q4‑40 | You're asked to migrate a production model from PPO-RLHF to DPO. What experiments would you run to validate parity, and what risks would you flag to leadership? | PPO-to-DPO · migration · validation · risk-assessment | 📝 |

---

## Why these questions matter

Post-training is where all frontier labs differentiate. These questions probe whether a candidate understands the **full alignment pipeline** — not just the mechanics of PPO or DPO, but the data strategy, reward modeling pitfalls, failure modes, and production considerations.

| Theme | Key questions |
|-------|--------------|
| **RL foundations** | Q4‑02, Q4‑05, Q4‑08, Q4‑16 — PPO, GAE, on/off-policy |
| **DPO & variants** | Q4‑09, Q4‑10, Q4‑15, Q4‑18 — derivation, failure modes, alternatives |
| **Reward modeling** | Q4‑03, Q4‑11, Q4‑25, Q4‑31 — training, hacking, robustness |
| **Advanced alignment** | Q4‑20, Q4‑22, Q4‑23, Q4‑27 — PRMs, GRPO, Constitutional AI |

---

## Reading order suggestions

**New to post-training?** Q4‑01 → Q4‑02 → Q4‑03 → Q4‑04 → Q4‑09

**Interview in 48 hours?** Q4‑09 (DPO) · Q4‑02 (RLHF overview) · Q4‑10 (DPO variants) · Q4‑11 (reward hacking) · Q4‑20 (PRMs)

**Full track:** Q4‑01 → Q4‑07 → Q4‑08 → Q4‑17 → Q4‑18 → Q4‑31

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 3 — Pretraining & Scaling Laws](../section-03-pretraining-and-scaling-laws/README.md) &nbsp;|&nbsp; [Section 5 — Reasoning, CoT & Inference-Time Compute ➡️](../section-05-reasoning-cot-inference-time-compute/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s4qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
