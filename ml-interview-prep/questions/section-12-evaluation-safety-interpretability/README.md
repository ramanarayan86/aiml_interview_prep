<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 11 — Agents, Tool Use & Code Generation](../section-11-agents-tool-use-code-generation/README.md) &nbsp;|&nbsp; [Section 13 — Diffusion, DiT & Generative Modeling ➡️](../section-13-diffusion-dit-generative-modeling/README.md)

</div>

---

# Section 12 · Evaluation, Safety, and Interpretability

<div align="center">

![Questions](https://img.shields.io/badge/questions-39-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F39-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-39%2F39-6c4fe0)

</div>

Evaluation, safety, and interpretability are the disciplines that separate responsible frontier AI from raw capability. This section covers **benchmark design and contamination**, **LLM-as-judge**, **red-teaming** (manual, automated, GCG), **mechanistic interpretability** (circuits, SAEs), **scalable oversight**, and the alignment research agenda.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q12‑01 | What's wrong with using accuracy on benchmarks like MMLU as a primary metric for frontier models? | MMLU · benchmark · accuracy · contamination | 📝 |
| Q12‑02 | Discuss contamination. How do you detect and mitigate test-set leakage in pretraining data? | contamination · leakage · detection · deduplication | 📝 |
| Q12‑03 | What is LLM-as-judge? When is it reliable? | LLM-judge · reliability · bias · calibration | 📝 |
| Q12‑04 | Explain hallucination. Distinguish factuality hallucinations, faithfulness hallucinations, and counterfactual confabulation. | hallucination · factuality · faithfulness · confabulation | 📝 |
| Q12‑05 | What is jailbreaking? Give three categorically different attack vectors. | jailbreaking · attack-vectors · safety · adversarial | 📝 |
| Q12‑06 | What is the difference between intrinsic and extrinsic evaluation of LLMs? | intrinsic · extrinsic · perplexity · downstream-task | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q12‑07 | Compare static benchmarks (MMLU, GPQA) vs dynamic ones (Chatbot Arena, LiveBench). What does each measure and miss? | static · dynamic · Arena · LiveBench | 📝 |
| Q12‑08 | Discuss the calibration of LLMs. Are modern post-trained models well-calibrated? Reference Kadavath et al. and the impact of RLHF on calibration. | calibration · Kadavath · RLHF · overconfidence | 📝 |
| Q12‑09 | Walk through Constitutional AI and RLAIF. When is AI feedback a reasonable substitute for human feedback? | Constitutional-AI · RLAIF · AI-feedback · reliability | 📝 |
| Q12‑10 | Discuss red-teaming methodologies — manual, automated (PAIR, GCG), and model-vs-model. What are the limits of each? | red-teaming · PAIR · GCG · model-vs-model | 📝 |
| Q12‑11 | What is a sparse autoencoder (SAE)? How does Anthropic / DeepMind use them for feature discovery? | SAE · sparse-autoencoder · features · interpretability | 📝 |
| Q12‑12 | Explain the Elo and Bradley-Terry models behind Chatbot Arena. What biases does the format introduce? | Elo · Bradley-Terry · Arena · biases | 📝 |
| Q12‑13 | What is the difference between a benchmark and a leaderboard? Why does Goodhart's law hit leaderboards hardest? | benchmark · leaderboard · Goodhart · overfitting | 📝 |
| Q12‑14 | Discuss prompt sensitivity in evaluations. How do you measure whether a model 'really knows' something vs got lucky with a prompt? | prompt-sensitivity · robustness · evaluation · paraphrase | 📝 |
| Q12‑15 | Walk through the differences between adversarial robustness (GCG, AutoDAN) and natural robustness (paraphrase invariance, format invariance). | adversarial · natural-robustness · GCG · paraphrase | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q12‑16 | Critique LLM-as-judge — discuss position bias, verbosity bias, self-preference bias, and the recent findings on judge-model consistency under perturbation. | LLM-judge · position-bias · verbosity · self-preference | 📝 |
| Q12‑17 | Discuss mechanistic interpretability — circuits, induction heads, and the progress from 'Toy Models of Superposition' to 'Scaling Monosemanticity.' What does it actually buy us for safety? | circuits · induction-heads · superposition · monosemanticity | 📝 |
| Q12‑18 | Walk through how you'd evaluate a frontier model for dangerous capability uplift (bio, cyber, autonomy). What's the evaluation methodology and its known limitations? | dangerous-capability · bio · cyber · uplift-evaluation | 📝 |
| Q12‑19 | Design an evaluation suite for a senior-research-grade VLM judge that calibrates against human preferences without overfitting to the judge's training distribution. | VLM-judge · eval-suite · distribution-shift · calibration | 📝 |
| Q12‑20 | Discuss the alignment-capability tradeoff. Does RLHF make models dumber (the 'alignment tax')? Reference findings on capabilities regression. | alignment-tax · RLHF · capability-regression · evidence | 📝 |
| Q12‑21 | Critically evaluate the current AI safety research agenda. Are interpretability, scalable oversight, and adversarial robustness the right pillars? | safety-agenda · interpretability · scalable-oversight · robustness | 📝 |
| Q12‑22 | Discuss scalable oversight — debate, market-making, and weak-to-strong generalization. What are the theoretical and empirical results? | scalable-oversight · debate · weak-to-strong · oversight | 📝 |
| Q12‑23 | Walk through deceptive alignment as a threat model. What evidence (or lack thereof) do we have, and what experiments would update you? | deceptive-alignment · threat-model · evidence · experiments | 📝 |
| Q12‑24 | Discuss the 'sleeper agents' result (Hubinger et al.). What does it imply about the durability of alignment training? | sleeper-agents · Hubinger · alignment-training · durability | 📝 |
| Q12‑25 | Critique probing-based interpretability. What does linear probing reveal vs miss? Compare to causal interventions (activation patching). | probing · linear-probe · activation-patching · causal | 📝 |
| Q12‑26 | Walk through the design of an automated red-team that generates novel jailbreaks via RL. What's the action space and reward signal? | automated-red-team · RL · jailbreaks · action-space | 📝 |
| Q12‑27 | Discuss the role of refusal training. How do you prevent over-refusal while maintaining harm avoidance? Reference XSTest and refusal-direction research. | refusal-training · over-refusal · XSTest · refusal-direction | 📝 |
| Q12‑28 | Critically evaluate frontier-model safety evaluations (Anthropic RSP, OpenAI Preparedness Framework, DeepMind Frontier Safety). | RSP · Preparedness-Framework · Frontier-Safety · comparison | 📝 |
| Q12‑29 | Discuss the connection between mechanistic interpretability and steering. How do steering vectors (representation engineering) relate to circuits? | steering-vectors · representation-engineering · circuits · RepE | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q12‑30 | Your model passes all internal safety evals but is jailbroken on launch day by a community Reddit post. Walk through your incident response and root-cause investigation. | jailbreak · incident-response · safety-eval · root-cause | 📝 |
| Q12‑31 | Walk through how you'd build a contamination-detection pipeline that runs continuously against new benchmarks. | contamination · detection · pipeline · continuous | 📝 |
| Q12‑32 | Your LLM-as-judge correlation with humans is 0.78. The product team wants to deploy it as a gating signal. Walk through what you'd require before approving. | LLM-judge · 0.78-correlation · gating · approval | 📝 |
| Q12‑33 | Walk through how you'd design red-teaming for a frontier model under preparedness-framework constraints. | red-teaming · preparedness · framework · design | 📝 |
| Q12‑34 | A safety researcher claims your model exhibits 'deceptive alignment' based on a probe study. Walk through how you'd evaluate the claim. | deceptive-alignment · probe · evaluation · claim | 📝 |
| Q12‑35 | Your model's MMLU score is 0.5 points higher than a competitor's, but it loses in Arena. Walk through how you'd frame the discrepancy to leadership. | MMLU · Arena · discrepancy · communication | 📝 |
| Q12‑36 | Walk through how you'd build an evaluation that distinguishes capability gains from spurious benchmark gains. | capability · spurious · benchmark · evaluation | 📝 |
| Q12‑37 | You're asked to set up an internal model-risk-management process. Walk through the structure, gating, and accountability you'd propose. | model-risk · governance · gating · accountability | 📝 |
| Q12‑38 | Walk through how you'd design an experiment to test whether RLHF causes the alignment tax in your model family. | alignment-tax · experiment · RLHF · capability | 📝 |
| Q12‑39 | Your model is being deployed in healthcare. Walk through the additional evaluations and safeguards you'd require beyond standard pretraining + RLHF. | healthcare · deployment · safeguards · evaluation | 📝 |

---

## Why these questions matter

Safety and evaluation are increasingly weighted at frontier labs. These questions test whether a candidate has **principled views on what evaluation means**, understands the alignment research landscape, and can design robust systems.

| Theme | Key questions |
|-------|--------------|
| **Evaluation design** | Q12‑01, Q12‑07, Q12‑13, Q12‑14 — benchmarks, leaderboards, sensitivity |
| **Safety & alignment** | Q12‑21, Q12‑22, Q12‑23, Q12‑24 — agenda, scalable oversight, sleeper agents |
| **Interpretability** | Q12‑11, Q12‑17, Q12‑25, Q12‑29 — SAEs, circuits, probing, steering |
| **Red-teaming** | Q12‑10, Q12‑26, Q12‑30, Q12‑33 — GCG, RL red-teams, incident response |

---

## Reading order suggestions

**New to the area?** Q12‑01 → Q12‑03 → Q12‑04 → Q12‑05 → Q12‑06

**Interview in 48 hours?** Q12‑17 (mechanistic interp) · Q12‑22 (scalable oversight) · Q12‑16 (LLM-judge critique) · Q12‑20 (alignment tax) · Q12‑24 (sleeper agents)

**Full track:** Q12‑01 → Q12‑06 → Q12‑07 → Q12‑15 → Q12‑16 → Q12‑17 → Q12‑21

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 11 — Agents, Tool Use & Code Generation](../section-11-agents-tool-use-code-generation/README.md) &nbsp;|&nbsp; [Section 13 — Diffusion, DiT & Generative Modeling ➡️](../section-13-diffusion-dit-generative-modeling/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s12qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
