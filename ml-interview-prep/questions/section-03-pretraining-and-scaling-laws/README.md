<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 2 — Tokenization & Embeddings](../section-02-tokenization-and-embeddings/README.md) &nbsp;|&nbsp; [Section 4 — Post-training: SFT, RLHF, DPO ➡️](../../README.md#section-4--post-training-sft-rlhf-dpo-and-beyond)

</div>

---

# Section 3 · Pretraining and Scaling Laws

<div align="center">

![Questions](https://img.shields.io/badge/questions-32-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-16%2F32-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-16%2F32-6c4fe0)

</div>

Pretraining is where the intelligence lives. Everything downstream — fine-tuning, RLHF, prompting — is steering a capability that was crystallized during pretraining on hundreds of billions of tokens. This section covers the **objective** (next-token prediction and its variants), the **scaling laws** that govern how loss falls with compute, data, and parameters, the **data engineering** choices that quietly dominate model quality, and the **training dynamics** (learning rate schedules, gradient clipping, instability patterns) that separate a successful run from a crashed one.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (template stub, awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q3‑01 | [What is language model pretraining? What objective is minimised and what does a model learn?](./q01-pretraining-objective.md) | pretraining · next-token · cross-entropy · CLM | ✅ |
| Q3‑02 | [What are neural scaling laws? Explain the Kaplan et al. (2020) findings on parameters, data, and compute.](./q02-scaling-laws-kaplan.md) | scaling-laws · Kaplan · power-law · compute-budget | ✅ |
| Q3‑03 | [What is the Chinchilla scaling law and how does it change the compute-optimal recipe?](./q03-chinchilla.md) | Chinchilla · compute-optimal · tokens-per-param · Hoffmann | ✅ |
| Q3‑04 | [What is a pretraining data mix and why does the mixture of domains matter for downstream quality?](./q04-data-mix.md) | data-mix · domain-weights · quality-filtering · deduplication | ✅ |
| Q3‑05 | [What are emergent abilities in LLMs? Are they real phase transitions or a measurement artefact?](./q05-emergent-abilities.md) | emergence · phase-transition · BIG-Bench · metric-artefact | ✅ |
| Q3‑06 | [How are learning rate schedules chosen for LLM pretraining? What is cosine decay with warmup?](./q06-lr-schedule.md) | learning-rate · cosine-decay · warmup · WSD | ✅ |
| Q3‑07 | [What is gradient clipping and why is it essential for LLM training stability?](./q07-gradient-clipping.md) | gradient-clipping · training-stability · loss-spike · Adam | ✅ |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q3‑08 | [How does the compute-optimal token count scale with model size under Chinchilla? Derive the 20× rule.](./q08-chinchilla-derivation.md) | Chinchilla · derivation · IsoFLOP · token-budget | ✅ |
| Q3‑09 | [What is data deduplication and why does near-duplicate data hurt scaling?](./q09-deduplication.md) | deduplication · MinHash · exact-match · data-quality | ✅ |
| Q3‑10 | [What is the role of batch size in LLM pretraining? How does gradient noise scale relate to the critical batch size?](./q10-batch-size.md) | batch-size · gradient-noise · critical-batch · throughput | ✅ |
| Q3‑11 | [What is mixed-precision training (BF16/FP32)? How does the master weight copy prevent underflow?](./q11-mixed-precision.md) | mixed-precision · BF16 · FP32 · master-weights · underflow | ✅ |
| Q3‑12 | [What is the Adam optimizer and why is it the default for LLM pretraining? What do the β₁, β₂, ε hyperparameters control?](./q12-adam-optimizer.md) | Adam · AdamW · weight-decay · β₁-β₂ · moment-estimates | ✅ |
| Q3‑13 | [What is weight decay and why is AdamW preferred over Adam for LLMs?](./q13-adamw-weight-decay.md) | AdamW · L2-regularization · decoupled-decay · overfitting | ✅ |
| Q3‑14 | [What are loss spikes during pretraining? How are they diagnosed and mitigated?](./q14-loss-spikes.md) | loss-spikes · instability · checkpoint-rollback · gradient-clip | ✅ |
| Q3‑15 | [How does the data quality filtering pipeline work? What heuristics are used in Common Crawl processing?](./q15-data-quality-filtering.md) | Common-Crawl · quality-filter · perplexity-filter · heuristics | ✅ |
| Q3‑16 | [What is compute-optimal training vs. inference-optimal training? How does LLaMA differ from GPT-4 in philosophy?](./q16-compute-vs-inference-optimal.md) | compute-optimal · inference-optimal · LLaMA · overtrained | ✅ |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q3‑17 | [Derive the Chinchilla loss function and the IsoFLOP analysis from scratch.](./q17-chinchilla-derivation-full.md) | Chinchilla · IsoFLOP · parametric-fit · Hoffmann | 📝 |
| Q3‑18 | [What is the neural scaling law for downstream task performance? How does loss predict accuracy?](./q18-downstream-scaling.md) | downstream-scaling · loss-accuracy · emergent-threshold | 📝 |
| Q3‑19 | [What is the MuP (Maximal Update Parametrization) and how does it enable hyperparameter transfer across scales?](./q19-mup.md) | MuP · hyperparameter-transfer · learning-rate · width-scaling | 📝 |
| Q3‑20 | [What happens during the cooldown / decay phase of pretraining? How does learning rate decay interact with checkpoint selection?](./q20-cooldown-phase.md) | cooldown · WSD · checkpoint · final-tokens | 📝 |
| Q3‑21 | [What is the Scaling Law for Neural Language Models in the context of different data regimes (data-constrained vs. compute-constrained)?](./q21-data-constrained-scaling.md) | data-constrained · repeated-epochs · Muennighoff | 📝 |
| Q3‑22 | [How does pretraining on code affect general reasoning ability? What is the evidence from CodeLlama and Llama-3?](./q22-code-pretraining-reasoning.md) | code-pretraining · reasoning · CodeLlama · Llama-3 | 📝 |
| Q3‑23 | [What are the Chinchilla-optimal predictions for GPT-4 class models? Where do the "100T parameter" predictions come from and why are they wrong?](./q23-gpt4-chinchilla-predictions.md) | GPT-4 · Chinchilla · extrapolation · parameter-count | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q3‑24 | [You have a fixed compute budget of 10²³ FLOPs. How do you allocate it between model size and training tokens?](./q24-compute-budget-allocation.md) | compute-budget · IsoFLOP · Chinchilla · allocation | 📝 |
| Q3‑25 | [Your pretraining loss curve has a spike at step 40K and never fully recovers. Walk through your diagnosis and recovery plan.](./q25-loss-spike-debug.md) | loss-spike · debugging · rollback · data-corruption | 📝 |
| Q3‑26 | [How would you design a data pipeline for a 10T-token pretraining run? What are the bottlenecks?](./q26-data-pipeline-design.md) | data-pipeline · deduplication · shuffling · throughput | 📝 |
| Q3‑27 | [A team wants to pretrain a model that is fast to serve but also high quality. How do you trade off model size vs. training tokens?](./q27-serve-vs-quality-tradeoff.md) | inference-optimal · overtrained · LLaMA · serving-cost | 📝 |
| Q3‑28 | [How do you continue pretraining (CPT) a model on a new domain without catastrophic forgetting of general capabilities?](./q28-continued-pretraining.md) | CPT · catastrophic-forgetting · domain-adaptation · replay | 📝 |
| Q3‑29 | [Walk through the Llama-3 pretraining recipe: data mix, curriculum, context length extension, and annealing.](./q29-llama3-recipe.md) | Llama-3 · pretraining-recipe · curriculum · annealing | 📝 |
| Q3‑30 | [What metrics do you track during pretraining to catch instability before it becomes unrecoverable?](./q30-pretraining-monitoring.md) | monitoring · gradient-norm · loss-curve · instability | 📝 |
| Q3‑31 | [How do you evaluate a partially-trained checkpoint to predict final model quality without running to convergence?](./q31-early-stopping-prediction.md) | early-stopping · loss-prediction · IsoFLOP · eval-set | 📝 |
| Q3‑32 | [Design a scaling experiment to validate that your new architecture follows expected scaling laws before committing to a large run.](./q32-scaling-experiment-design.md) | scaling-experiment · IsoFLOP · pilot-run · architecture-validation | 📝 |

---

## Why these questions matter

Scaling-law questions are now standard at frontier labs because they probe whether a candidate can **reason about resource allocation under uncertainty**. The questions above cover four recurring interview themes:

| Theme | Key questions |
|-------|--------------|
| **Law derivation** | Q3‑02, Q3‑03, Q3‑08, Q3‑17 — rebuild Kaplan and Chinchilla from first principles |
| **Data engineering** | Q3‑04, Q3‑09, Q3‑15, Q3‑26 — quality filtering, deduplication, domain mixing |
| **Training dynamics** | Q3‑06, Q3‑07, Q3‑11, Q3‑12, Q3‑14 — schedules, optimizers, instability |
| **Resource allocation** | Q3‑16, Q3‑24, Q3‑27, Q3‑29 — compute-optimal vs. inference-optimal decisions |

---

## Reading order suggestions

**New to pretraining?** Q3‑01 → Q3‑02 → Q3‑03 → Q3‑06 → Q3‑07

**Interview in 48 hours?** Q3‑03 (Chinchilla) · Q3‑05 (emergence) · Q3‑02 (Kaplan) · Q3‑04 (data mix) · Q3‑16 (compute vs. inference optimal)

**Full Basic track:** Q3‑01 → Q3‑02 → Q3‑03 → Q3‑04 → Q3‑05 → Q3‑06 → Q3‑07

**Full Intermediate track:** Q3‑08 → Q3‑09 → Q3‑10 → Q3‑11 → Q3‑12 → Q3‑13 → Q3‑14 → Q3‑15 → Q3‑16

**Applied / debugging track:** Q3‑24 → Q3‑25 → Q3‑26 → Q3‑27 → Q3‑28 → Q3‑29 → Q3‑30 → Q3‑31 → Q3‑32

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 2 — Tokenization & Embeddings](../section-02-tokenization-and-embeddings/README.md) &nbsp;|&nbsp; [Section 4 — Post-training: SFT, RLHF, DPO ➡️](../../README.md#section-4--post-training-sft-rlhf-dpo-and-beyond)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s3qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
