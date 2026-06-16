<!-- ============================ TOP NAV ============================ -->
<div align="center">

[🏠 Home](../../README.md) &nbsp;•&nbsp; [📚 Section 3 — Pretraining & Scaling Laws](./README.md) &nbsp;•&nbsp; [⬅️ Q3‑15 — Data Quality Filtering](./q15-data-quality-filtering.md) &nbsp;•&nbsp; [Q3‑17 — Chinchilla Derivation Full ➡️](./q17-chinchilla-derivation-full.md)

</div>

---

# Q3‑16 · What is compute-optimal training vs. inference-optimal training? How does LLaMA differ from GPT-4 in philosophy?

<div align="center">

![Section](https://img.shields.io/badge/Section_3-Pretraining_%26_Scaling_Laws-1f4fa3)
![Level](https://img.shields.io/badge/Level-Intermediate-c77a12)
![Tags](https://img.shields.io/badge/tags-compute--optimal·inference--optimal·LLaMA·overtrained-c77a12)
![Read](https://img.shields.io/badge/read_time-~18_min-5b6675)

</div>

> [!IMPORTANT]
> **The 20-second answer.** **Compute-optimal training** (the Chinchilla recipe) minimises loss for a fixed training FLOP budget: use ~20 tokens per parameter. **Inference-optimal training** deliberately uses a *smaller* model trained on *far more* tokens than Chinchilla prescribes — paying extra training FLOPs to shrink the model size so that every one of the millions of inference calls is cheaper. LLaMA explicitly follows the inference-optimal philosophy: LLaMA-1 7B was trained on 1T tokens (143 tokens/param vs Chinchilla's 20), and LLaMA-3 8B on 15T tokens (≈1875 tokens/param). GPT-4 is believed to follow a compute-optimal recipe aligned with the training budget. The key insight is that **Chinchilla optimises training cost alone**, but total deployment cost = training cost + N_requests × inference cost per request — and for large-scale deployment, inference cost dominates.

---

## Table of contents

1. [First principles: what is a FLOP budget?](#1--first-principles-what-is-a-flop-budget)
2. [Compute-optimal training — the Chinchilla recipe](#2--compute-optimal-training--the-chinchilla-recipe)
3. [Why Chinchilla-optimal is not deployment-optimal](#3--why-chinchilla-optimal-is-not-deployment-optimal)
4. [Inference-optimal training — the core idea](#4--inference-optimal-training--the-core-idea)
5. [Breakeven analysis: when is overtraining worthwhile?](#5--breakeven-analysis-when-is-overtraining-worthwhile)
6. [LLaMA 1, 2, and 3 — a case study in inference optimality](#6--llama-1-2-and-3--a-case-study-in-inference-optimality)
7. [GPT-4 philosophy — compute-optimal at scale](#7--gpt-4-philosophy--compute-optimal-at-scale)
8. [Mistral 7B, Phi, and Gemini Nano — the edge continuum](#8--mistral-7b-phi-and-gemini-nano--the-edge-continuum)
9. [Is "overtrained" the right word?](#9--is-overtrained-the-right-word)
10. [Scaling law for overtrained models](#10--scaling-law-for-overtrained-models)
11. [Tokens-per-parameter comparison across major LLMs](#11--tokens-per-parameter-comparison-across-major-llms)
12. [Practical decision tree](#12--practical-decision-tree)
13. [Interview drill](#13--interview-drill)
14. [Common misconceptions](#14--common-misconceptions)
15. [References](#15--references)

---

## 1 · First principles: what is a FLOP budget?

Training a language model costs **compute**, measured in floating-point operations (FLOPs). For a dense Transformer with $N$ parameters trained on $D$ tokens, the training FLOPs are approximately:

$$C \approx 6ND$$

The factor of 6 accounts for the forward pass (2 FLOPs per multiply-add), the backward pass (roughly 2× the forward pass), and activation recomputation overhead. This approximation, introduced by Kaplan et al. (2020) and confirmed by Hoffmann et al. (2022), holds to within ~10% for standard dense Transformers.

A **FLOP budget** $C$ is the total compute available — set by how many GPUs you have, how long you are willing to run them, and the cost per GPU-hour. Given a fixed $C$, there are many $(N, D)$ pairs you can choose:

- Large $N$, small $D$: a big model trained on few tokens.
- Small $N$, large $D$: a small model trained on many tokens.
- The **Chinchilla-optimal** point: the $(N^*, D^*)$ pair that minimises loss for that budget.

The entire question of compute-optimal vs. inference-optimal is about which $(N, D)$ you choose — and that choice has profound consequences for **who pays** and **how much**.

---

## 2 · Compute-optimal training — the Chinchilla recipe

Hoffmann et al. (2022) trained over 400 models ranging from 70M to 16B parameters on 5B to 500B tokens, fitting a parametric loss function:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where $E$ is the irreducible entropy of natural language, and $A, B, \alpha, \beta$ are fit constants. Holding $C = 6ND$ fixed and minimising $L$ with respect to $N$ and $D$ yields the **Chinchilla scaling law**:

$$N^* \propto C^{0.50}, \quad D^* \propto C^{0.50}$$

Both model size and token count should scale equally with compute. The practical rule of thumb:

$$D^* \approx 20 \times N^*$$

That is, train each parameter on approximately 20 tokens. Chinchilla 70B achieved lower loss than Gopher 280B while using 4× fewer parameters — because Gopher was severely undertrained (only ~1 token/param).

**What "compute-optimal" means operationally:** given a fixed training budget, choose the model size and token count that minimises validation loss at the *end of training*. The model is then deployed. The entire cost accounting is about the single training run.

> [!NOTE]
> Compute-optimal is sometimes called "Chinchilla-optimal" or "iso-compute optimal." All three terms refer to the same thing: minimising loss for a fixed $C$ with no consideration of post-training inference costs.

---

## 3 · Why Chinchilla-optimal is not deployment-optimal

Chinchilla optimises one number: **loss at the end of training**. But a model is not used once — it is used millions or billions of times after training. The **total cost of ownership** of a model is:

$$\text{Total cost} = C_{\text{train}} + N_{\text{requests}} \times C_{\text{infer per request}}$$

The inference cost per request scales roughly linearly with model size $N$ (for a given context length and batch size). A 70B model costs approximately 10× more per token at inference than a 7B model.

Consider two strategies for achieving a target quality level $L^*$:

| Strategy | Model size | Training tokens | Training cost | Inference cost/request |
|---|---|---|---|---|
| Compute-optimal | 67B | 1.34T | $C_1$ | High |
| Inference-optimal | 7B | 1T | $C_1 + \Delta$ | 10× lower |

The inference-optimal strategy pays extra training FLOPs ($\Delta$) to use a model 10× smaller at inference. If you serve this model $N_{\text{req}}$ times, the inference savings are:

$$\text{Savings} = N_{\text{req}} \times (C_{\text{infer, 67B}} - C_{\text{infer, 7B}})$$

For large enough $N_{\text{req}}$, savings exceed $\Delta$ and the inference-optimal strategy wins on total cost.

---

## 4 · Inference-optimal training — the core idea

**Definition.** Inference-optimal training deliberately selects a model *smaller* than the Chinchilla-optimal size for a given quality target, then compensates by training on *more tokens* than Chinchilla prescribes. The goal is to match (or approach) the quality of the Chinchilla-optimal model while reducing inference cost.

The key empirical observation, confirmed by Touvron et al. (2023a) and subsequent work, is that **LLM training loss continues to decrease monotonically with more tokens, even well past the Chinchilla-optimal token count**. There is no U-curve — adding more tokens does not hurt the model. The loss just decreases more slowly than during the Chinchilla-optimal training period.

This means: if you have a 7B model that is Chinchilla-optimal at 140B tokens, you can keep training it on 1T tokens and it will continue to improve. By 1T tokens it will have recovered much of the quality gap relative to the Chinchilla-optimal 67B model — but it will be 9.6× cheaper at inference.

<div align="center">
<img src="../../assets/s3q16/fig1-compute-vs-inference-optimal.svg" alt="Two-axis plot showing model size N on the x-axis from 1B to 100B and loss at target quality on the y-axis. A dashed teal Chinchilla-optimal frontier curve marks the minimum compute needed to reach each quality level at each model size. A solid navy fixed-compute curve (C=10^23) shows quality achieved for each model size at fixed training compute. A solid amber fixed-inference-cost curve shows that smaller models trained longer can match larger models trained for less. The compute-optimal point is marked at approximately 67B where the curves intersect. An arrow pointing left toward 7B-13B marks the inference-optimal zone with the annotation train longer." width="92%">
<br><sub><b>Figure 1.</b> Compute-optimal vs. inference-optimal: the LLaMA design decision. The navy curve (fixed training budget C = 10^23 FLOPs) peaks in quality at the Chinchilla-optimal ~67B. Moving left on this curve to 7B gives worse quality for the same training budget — but the amber curve shows that training the 7B for more tokens (inference-optimal) can recover that quality at 10× lower inference cost per request. LLaMA exploits exactly this trade-off.</sub>
</div>

---

## 5 · Breakeven analysis: when is overtraining worthwhile?

Let $\Delta C_{\text{train}}$ be the extra training cost from using more tokens on a smaller model, and let $\Delta C_{\text{infer}}$ be the savings per inference request from using that smaller model. The breakeven number of requests is:

$$N_{\text{req}}^{\text{break}} = \frac{\Delta C_{\text{train}}}{\Delta C_{\text{infer per request}}}$$

**Worked example.** Suppose training Chinchilla-optimal 67B costs $10M and the inference-optimal 7B trained on 1T tokens costs $1.5M to train. The inference cost per 1M tokens is $10 for 67B and $1 for 7B.

$$N_{\text{req}}^{\text{break}} = \frac{\$10M - \$1.5M}{\$10/\text{req} - \$1/\text{req}} = \frac{\$8.5M}{\$9/\text{req}} \approx 944{,}000 \text{ requests}$$

At about 1M requests the strategies cost the same total. Every request beyond 1M saves $9. For any consumer product or API serving millions of daily requests, inference-optimal is the obvious choice.

**The asymmetry of training vs. inference costs** makes the analysis stark:

- Training is paid **once** by the model developer (or research team).
- Inference is paid **per request**, typically by the end user or the serving infrastructure.
- At 100M requests per day (a modest production API), even a $1 per-request savings amounts to $36.5B per year.

This is why Meta, Mistral, and Microsoft (Phi series) have all explicitly adopted inference-optimal strategies for their open-weight models, even though it means spending more on training compute than Chinchilla prescribes.

---

## 6 · LLaMA 1, 2, and 3 — a case study in inference optimality

The LLaMA family is the most explicit and well-documented example of inference-optimal training. Touvron et al. (2023a) state directly in the abstract: *"We train at a higher token budget than is Chinchilla-optimal to reduce inference costs."*

### LLaMA 1 (February 2023)

| Model | Parameters | Training tokens | Tokens / param | Chinchilla-optimal tokens/param |
|---|---|---|---|---|
| LLaMA 7B | 7B | 1T | **143** | ~20 |
| LLaMA 13B | 13B | 1T | **77** | ~20 |
| LLaMA 33B | 33B | 1.4T | **42** | ~20 |
| LLaMA 65B | 65B | 1.4T | **22** | ~20 |

The 7B model was trained on 7× more tokens than Chinchilla prescribes. The result was that LLaMA 7B outperformed GPT-3 175B (which was severely undertrained at ~4 tokens/param) on many benchmarks — a smaller model trained longer beats a larger model trained insufficiently.

### LLaMA 2 (July 2023)

LLaMA 2 extended the token budget further: the 70B model was trained on 2T tokens (~29 tokens/param). While still above Chinchilla-optimal, the gap narrowed compared to LLaMA 1's smaller models, because the Chinchilla-optimal token count for 70B is already ~1.4T.

### LLaMA 3 (April 2024)

LLaMA 3 8B was trained on **15T tokens** — approximately **1875 tokens/param**. This is one of the most extreme documented cases of inference-optimal training. The quality gains from massively overtraining the 8B model are substantial: LLaMA 3 8B outperforms LLaMA 2 70B on most benchmarks despite having 8.75× fewer parameters, making it ideal for edge, mobile, and high-throughput serving.

> [!NOTE]
> The trend across LLaMA generations is clear: each successive generation pushes the tokens/param ratio higher. This reflects the empirical finding that loss continues to decrease with more tokens — there is always value in training longer if inference cost matters.

---

## 7 · GPT-4 philosophy — compute-optimal at scale

GPT-4's training details have not been publicly disclosed by OpenAI. However, several inferences are reasonable based on what is known:

**GPT-4 is a very large model.** Estimates from leaked reports and independent analysis place GPT-4 at roughly 1–2T parameters (mixture-of-experts architecture), though this is unconfirmed. At that scale, the Chinchilla-optimal token count is already extremely large — a 1T parameter model requires ~20T tokens for Chinchilla-optimal training. Given the cost of generating and curating 20T high-quality tokens, it is likely that GPT-4 was trained for *fewer* tokens than Chinchilla prescribes, making it **undertrained** by Chinchilla standards.

**The GPT-4 philosophy is scale-first.** OpenAI's approach in GPT-3 (Brown et al., 2020) was to maximise model scale given the available compute, with relatively few tokens per parameter (~4 tokens/param for GPT-3, severely undertrained by Chinchilla). GPT-4 represents a continuation of this philosophy: invest in a very large, capable model and serve it via API where inference cost is directly charged to users, decoupling the training-inference cost trade-off.

**The key difference from LLaMA:**

| Dimension | LLaMA philosophy | GPT-4 philosophy |
|---|---|---|
| Primary constraint | Inference cost (open weights, community use) | Model capability ceiling |
| Model size | Small-to-medium (7B–70B) | Very large (estimated 1T+) |
| Token budget | Massively overtrained | Approximately Chinchilla-optimal or undertrained |
| Deployment | Open weights, self-hosted | Proprietary API |
| Serving economics | User bears inference cost | OpenAI bears inference cost, charges per token |

The two philosophies are not in conflict — they reflect genuinely different deployment contexts. GPT-4 is an API product where OpenAI can optimise serving infrastructure (batching, speculative decoding, hardware) without the constraint of fitting the model on a user's GPU. LLaMA is designed to run on commodity hardware where model size is the primary constraint.

---

## 8 · Mistral 7B, Phi, and Gemini Nano — the edge continuum

Several other model families have adopted inference-optimal strategies, each pushing further toward the edge:

**Mistral 7B (2023).** Jiang et al. (2023) trained Mistral 7B on approximately 1–1.4T tokens (~143–200 tokens/param) with the explicit goal of edge and consumer deployment. Mistral 7B outperformed LLaMA 2 13B on most benchmarks, demonstrating that a smaller, well-trained model can match a larger, under-trained one.

**Microsoft Phi series (2023–2024).** The Phi-1 through Phi-3 models (1.3B–14B parameters) are trained on "textbook-quality" synthetic and curated data for 100B–5T tokens. Phi-3-mini (3.8B) trained on 3.3T tokens achieves performance comparable to Mistral 7B and LLaMA 2 7B while being 2× smaller — an extreme inference-optimal design for mobile deployment.

**Gemini Nano (2023).** Google's Gemini Nano 1 (1.8B) and Nano 2 (3.25B) are designed to run on-device on Pixel phones. Their training details are not disclosed, but the design philosophy is necessarily inference-optimal: the models must fit in device RAM and perform inference at interactive speeds on a mobile NPU.

The pattern across all these models is identical: **as the target deployment hardware becomes more constrained, the inference-optimal strategy becomes more aggressive** — smaller model sizes paired with ever-larger token budgets.

---

## 9 · Is "overtrained" the right word?

The term "overtrained" is borrowed from classical machine learning, where it implies **overfitting**: the model memorises training data and performs worse on held-out test data. For LLMs trained on internet-scale data, this usage is **misleading**:

- LLM pretraining operates in a regime where the number of unique training tokens (1T–15T) is vastly smaller than the model's capacity to memorise, but the diversity of the data is so high that generalisation is maintained even at 1875 tokens/param.
- Validation loss continues to **decrease** at 1T tokens for a 7B model — if the model were overfitting, validation loss would increase.
- What is "over" about overtraining is only that the training is beyond the Chinchilla-optimal point for **training compute efficiency** — not that the model is learning the wrong thing.

> [!WARNING]
> Never say "the model is overfitting" as a synonym for "overtrained" in this context. The model is not overfitting. It is trained past the point of maximum training compute efficiency, which is a deliberate choice, not an error. An interviewer who hears "overfitting" will likely mark you down.

A better term is **training-compute-inefficient but inference-efficient**: you are spending more training FLOPs per unit of quality improvement than the Chinchilla-optimal point, but you are doing so because the inference savings justify the extra training cost.

---

## 10 · Scaling law for overtrained models

Hoffmann et al. (2022) noted that their parametric loss function $L(N, D)$ continues to decrease with $D$ for fixed $N$ — there is no minimum in the $D$ direction. This means:

$$\frac{\partial L}{\partial D} < 0 \quad \text{for all } D$$

Training loss always decreases with more tokens, all else equal. The rate of decrease slows (the benefit is sub-linear), but it never reverses. This is the theoretical underpinning of inference-optimal training.

Touvron et al. (2023a) provide empirical confirmation: LLaMA 7B at 1T tokens has a validation loss (on diverse benchmarks) that is lower than LLaMA 7B at 200B tokens and lower than LLaMA 13B at 200B tokens. More tokens on a smaller model eventually beats fewer tokens on a larger model.

The **diminishing returns** structure means the gain from the 800B tokens between 200B and 1T is less than the gain from the first 200B tokens. This is why the LLaMA 3 jump to 15T tokens yields improvements, but the improvements per additional token are smaller at 10T tokens than at 1T tokens. The key question is whether the cost of those additional tokens is less than the cumulative inference savings they enable.

---

## 11 · Tokens-per-parameter comparison across major LLMs

<div align="center">
<img src="../../assets/s3q16/fig2-tokens-per-param-comparison.svg" alt="Horizontal bar chart comparing tokens per parameter for major language models. A vertical dashed line marks 20, the Chinchilla-optimal baseline. GPT-3 175B is shown at approximately 4 tokens per param labeled as undertrained. Gopher 280B is shown near 1 token per param labeled severely undertrained. Chinchilla 70B is shown at 20 tokens per param labeled as the optimal baseline. LLaMA-1 7B is shown at 143 tokens per param. LLaMA-2 70B is shown at 29 tokens per param. LLaMA-3 8B is shown at 1875 tokens per param. Mistral 7B is shown at approximately 200 tokens per param. Annotations mark the region below 20 as compute-efficient but serve-expensive and the region above 20 as inference-optimal." width="92%">
<br><sub><b>Figure 2.</b> Tokens per parameter for major LLMs versus the Chinchilla-optimal baseline of 20 tokens/param (dashed line). GPT-3 and Gopher were severely undertrained by Chinchilla standards. LLaMA 1 7B and Mistral 7B are modestly overtrained. LLaMA 3 8B at 1875 tokens/param represents the far extreme of inference-optimal design. Note the log scale on the x-axis.</sub>
</div>

The chart reveals the industry's evolution in three phases:

1. **Pre-Chinchilla (2020–2021):** GPT-3, Gopher — undertrained large models, compute-inefficient and serve-expensive.
2. **Chinchilla era (2022):** Hoffmann et al. establish the 20 tokens/param optimum; Chinchilla 70B validates it.
3. **Inference-optimal era (2023–present):** LLaMA, Mistral, Phi, Gemini Nano — deliberately overtrained small models optimised for deployment cost.

---

## 12 · Practical decision tree

Use this framework when deciding between compute-optimal and inference-optimal training:

```
Is inference cost a concern?
├── No (research, one-off evaluation, capability benchmark)
│   └── Use compute-optimal (Chinchilla recipe: 20 tokens/param)
└── Yes
    ├── How many inference requests do you expect?
    │   ├── < 1M requests (small internal tool)
    │   │   └── Compute-optimal is probably fine;
    │   │       breakeven not reached
    │   ├── 1M – 100M requests (mid-scale product)
    │   │   └── Inference-optimal is worth considering;
    │   │       run breakeven calculation
    │   └── > 100M requests (large-scale API or consumer product)
    │       └── Inference-optimal is almost certainly correct;
    │           train smaller model for more tokens
    └── What hardware will serve the model?
        ├── Data centre with full GPU cluster
        │   └── Moderate inference-optimal bias (LLaMA-2 style)
        └── Edge / mobile / consumer GPU
            └── Aggressive inference-optimal (LLaMA-3 / Phi style;
                target model size < 8B, maximize tokens)
```

---

## 13 · Interview drill

<details>
<summary><b>Q: Why is Chinchilla-optimal not the same as deployment-optimal?</b></summary>

Chinchilla-optimal minimises loss for a fixed training FLOP budget. It treats the training run as the entire cost. Deployment-optimal accounts for total cost of ownership: training cost + (number of inference requests × cost per inference request). Inference cost scales with model size; a Chinchilla-optimal model for a large training budget will be large, making every inference call expensive. If the model will be served millions of times, it is often cheaper overall to pay more training FLOPs for a smaller model that costs less per inference request.
</details>

<details>
<summary><b>Q: LLaMA 7B was trained on 1T tokens. How many tokens does Chinchilla say it should be trained on?</b></summary>

The Chinchilla rule of thumb is 20 tokens per parameter. For 7B parameters, that is 20 × 7B = 140B tokens. LLaMA 7B was trained on 1T tokens — roughly 7× more than Chinchilla prescribes. This is intentional: Touvron et al. explicitly state they train beyond Chinchilla-optimal to reduce inference costs for community use.
</details>

<details>
<summary><b>Q: Is LLaMA 3 8B "overfitting" by being trained on 15T tokens?</b></summary>

No. Overfitting in the classical sense means validation loss increases while training loss decreases — the model memorises training data. For LLaMA 3 8B at 15T tokens, validation loss continues to decrease (it does not turn up). The model is trained past the Chinchilla-optimal point for training compute efficiency, but this is not the same as overfitting. The term "overtrained" is misleading — it means "trained beyond Chinchilla-optimal," not "overfitting." A better description is "inference-optimal" or "training-compute-inefficient."
</details>

<details>
<summary><b>Q: GPT-3 was trained on approximately 300B tokens with 175B parameters — is it compute-optimal?</b></summary>

No. GPT-3 was trained before the Chinchilla paper. At 175B parameters, Chinchilla-optimal would require roughly 3.5T tokens (20 × 175B). GPT-3 was trained on only 300B tokens — about 1.7 tokens/param, severely undertrained by Chinchilla standards. Brown et al. (2020) were following Kaplan et al. (2020) scaling laws, which incorrectly suggested model size should scale faster than data. Chinchilla corrected this, showing that data scaling should match model size scaling.
</details>

<details>
<summary><b>Q: What is the practical significance of LLaMA 3 8B outperforming LLaMA 2 70B?</b></summary>

It demonstrates that inference-optimal training can produce a 8.75× smaller model that matches the quality of a larger, less-trained model. For anyone deploying at scale, this means: instead of serving a 70B model (which requires multi-GPU inference or very high-end single GPU), you can serve an 8B model on a single consumer GPU or a modest server, with 8-10× lower inference cost per request, at equal or better quality. This is the practical payoff of the inference-optimal philosophy — it democratises high-quality model deployment.
</details>

<details>
<summary><b>Q: A colleague says "we should always train compute-optimally to maximise quality per FLOP." How do you respond?</b></summary>

This is correct if your only concern is training efficiency and you will use the model a fixed number of times. But "quality per training FLOP" is the wrong objective when inference cost matters. The correct objective is "quality per total-cost-of-ownership FLOP," which includes all inference calls. For most production deployments serving millions of requests, optimising for inference cost dominates the total cost calculation. The compute-optimal strategy is right for research and capability benchmarking; the inference-optimal strategy is right for deployed products. Both are correct in their respective contexts — the key is to be explicit about which cost you are optimising.
</details>

---

## 14 · Common misconceptions

| Misconception | Reality |
|---|---|
| "Overtrained models are overfitting." | They are not. LLM validation loss continues to decrease with more tokens. "Overtrained" means past Chinchilla-optimal for training efficiency, not overfitting. |
| "Chinchilla proved you should always use 20 tokens/param." | Chinchilla proved 20 tokens/param minimises loss for a fixed *training* budget. It says nothing about deployment costs. |
| "GPT-4 is inference-optimal." | GPT-4 is likely a very large model (possibly 1T+ params) trained approximately Chinchilla-optimally for its scale. LLaMA is the inference-optimal approach; GPT-4 targets peak capability. |
| "You can always improve quality by adding more tokens." | Yes, but with diminishing returns. The marginal gain per token decreases as tokens/param increases. At 1875 tokens/param the gain per additional token is very small. |
| "Inference-optimal training is only for small models." | The principle applies at any scale. The decision is about the relative costs of training and serving, not about model size per se. A 70B model can be inference-optimal if trained on 1.4T+ tokens. |
| "GPT-3 was undertrained because OpenAI made a mistake." | GPT-3 followed the Kaplan et al. (2020) scaling laws, which preceded Chinchilla. It was not a mistake — Chinchilla corrected the field's understanding of the optimal training frontier. |

---

## 15 · References

1. Hoffmann, J. et al. — **Training Compute-Optimal Large Language Models** (Chinchilla). *NeurIPS 2022 / arXiv:2203.15556.* — Establishes the 20 tokens/param rule and the parametric loss function $L(N, D)$ used throughout this answer.

2. Touvron, H. et al. — **LLaMA: Open and Efficient Foundation Language Models**. *arXiv:2302.13971, 2023a.* — Introduces the explicit inference-optimal philosophy; trains 7B on 1T tokens (7× Chinchilla) to minimise inference cost.

3. Touvron, H. et al. — **Llama 2: Open Foundation and Fine-Tuned Chat Models**. *arXiv:2307.09288, 2023b.* — Extends to 2T tokens for 70B model; confirms the diminishing-returns structure of overtraining.

4. Meta AI — **Llama 3 Model Card and Technical Overview**. *meta.com/llama, 2024.* — LLaMA 3 8B trained on 15T tokens; most extreme published case of inference-optimal training.

5. Kaplan, J. et al. — **Scaling Laws for Neural Language Models**. *arXiv:2001.08361, 2020.* — The pre-Chinchilla scaling laws; incorrectly suggested model size should scale faster than data, leading to GPT-3 and Gopher being undertrained.

6. Brown, T. et al. — **Language Models are Few-Shot Learners** (GPT-3). *NeurIPS 2020 / arXiv:2005.14165.* — GPT-3 training at 300B tokens on 175B params (~1.7 tokens/param); severely undertrained by Chinchilla standards.

7. Rae, J. et al. — **Scaling Language Models: Methods, Analysis & Insights from Training Gopher**. *arXiv:2112.11446, 2021.* — Gopher 280B trained on ~300B tokens (~1 token/param); the most extreme example of pre-Chinchilla undertraining.

8. Jiang, A. et al. — **Mistral 7B**. *arXiv:2310.06825, 2023.* — Mistral 7B trained inference-optimally for edge and consumer deployment; outperforms LLaMA 2 13B.

9. Abdin, M. et al. — **Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone**. *arXiv:2404.14219, 2024.* — Phi-3-mini 3.8B trained on 3.3T tokens; extreme inference-optimal for mobile deployment.

10. Anil, R. et al. — **Gemini: A Family of Highly Capable Multimodal Models**. *arXiv:2312.11805, 2023.* — Gemini Nano 1 and 2; on-device inference-optimal models for Pixel phones.

11. Clark, A. et al. — **Unified Scaling Laws for Routed Language Models**. *arXiv:2202.01169, 2022.* — Extends Chinchilla analysis to mixture-of-experts architectures; relevant context for GPT-4 MoE estimates.

---

<!-- ============================ BOTTOM NAV ============================ -->
<div align="center">

[⬅️ Q3‑15 — Data Quality Filtering](./q15-data-quality-filtering.md) &nbsp;|&nbsp; [📚 Back to Section 3](./README.md) &nbsp;|&nbsp; [🏠 Home](../../README.md) &nbsp;|&nbsp; [Q3‑17 — Chinchilla Derivation Full ➡️](./q17-chinchilla-derivation-full.md)

<sub>Found an error or have a sharper intuition? See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a> — answers follow the <a href="../../_TEMPLATE.md">answer template</a>.</sub>

</div>
