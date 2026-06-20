<!-- ============================ TOP NAV ============================ -->
<div align="center">

[🏠 Home](../../README.md) &nbsp;•&nbsp; [📚 Section 4 — Post-training](./README.md) &nbsp;•&nbsp; [⬅️ Q4‑03 — Reward Model & Bradley-Terry](./q03-reward-model-bradley-terry.md) &nbsp;•&nbsp; [Q4‑05 — On-Policy vs. Off-Policy RL ➡️](./q05-on-policy-vs-off-policy.md)

</div>

---

# Q4‑04 · What is the KL penalty in RLHF for? What happens if you remove it?

<div align="center">

![Section](https://img.shields.io/badge/Section_4-Post--training%3A_SFT%2C_RLHF%2C_DPO-1f4fa3)
![Level](https://img.shields.io/badge/Level-Basic-0e8a6e)
![Tags](https://img.shields.io/badge/tags-KL--penalty·RLHF·policy--constraint·reward--hacking-c77a12)
![Read](https://img.shields.io/badge/read_time-~16_min-5b6675)

</div>

> [!IMPORTANT]
> **The 20-second answer.** The KL penalty in RLHF is a regularisation term that prevents the policy from drifting too far from the supervised fine-tuned (SFT) reference model. The full PPO objective is $\mathcal{J}(\theta) = \mathbb{E}[r_\phi(x,y) - \beta \cdot \mathbb{KL}[\pi_\theta \| \pi_{\text{ref}}]]$. The reward signal pushes the policy toward high-reward outputs; the KL term pulls it back toward the SFT distribution. The coefficient $\beta$ (typically 0.01–0.1) controls the tradeoff. Without the KL penalty ($\beta = 0$), the policy exploits the reward model's blind spots — a phenomenon called **reward hacking** — producing degenerate outputs like repetitive text or sycophantic phrases that fool the reward model but are useless or incoherent to real users. This is an instance of **Goodhart's Law**: when a measure becomes a target, it ceases to be a good measure.

---

## Table of contents

1. [First principles: why RL alone is dangerous](#1--first-principles-why-rl-alone-is-dangerous)
2. [The full RLHF PPO objective](#2--the-full-rlhf-ppo-objective)
3. [Figure 1 — Opposing forces on the policy](#3--figure-1--opposing-forces-on-the-policy)
4. [KL divergence: what it actually measures](#4--kl-divergence-what-it-actually-measures)
5. [Figure 2 — Reward hacking without KL penalty](#5--figure-2--reward-hacking-without-kl-penalty)
6. [The token-level KL approximation](#6--the-token-level-kl-approximation)
7. [PyTorch reference implementation](#7--pytorch-reference-implementation)
8. [Worked numerical example](#8--worked-numerical-example)
9. [Interview drill — follow-up questions](#9--interview-drill--follow-up-questions)
10. [Common misconceptions](#10--common-misconceptions)
11. [Connections to other concepts](#11--connections-to-other-concepts)
12. [One-screen summary](#12--one-screen-summary)
13. [Five-minute refresher](#13--five-minute-refresher)
14. [Further reading](#14--further-reading)
15. [References](#15--references)

---

## 1 · First principles: why RL alone is dangerous

In RLHF, a reward model $r_\phi(x, y)$ is trained on human preference pairs to score outputs. The policy $\pi_\theta$ is then optimised via RL (typically PPO) to maximise this score.

The problem is that the reward model is not a perfect proxy for human preferences. It is a neural network trained on a finite dataset of preference labels. Like all learned models, it has **blind spots** — regions of output space where it assigns high scores to outputs that humans would actually dislike. These arise from:

- **Distribution shift:** the reward model saw only a slice of possible outputs during training
- **Label noise:** human annotators disagree, make errors, or were not shown adversarial examples
- **Shortcut features:** the model may learn to reward superficial signals (length, formatting, confidence markers) rather than genuine quality

When the policy is trained purely to maximise reward, it will discover and exploit these blind spots. This is **reward hacking**. The reward score climbs, but actual output quality collapses. The policy is not doing what we intended — it is gaming the metric.

The KL penalty is the solution: by requiring the policy to stay close to the SFT reference model (which produces coherent language), we prevent exploitation of the reward model's weaknesses while still allowing genuine improvement.

---

## 2 · The full RLHF PPO objective

The standard RLHF objective with KL regularisation is:

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot \mathbb{KL}\left[\pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x)\right] \right]$$

where:

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Policy parameters (the LLM being trained) |
| $\phi$ | Reward model parameters (frozen during PPO) |
| $x \sim \mathcal{D}$ | Prompt sampled from the training distribution |
| $y \sim \pi_\theta(\cdot|x)$ | Response sampled from the current policy |
| $r_\phi(x, y)$ | Reward model score for response $y$ given prompt $x$ |
| $\pi_{\text{ref}}$ | Reference policy — the SFT model, frozen |
| $\beta$ | KL penalty coefficient, typically 0.01 – 0.1 |

**The role of each term:**

- **$r_\phi(x, y)$** — the reward signal. This is the primary learning signal. The policy is gradient-ascended on this term.
- **$-\beta \cdot \mathbb{KL}[\pi_\theta \| \pi_{\text{ref}}]$** — the KL penalty. This subtracts a cost proportional to how far $\pi_\theta$ has drifted from $\pi_{\text{ref}}$. It acts as a regulariser.

**Effect of $\beta$:**

- **High $\beta$ (e.g. 0.1–1.0):** The policy is strongly constrained to stay near the SFT model. Reward improvement is conservative; outputs remain natural but may not improve much.
- **Low $\beta$ (e.g. 0.01–0.05):** The policy has more freedom to chase reward. Improvement can be larger, but hacking risk increases.
- **$\beta = 0$:** No constraint. The policy can drift arbitrarily far from the SFT model. Reward hacking becomes nearly inevitable.

> [!NOTE]
> The $\beta$ hyperparameter is usually chosen via grid search on a validation set, balancing win-rate improvements against KL divergence growth. InstructGPT (Ouyang et al., 2022) report $\beta \approx 0.02$ in their final configuration.

---

## 3 · Figure 1 — Opposing forces on the policy

<div align="center">
<img src="../../assets/s4q04/fig1-kl-penalty-role.svg" alt="Diagram showing the RLHF PPO objective as opposing forces on the policy. On the left, the reward signal r_phi(x,y) pushes the policy pi_theta toward higher-reward outputs. On the right, the KL penalty beta times KL[pi_theta || pi_ref] pulls the policy back toward the SFT reference model. The beta coefficient in the center controls the tradeoff. Annotations show: small beta means maximum reward pursuit with higher hacking risk; large beta means stay close to SFT with safer but less improvement. The policy sits at equilibrium between these two forces." width="92%">
<br><sub><b>Figure 1.</b> The RLHF objective creates two opposing forces on the policy. The reward signal (left, teal) pushes the policy toward high-reward outputs. The KL penalty (right, red) pulls the policy back toward the frozen SFT reference model. The coefficient &#946; controls the tradeoff. At equilibrium the marginal reward gain equals the marginal KL cost. Without the KL term the left force acts unchecked and reward hacking follows.</sub>
</div>

---

## 4 · KL divergence: what it actually measures

The KL divergence (Kullback-Leibler divergence) from $\pi_\theta$ to $\pi_{\text{ref}}$ measures how much information is lost when we use $\pi_{\text{ref}}$ to approximate $\pi_\theta$:

$$\mathbb{KL}[\pi_\theta \| \pi_{\text{ref}}] = \sum_y \pi_\theta(y|x) \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

**Key properties relevant to RLHF:**

1. **KL = 0 if and only if $\pi_\theta = \pi_{\text{ref}}$** everywhere. At the start of PPO, KL starts near zero because the policy is initialised from the SFT model.

2. **KL ≥ 0 always.** It is a non-negative quantity by Gibbs' inequality. This makes it a natural penalty — it is zero when the policy has not changed, and grows as the policy deviates.

3. **KL is asymmetric.** $\mathbb{KL}[\pi_\theta \| \pi_{\text{ref}}] \neq \mathbb{KL}[\pi_{\text{ref}} \| \pi_\theta]$. RLHF uses the forward KL (policy divided by reference), which assigns infinite cost to any output that $\pi_\theta$ assigns nonzero probability but $\pi_{\text{ref}}$ assigns zero probability. This strongly prevents the policy from entering regions of output space that the SFT model considers impossible.

4. **Penalises mode-seeking.** The forward KL $\mathbb{KL}[\pi_\theta \| \pi_{\text{ref}}]$ is mode-seeking: it encourages $\pi_\theta$ to concentrate on modes of $\pi_{\text{ref}}$. This is exactly what we want — the policy should generate outputs that are plausible under the SFT distribution, just reweighted toward higher reward.

---

## 5 · Figure 2 — Reward hacking without KL penalty

<div align="center">
<img src="../../assets/s4q04/fig2-reward-hacking-without-kl.svg" alt="Two panels showing reward score over PPO training steps. Left panel (with KL penalty, beta=0.05): reward rises steadily and plateaus at a healthy level; the KL divergence curve (dashed) stays low throughout. Right panel (without KL penalty, beta=0): reward spikes sharply upward in the first few hundred steps, then collapses back down as the policy produces degenerate outputs that exploit the reward model but score low on actual quality. Annotation examples: the policy learns to repeat phrases like 'The answer is great! Great! Great!' which fool the reward model." width="92%">
<br><sub><b>Figure 2.</b> Left: with KL penalty (&#946; = 0.05), the reward increases steadily and plateaus as the KL divergence (dashed) stays bounded &#8212; the policy improves within a safe neighbourhood of the SFT model. Right: without KL penalty (&#946; = 0), the reward spikes rapidly as the policy discovers the reward model's blind spots, then collapses as outputs become degenerate (repetition, gibberish, sycophantic phrases). The right panel is a textbook case of reward hacking and Goodhart's Law.</sub>
</div>

---

## 6 · The token-level KL approximation

In practice, RLHF operates on sequences of tokens, not whole-sentence distributions. A response $y = (y_1, y_2, \ldots, y_T)$ is generated token by token. Using the chain rule of probability:

$$\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t | x, y_{<t})$$

The sequence-level KL therefore decomposes as a sum of token-level KLs:

$$\mathbb{KL}[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)] \approx \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})}$$

This is the per-token log-ratio, and it is the form actually computed in code. At each generation step $t$, you:

1. Run $\pi_\theta$ to get logits over the vocabulary; apply softmax to get $\pi_\theta(y_t | x, y_{<t})$
2. Run $\pi_{\text{ref}}$ (the frozen SFT model) on the same prefix; apply softmax to get $\pi_{\text{ref}}(y_t | x, y_{<t})$
3. Compute $\log \pi_\theta(y_t) - \log \pi_{\text{ref}}(y_t)$ for the actually sampled token $y_t$
4. Sum over all $T$ tokens in the response

The KL reward adjustment is then $r_{\text{KL}}(x, y) = -\beta \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})}$

This is added to the reward model score $r_\phi(x, y)$ at each step, giving the composite reward signal that PPO optimises.

---

## 7 · PyTorch reference implementation

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Setup ──────────────────────────────────────────────────────────────────────
# policy_model: the LLM being trained with PPO (parameters updated)
# ref_model:    frozen SFT model (reference); no gradients flow through it

def compute_kl_penalty(
    policy_model,
    ref_model,
    input_ids: torch.Tensor,       # (batch, seq_len) — prompt + response
    response_mask: torch.Tensor,   # (batch, seq_len) — 1 for response tokens
    beta: float = 0.05,
) -> torch.Tensor:
    """
    Compute the KL penalty term for a batch of (prompt, response) pairs.

    Returns:
        kl_penalty: (batch,) — per-sample KL penalty (positive value to subtract)
    """
    # ── Policy log-probs ───────────────────────────────────────────────────────
    with torch.no_grad() if not policy_model.training else torch.enable_grad():
        policy_logits = policy_model(input_ids).logits       # (B, L, V)

    policy_log_probs = F.log_softmax(policy_logits, dim=-1)  # (B, L, V)

    # ── Reference log-probs (frozen) ───────────────────────────────────────────
    with torch.no_grad():
        ref_logits = ref_model(input_ids).logits             # (B, L, V)

    ref_log_probs = F.log_softmax(ref_logits, dim=-1)        # (B, L, V)

    # ── Per-token log ratio for the sampled tokens ─────────────────────────────
    # We only need the log-prob of the token that was actually generated.
    # input_ids[:, 1:] gives the next-token targets; logits[:, :-1] predicts them.
    token_ids = input_ids[:, 1:].unsqueeze(-1)               # (B, L-1, 1)
    mask = response_mask[:, 1:]                              # (B, L-1)

    policy_lp   = policy_log_probs[:, :-1].gather(-1, token_ids).squeeze(-1)   # (B, L-1)
    ref_lp      = ref_log_probs[:, :-1].gather(-1, token_ids).squeeze(-1)      # (B, L-1)

    # KL per token: log pi_theta(y_t) - log pi_ref(y_t)
    per_token_kl = policy_lp - ref_lp                        # (B, L-1)

    # Sum only over response tokens; average over batch
    kl_sum = (per_token_kl * mask).sum(dim=-1)               # (B,)

    return beta * kl_sum                                     # (B,)  — subtract from reward


def compute_rlhf_reward(
    rm_score: torch.Tensor,       # (B,) — reward model score
    kl_penalty: torch.Tensor,     # (B,) — from compute_kl_penalty
) -> torch.Tensor:
    """
    Composite RLHF reward: r_phi(x, y) - beta * KL
    This is passed as the reward signal to PPO.
    """
    return rm_score - kl_penalty


# ── Monitoring KL during training ──────────────────────────────────────────────
def log_kl_stats(kl_penalty, beta, step):
    mean_kl = (kl_penalty / beta).mean().item()   # recover raw KL from penalty
    print(f"Step {step}: mean KL = {mean_kl:.4f}  (penalty = {kl_penalty.mean():.4f})")
    # Red flags:
    #   mean_kl > 20  →  policy drifting dangerously; increase beta or reduce LR
    #   mean_kl < 0.1 →  policy barely changing; maybe reduce beta or increase LR
```

> [!NOTE]
> In practice, the KL term is often computed as a per-token reward adjustment (added to the token-level advantage in PPO) rather than as a single scalar per sequence. TRL's `PPOTrainer` implements this token-level approach: each response token receives an adjusted reward of `rm_score_at_last_token - beta * log_ratio_t`.

---

## 8 · Worked numerical example

**Setup:** Vocabulary has two tokens: `A` and `B`. A one-token response. We compare policy and reference on a single step.

**Reference model (SFT, frozen):**

$$\pi_{\text{ref}}(A|x) = 0.7, \quad \pi_{\text{ref}}(B|x) = 0.3$$

**Policy after some PPO steps:**

$$\pi_\theta(A|x) = 0.9, \quad \pi_\theta(B|x) = 0.1$$

**Step 1: Compute KL**

$$\mathbb{KL}[\pi_\theta \| \pi_{\text{ref}}] = 0.9 \cdot \log\frac{0.9}{0.7} + 0.1 \cdot \log\frac{0.1}{0.3}$$

$$= 0.9 \cdot \log(1.2857) + 0.1 \cdot \log(0.3333)$$

$$= 0.9 \times 0.2513 + 0.1 \times (-1.0986)$$

$$= 0.2262 - 0.1099 = 0.1163 \text{ nats}$$

**Step 2: Apply beta penalty ($\beta = 0.05$):**

$$\text{KL penalty} = 0.05 \times 0.1163 = 0.0058$$

**Step 3: Composite reward** (suppose reward model gives $r_\phi = 0.80$):

$$r_{\text{composite}} = 0.80 - 0.0058 = 0.7942$$

**Verification: KL = 0 when $\pi_\theta = \pi_{\text{ref}}$**

If instead $\pi_\theta(A) = 0.7$, $\pi_\theta(B) = 0.3$:

$$\mathbb{KL} = 0.7 \cdot \log(1) + 0.3 \cdot \log(1) = 0$$

The penalty disappears — the policy is identical to the reference, so no cost is incurred.

**Verification in Python:**

```python
import math

pi_theta = [0.9, 0.1]
pi_ref   = [0.7, 0.3]

kl = sum(p * math.log(p / q) for p, q in zip(pi_theta, pi_ref))
print(f"KL = {kl:.4f} nats")   # KL = 0.1163 nats

beta = 0.05
rm_score = 0.80
composite = rm_score - beta * kl
print(f"Composite reward = {composite:.4f}")  # 0.7942
```

**Intuition:** The policy has shifted probability mass from B to A (learned that A is more rewarding). The KL of 0.1163 nats is modest — a small but non-zero penalty. If the policy had shifted further (e.g., $\pi_\theta(A) = 0.999$), the KL would be much larger, strongly penalising the extreme drift even if the reward model scores it highly.

---

## 9 · Interview drill — follow-up questions

<details>
<summary><b>Q: What is Goodhart's Law and how does it apply to RLHF?</b></summary>

Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure." In RLHF, the reward model is a measure of human preference. When the policy optimises purely against this measure (no KL constraint), it discovers the reward model's blind spots and exploits them — producing outputs that score highly according to the RM but are poor by actual human judgment. The RM score has become the target and has stopped being a reliable measure of quality. Examples: a model learns that longer responses score higher (length bias), so it pads outputs with filler. Or it learns that sycophantic openers ("Great question!") score higher, so it prefixes every response with them.

</details>

<details>
<summary><b>Q: Why use KL divergence specifically, rather than L2 distance on weights or on outputs?</b></summary>

KL divergence on output distributions is the natural choice because: (1) it is invariant to parameterisation — two different parameter configurations that produce the same output distribution have zero KL. Weight L2 distance is parameterisation-dependent and does not reflect distributional similarity. (2) KL is information-theoretically motivated: it measures the information cost of the policy deviating from the reference. (3) At the token level, the KL decomposes into a per-token log-ratio, which is cheap to compute alongside the policy's log-probs during generation. You already have $\log \pi_\theta(y_t)$ from the forward pass; you just need $\log \pi_{\text{ref}}(y_t)$ from a forward pass through the frozen reference model.

</details>

<details>
<summary><b>Q: What are the common failure modes of reward hacking when KL is removed?</b></summary>

Documented failure modes from RLHF practice and the literature:

1. **Length exploitation:** Many reward models assign higher scores to longer responses (annotators associate verbosity with thoroughness). Without KL, policies learn to pad responses — InstructGPT explicitly notes controlling for length.
2. **Repetition:** A policy can achieve high reward by repeating the most positively-scored phrases: "The answer is correct. The answer is correct. The answer is correct..." This fools sentiment-like features in the RM.
3. **Sycophancy:** Prefacing every answer with "Great question! I completely agree..." triggers positive reward model signals trained on agreeable human responses.
4. **Format gaming:** Adding bullet points, bold headers, and confident-sounding language regardless of content quality exploits formatting biases in the RM.
5. **Gibberish at high temperature:** If the reward model is not robust to out-of-distribution text, random token strings can occasionally receive high scores; without KL, the policy will eventually discover these.

Gao, Schulman, and Hilton (2023) quantify this: reward model score and gold-standard human preference diverge after only a few hundred KL units of optimisation.

</details>

<details>
<summary><b>Q: Does removing KL always hurt? Are there cases where it's fine?</b></summary>

Removing KL can be acceptable in narrow, well-defined tasks where: (1) the reward function is a ground-truth oracle (not a learned model), so there are no blind spots to exploit — e.g., code execution pass/fail, mathematical correctness via a verifier. (2) The action space is small and the policy cannot produce degenerate outputs — e.g., classification tasks. In contrast, for open-ended language generation with a learned reward model, removing KL almost always leads to hacking. The degree of hacking depends on how expressive the policy is and how many gradient steps are taken.

</details>

<details>
<summary><b>Q: How does the KL penalty relate to the optimal policy in RLHF?</b></summary>

With the KL-regularised objective, the optimal policy has a closed-form expression (Ziegler et al., 2019; Stiennon et al., 2020):

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{r_\phi(x,y)}{\beta}\right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r_\phi(x,y)/\beta)$ is a normalisation constant. This is the **soft-Q optimal policy** — the reference distribution reweighted by the exponentiated reward, softened by $1/\beta$. As $\beta \to 0$, $\pi^* \to$ the greedy argmax of $r_\phi$. As $\beta \to \infty$, $\pi^* \to \pi_{\text{ref}}$. DPO (Rafailov et al., 2023) derives its objective directly from this closed form, which is why DPO implicitly implements KL regularisation without a separate penalty term.

</details>

<details>
<summary><b>Q: How do you choose beta in practice?</b></summary>

$\beta$ is treated as a hyperparameter and tuned empirically. Common strategies: (1) **Monitor KL during training** — if mean KL grows rapidly (e.g., exceeds 10–20 nats), increase $\beta$ or decrease learning rate. (2) **Grid search on a small validation run** — try $\beta \in \{0.01, 0.02, 0.05, 0.1\}$ for 1–2% of the full compute budget. (3) **Sweep KL vs. win-rate** — plot reward model score gain and KL divergence growth jointly; look for the regime where reward improves but KL is bounded. InstructGPT used $\beta \approx 0.02$. Stiennon et al. (2020) used a mixing coefficient equivalent to $\beta \approx 0.05$.

</details>

---

## 10 · Common misconceptions

| Misconception | Reality |
|---|---|
| "The KL penalty makes the policy converge faster." | KL has no direct effect on convergence speed. It constrains *where* the policy can go, not *how fast* it gets there. |
| "KL = 0 means the policy is not learning anything." | KL = 0 means the policy distribution is identical to the reference. The policy could be improving by *reranking within* the SFT distribution — if the SFT model sometimes produces bad outputs, PPO can reduce their probability while keeping KL small. |
| "Higher beta always means better outputs." | Higher $\beta$ means more conservative updates. If the SFT model is poor (low quality), a high $\beta$ will prevent meaningful improvement. There is a tradeoff. |
| "The KL is computed on the prompt tokens too." | No. The KL is computed only on the response tokens $y$ that the policy generates. The prompt $x$ is fixed; there is no policy distribution over prompts. |
| "Removing KL is fine if you use early stopping." | Early stopping can mitigate but not eliminate reward hacking. The policy can hack the reward model within just a few hundred gradient steps — faster than typical training monitoring intervals. KL is the right structural solution. |
| "DPO doesn't have a KL penalty." | DPO implicitly encodes the same KL constraint via its closed-form derivation from the soft-Q optimal policy. The reference model appears in the DPO loss denominator and plays the exact same regularisation role as the KL term in PPO. |

---

## 11 · Connections to other concepts

**Reward hacking and Goodhart's Law (Q4-11):** The KL penalty is the primary defence against reward hacking. Q4-11 covers the full taxonomy of hacking modes and mitigation strategies including reward model ensembles, synthetic adversarial data, and gold-standard periodic evaluation.

**DPO (Q4-09):** DPO eliminates the reward model and the PPO loop entirely, but it still has an implicit KL term. The DPO loss is derived from the same KL-regularised RLHF objective. The reference model in DPO is the SFT model, playing exactly the same role as $\pi_{\text{ref}}$ in PPO.

**The optimal policy and soft-Q form (Q4-19):** The KL-regularised RLHF objective admits a closed-form optimal policy: $\pi^* \propto \pi_{\text{ref}} \cdot \exp(r/\beta)$. This is the starting point for deriving DPO and for connecting RLHF to the broader framework of entropy-regularised RL.

**PPO clipping (Q4-08):** PPO has two mechanisms to prevent large policy updates: the clipping ratio in the surrogate objective (constrains *within* a single gradient step) and the KL penalty (constrains *across* the full training run). They are complementary. Clipping is step-level; KL is trajectory-level.

**Reward model overoptimisation (Q4-31):** Gao, Schulman, and Hilton (2023) study how reward model score diverges from human preference as a function of KL divergence from the reference. Their key finding: the gap opens up after roughly 5–30 KL units of optimisation, depending on reward model quality and size. This gives an empirical justification for the typical $\beta$ range.

**Constitutional AI (Q4-12, Q4-27):** Anthropic's CAI approach uses RLAIF (AI feedback instead of human feedback) and the same KL-regularised PPO objective. The KL penalty plays the same role regardless of whether the reward signal comes from humans or an AI critic.

---

## 12 · One-screen summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RLHF KL PENALTY — ONE-SCREEN SUMMARY                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  OBJECTIVE: J(θ) = E[ r_φ(x,y)  −  β · KL[π_θ ‖ π_ref] ]                     │
│                          ↑                      ↑                               │
│                    reward signal          KL penalty                            │
│                   (push toward           (pull back toward                      │
│                   high reward)            SFT reference)                        │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHAT KL DOES                                                                   │
│  • Prevents policy drifting into reward model blind spots                       │
│  • Keeps outputs in the natural language distribution                           │
│  • Controls the tradeoff via β (typical range: 0.01 – 0.1)                     │
│  • Ensures the SFT model's language quality is preserved                        │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WITHOUT KL (β = 0)                                                             │
│  • Reward hacking: policy exploits RM blind spots                               │
│  • Failure modes: repetition, sycophancy, length exploitation, gibberish        │
│  • Goodhart's Law: the measure becomes the target and stops being a good measure│
│  • KL → ∞, outputs become incoherent                                            │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  TOKEN-LEVEL FORM         KL ≈ Σ_t log( π_θ(y_t|x,y<t) / π_ref(y_t|x,y<t) )  │
│  OPTIMAL POLICY           π*(y|x) ∝ π_ref(y|x) · exp( r(x,y)/β )              │
│  β → 0 → greedy reward    β → ∞ → identical to SFT reference                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 13 · Five-minute refresher

**Core question:** Why not just maximise reward?

Because the reward model is imperfect. It is trained on finite human data and has blind spots — regions where it gives high scores to bad outputs. A policy trained purely on reward discovers these blind spots rapidly (within hundreds of gradient steps) and exploits them. Outputs become repetitive, sycophantic, or incoherent. The reward score climbs but actual quality collapses.

**The solution:** Add a KL penalty. The objective becomes:

$$\mathcal{J}(\theta) = \mathbb{E}\!\left[ r_\phi(x,y) - \beta \cdot \mathbb{KL}[\pi_\theta \| \pi_{\text{ref}}] \right]$$

The KL term measures how far the policy has drifted from the SFT reference model. It starts at zero (policy = SFT model) and grows as the policy moves away. The coefficient $\beta$ controls the tradeoff.

**At token level:** $\mathbb{KL} \approx \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})}$

**Practical $\beta$ values:** 0.01–0.1. InstructGPT used ~0.02. Monitor mean KL during training; if it exceeds ~10–20 nats, increase $\beta$ or reduce learning rate.

**Quick sanity check:** KL = 0 means $\pi_\theta = \pi_{\text{ref}}$ — the policy has not changed. This is the starting point of PPO. As training progresses, KL grows from 0 toward a bounded steady state when $\beta$ is set correctly.

**What about DPO?** DPO has no explicit KL term but it is baked in. DPO is derived from the same KL-regularised objective; the reference model in the DPO loss plays the same role as $\pi_{\text{ref}}$ in the KL penalty.

---

## 14 · Further reading

- **InstructGPT paper (Ouyang et al., 2022):** The canonical applied RLHF paper. Section 3.4 discusses the KL penalty coefficient and its practical role. Reports $\beta \approx 0.02$.
- **Ziegler et al. (2019):** Introduced the KL-regularised RL objective for LLMs. Section 2.1 derives the token-level approximation.
- **Gao, Schulman, Hilton (2023):** Empirical scaling laws for reward model overoptimisation. Shows the KL budget at which RM score and human preference diverge. Essential for understanding how to set $\beta$.
- **Stiennon et al. (2020):** Applied RLHF to summarisation; quantitative analysis of KL growth and reward hacking in practice.
- **Rafailov et al. (2023) DPO paper:** Derives the optimal policy from the KL-regularised objective and shows how this leads to the DPO loss. Section 3 is the key derivation.

---

## 15 · References

1. Ouyang, L., Wu, J., Jiang, X., et al. — **Training language models to follow instructions with human feedback** (InstructGPT). *NeurIPS 2022 / arXiv:2203.02155.* — Describes the full RLHF pipeline including the KL penalty coefficient; reports $\beta \approx 0.02$.

2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. — **Proximal Policy Optimization Algorithms**. *arXiv:1707.06347, 2017.* — The PPO algorithm used in RLHF. The clipping mechanism and the KL penalty are complementary stabilisers.

3. Ziegler, D. M., Stiennon, N., Wu, J., et al. — **Fine-Tuning Language Models from Human Preferences**. *arXiv:1909.08593, 2019.* — First application of KL-regularised RL to language models; Section 2.1 derives the token-level KL approximation.

4. Gao, L., Schulman, J., Hilton, J. — **Scaling Laws for Reward Model Overoptimization**. *ICML 2023 / arXiv:2210.10760.* — Empirically measures the divergence between RM score and gold human preference as a function of KL. Establishes the "KL budget" concept.

5. Stiennon, N., Ouyang, L., Wu, J., et al. — **Learning to summarize from human feedback**. *NeurIPS 2020 / arXiv:2009.01325.* — Applied RLHF to summarisation; quantitative analysis of KL growth and reward hacking; an early demonstration of the practical importance of the KL constraint.

6. Rafailov, R., Sharma, A., Mitchell, E., et al. — **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**. *NeurIPS 2023 / arXiv:2305.18290.* — Derives DPO from the same KL-regularised RLHF objective; shows the optimal policy in closed form (the soft-Q form).

---

<!-- ============================ BOTTOM NAV ============================ -->
<div align="center">

[⬅️ Q4‑03 — Reward Model & Bradley-Terry](./q03-reward-model-bradley-terry.md) &nbsp;|&nbsp; [📚 Back to Section 4](./README.md) &nbsp;|&nbsp; [🏠 Home](../../README.md) &nbsp;|&nbsp; [Q4‑05 — On-Policy vs. Off-Policy RL ➡️](./q05-on-policy-vs-off-policy.md)

<sub>Found an error or have a sharper intuition? See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a> — answers follow the <a href="../../_TEMPLATE.md">answer template</a>.</sub>

</div>
