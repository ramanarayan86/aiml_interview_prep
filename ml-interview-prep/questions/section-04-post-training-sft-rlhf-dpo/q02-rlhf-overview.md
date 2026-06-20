<!-- ============================ TOP NAV ============================ -->
<div align="center">

[🏠 Home](../../README.md) &nbsp;•&nbsp; [📚 Section 4 — Post-training](./README.md) &nbsp;•&nbsp; [⬅️ Q4‑01 — SFT & LIMA](./q01-sft-lima.md) &nbsp;•&nbsp; [Q4‑03 — Reward Model ➡️](./q03-reward-model.md)

</div>

---

# Q4‑02 · Explain RLHF at a high level. What are the three stages (SFT, RM, PPO)?

<div align="center">

![Section](https://img.shields.io/badge/Section_4-Post--training%3A_SFT%2C_RLHF%2C_DPO_and_Beyond-1f4fa3)
![Level](https://img.shields.io/badge/Level-Basic-0e8a6e)
![Tags](https://img.shields.io/badge/tags-RLHF·reward--model·PPO·three--stage-1f4fa3)
![Read](https://img.shields.io/badge/read_time-~20_min-5b6675)

</div>

> [!IMPORTANT]
> **The 20-second answer.** RLHF (Reinforcement Learning from Human Feedback) converts a raw pretrained language model into a helpful, harmless assistant by running three sequential stages. **Stage 1 — SFT:** fine-tune the base model on a small set of high-quality, human-written instruction-response pairs using standard cross-entropy; this sets a strong behavioral prior and initialises the policy. **Stage 2 — Reward Model (RM):** collect human preference judgements over pairs of model-generated responses, then train a scalar-valued reward model on those pairs using the Bradley-Terry log-likelihood loss; the RM learns to score responses the way humans would. **Stage 3 — PPO:** optimise the SFT model (the policy) to maximise the RM score via Proximal Policy Optimization, while subtracting a KL-divergence penalty against the frozen SFT reference model to prevent reward hacking and mode collapse. The result is a policy that generates responses humans prefer without deviating catastrophically from the pretrained distribution.

---

## Table of contents

1. [First principles](#1--first-principles)
2. [The core mechanism — three stages in detail](#2--the-core-mechanism--three-stages-in-detail)
3. [Figure 1 — RLHF three-stage pipeline](#3--figure-1--rlhf-three-stage-pipeline)
4. [Step-by-step worked example](#4--step-by-step-worked-example)
5. [Figure 2 — PPO reward signal flow](#5--figure-2--ppo-reward-signal-flow)
6. [Algorithm / pseudocode](#6--algorithm--pseudocode)
7. [PyTorch reference implementation](#7--pytorch-reference-implementation)
8. [Worked numerical example](#8--worked-numerical-example)
9. [Interview drill — follow-up questions](#9--interview-drill--follow-up-questions)
10. [Common misconceptions](#10--common-misconceptions)
11. [Connections to other concepts](#11--connections-to-other-concepts)
12. [One-screen summary](#12--one-screen-summary)
13. [Five-minute refresher](#13--five-minute-refresher)
14. [Further reading](#14--further-reading)
15. [Bottom navigation](#15--bottom-navigation)

---

## 1 · First principles

A pretrained language model learns to predict the next token on internet text. That objective does not guarantee the model will be helpful, honest, or harmless — it merely makes the model a good distribution estimator over human-written text, which includes unhelpful, deceptive, and harmful content.

The fundamental problem RLHF solves is: **how do we specify and optimise for what humans actually want from a language model, when that preference is complex, multi-dimensional, and difficult to encode in a simple loss function?**

The answer is to use humans themselves as the reward signal. Rather than hand-engineering a reward function, RLHF:

1. Bootstraps the model with supervised imitation of good behavior (SFT).
2. Trains a proxy reward function by having humans compare pairs of model outputs.
3. Uses reinforcement learning to optimise the model policy against that proxy reward, with a regularisation term to prevent exploiting the proxy.

This approach, introduced at scale by Christiano et al. (2017) for Atari and robotics tasks, was applied to language models by Stiennon et al. (2020) for summarisation and canonised in InstructGPT (Ouyang et al., 2022) for general instruction following.

**Why not just do more SFT?** SFT on demonstrations teaches the model to imitate the surface of human-written answers. It does not teach the model to rank among multiple possible answers, to recognise what makes a response better or worse, or to generalise to novel prompts. Preference learning captures a richer signal: a human annotator comparing two responses reveals information about the reward function even when neither response is perfect.

---

## 2 · The core mechanism — three stages in detail

### Stage 1 · Supervised Fine-Tuning (SFT)

**Goal:** Produce a policy $\pi_0$ that generates plausible instruction-following responses.

**Data:** A small, carefully curated dataset of (prompt $x$, response $y$) pairs written or selected by human labellers. InstructGPT used roughly 13K demonstrations; quality matters far more than quantity here (see the LIMA hypothesis).

**Training objective:** Standard language model cross-entropy over the response tokens:

$$\mathcal{L}_\text{SFT} = -\sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})$$

**Output:** The SFT model $\pi_0$, which will serve as both the initial policy for PPO and the frozen reference policy throughout Stage 3.

**Why it is necessary:** Starting PPO directly from a pretrained base model is extremely unstable — the policy generates incoherent responses, which the reward model cannot reliably rank, producing noisy gradients. SFT provides a strong behavioral prior.

---

### Stage 2 · Reward Model (RM) Training

**Goal:** Train a scalar-valued function $r_\phi(x, y)$ that scores how much a human would prefer response $y$ to prompt $x$.

**Data collection:** For each prompt $x$, the SFT model generates $k$ responses $\{y_1, \ldots, y_k\}$ (typically $k = 4$–$9$). Human annotators rank or compare pairs, producing a preference dataset $\mathcal{D} = \{(x, y_w, y_l)\}$ where $y_w$ is preferred over $y_l$.

**Bradley-Terry model:** The probability that a human prefers $y_w$ over $y_l$ is modelled as:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)$$

where $\sigma$ is the sigmoid function. This is the **Bradley-Terry pairwise comparison model**.

**Training loss:** Maximise the log-likelihood of observed preferences:

$$\mathcal{L}_\text{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

**Architecture:** The RM is typically the SFT model with the final unembedding head replaced by a linear layer mapping to a scalar. This reuses the language model's internal representations.

**Output:** A trained reward model $r_\phi$ that assigns higher scalars to responses humans prefer.

---

### Stage 3 · PPO (Proximal Policy Optimization)

**Goal:** Optimise the language model policy $\pi_\theta$ to maximise expected reward $r_\phi(x, y)$ while staying close to the reference policy $\pi_0$.

**The full per-token reward signal at generation step $t$:**

$$\tilde{r}(x, y) = r_\phi(x, y) - \beta \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_0(y_t \mid x, y_{<t})}$$

The second term is the per-token KL divergence between the current policy and the reference. Coefficient $\beta$ (typically 0.02–0.2) controls how much drift is penalised.

**The RLHF objective:**

$$\max_{\theta} \mathbb{E}_{x \sim \mathcal{P},\, y \sim \pi_\theta(\cdot|x)}\!\left[r_\phi(x, y) - \beta \,\mathrm{KL}\!\left(\pi_\theta(\cdot|x) \,\|\, \pi_0(\cdot|x)\right)\right]$$

**PPO update rule:** PPO is an actor-critic method that limits the size of each policy update using a clipped surrogate objective:

$$\mathcal{L}^\text{PPO}(\theta) = \mathbb{E}\!\left[\min\!\left(r_t(\theta)\,\hat{A}_t,\; \mathrm{clip}(r_t(\theta), 1{-}\varepsilon, 1{+}\varepsilon)\,\hat{A}_t\right)\right]$$

where $r_t(\theta) = \pi_\theta(y_t \mid x, y_{<t}) / \pi_{\theta_\text{old}}(y_t \mid x, y_{<t})$ is the probability ratio and $\hat{A}_t$ is the advantage estimate (KL-adjusted reward minus a value-function baseline).

**Why PPO and not simpler RL?** Language generation is a long-horizon discrete action space (vocabulary × sequence length). PPO's clipping prevents large destabilising updates, which is critical when the reward landscape is sparse and noisy.

---

## 3 · Figure 1 — RLHF three-stage pipeline

<div align="center">
<img src="../../assets/s4q02/fig1-rlhf-three-stages.svg" alt="RLHF three-stage pipeline. Stage 1 (navy): Base LM plus instruction data feeds into SFT model (pi-0 policy). Stage 2 (teal): SFT generates k response pairs, human raters provide preference labels, Reward Model trained via Bradley-Terry log-likelihood. Stage 3 (amber): Policy pi-theta initialised from SFT, frozen reference policy pi-0, PPO updates policy using reward r-phi minus beta KL penalty; feedback loop drives iterative updates. Output: RLHF-aligned LM that is helpful, harmless, honest." width="92%">
<br><sub><b>Figure 1.</b> The RLHF three-stage pipeline. Stage 1 (navy) performs supervised fine-tuning to initialise the policy. Stage 2 (teal) collects human preferences over SFT outputs and trains a scalar reward model via Bradley-Terry log-likelihood. Stage 3 (amber) runs PPO: the policy generates responses, the reward model scores them, a KL penalty against the frozen reference is subtracted, and the policy is updated. The loop iterates until the policy converges or a fixed number of PPO steps is reached.</sub>
</div>

---

## 4 · Step-by-step worked example

To make the three stages concrete, trace through a single instruction: **"Summarise the paper on attention in two sentences."**

**Stage 1 — SFT:**
- Dataset contains 50 human-written (prompt, summary) pairs.
- The base model is fine-tuned until it produces coherent two-sentence summaries.
- Output: $\pi_0$ that writes plausible summaries.

**Stage 2 — RM:**
- For the prompt above, $\pi_0$ generates 4 candidate summaries $y_1, y_2, y_3, y_4$.
- Human annotator ranks: $y_2 \succ y_4 \succ y_1 \succ y_3$ (prefers $y_2$ most, $y_3$ least).
- This produces $\binom{4}{2} = 6$ preference pairs: $(y_2, y_4), (y_2, y_1), (y_2, y_3), (y_4, y_1), (y_4, y_3), (y_1, y_3)$.
- The RM is trained on these pairs via Bradley-Terry loss.
- After training: $r_\phi(x, y_2) = 2.5$, $r_\phi(x, y_4) = 1.9$, $r_\phi(x, y_1) = 1.1$, $r_\phi(x, y_3) = 0.3$.

**Stage 3 — PPO:**
- Policy $\pi_\theta$ (initialised from $\pi_0$) generates a new summary $y$ for the prompt.
- $r_\phi(x, y) = 1.8$ (raw RM score).
- KL divergence between $\pi_\theta$ and $\pi_0$ on this response: $\mathrm{KL} = 0.023$.
- With $\beta = 0.1$: adjusted reward $\tilde{r} = 1.8 - 0.1 \times 0.023 = 1.798$.
- Value baseline $V(x) = 1.2$; advantage $\hat{A} = 1.798 - 1.2 = 0.598$.
- PPO updates $\theta$ to increase the probability of this response, clipped to at most $\varepsilon = 0.2$ change in probability ratio.

After many such iterations, the policy learns to generate summaries that score highly on $r_\phi$ without drifting too far from the original SFT behavior.

---

## 5 · Figure 2 — PPO reward signal flow

<div align="center">
<img src="../../assets/s4q02/fig2-rlhf-reward-signal-flow.svg" alt="PPO loop detail. Six steps connected left to right: (1) sample prompt x from dataset; (2) policy pi-theta generates response y; (3) reward model scores y giving r-phi(x,y); (4) KL penalty branch subtracts beta KL(pi-theta || pi-0) using the frozen reference policy pi-0; (5) advantage A-hat = adjusted-reward minus value-baseline V(x); (6) PPO clipped policy gradient update. A feedback loop returns updated theta back to the policy for the next rollout. The PPO clipped objective equation is annotated with the KL penalty term highlighted in red." width="92%">
<br><sub><b>Figure 2.</b> Detailed PPO reward signal flow. The policy generates a response (step 2), the reward model scores it (step 3), and the KL penalty branch (step 4, red) subtracts $\beta \log(\pi_\theta / \pi_0)$ using the frozen reference policy. The advantage estimate (step 5, purple) subtracts a value-function baseline. The PPO clipped surrogate (step 6, amber) updates the policy weights, and the loop feeds new weights back for the next rollout batch.</sub>
</div>

---

## 6 · Algorithm / pseudocode

```text
===== RLHF THREE-STAGE TRAINING =====

INPUT:
  base_model    -- pretrained LM
  D_sft         -- labeled (prompt, response) pairs
  D_pref        -- human preference pairs (prompt, y_chosen, y_rejected)
  beta          -- KL penalty coefficient (e.g. 0.1)
  epsilon       -- PPO clip ratio (e.g. 0.2)
  N_ppo         -- number of PPO iterations

=== STAGE 1: SFT ===
  pi_0 = copy(base_model)
  for each (x, y) in D_sft:
      loss = -sum_t log pi_0(y_t | x, y_{<t})   # cross-entropy
      pi_0 = pi_0 - lr * grad(loss)
  # pi_0 is now the SFT model and frozen reference

=== STAGE 2: REWARD MODEL ===
  r_phi = SFT_head_replaced_with_scalar(pi_0)
  for each (x, y_w, y_l) in D_pref:
      reward_chosen   = r_phi(x, y_w)
      reward_rejected = r_phi(x, y_l)
      loss = -log( sigmoid(reward_chosen - reward_rejected) )
      r_phi = r_phi - lr * grad(loss)
  # r_phi is now the trained reward model

=== STAGE 3: PPO ===
  pi_theta = copy(pi_0)          # policy init from SFT
  pi_ref   = copy(pi_0)          # frozen reference
  freeze(pi_ref)

  for iteration = 1 to N_ppo:

    # --- Rollout phase ---
    for each prompt x in rollout_batch:
        y = sample(pi_theta, x)             # generate response
        r_raw = r_phi(x, y)                  # reward model score
        kl    = sum_t log( pi_theta(y_t|x,y_<t)
                         / pi_ref(y_t|x,y_<t) )
        r_adjusted = r_raw - beta * kl       # total reward

    # --- Value and advantage estimation ---
    V_x = value_head(x)                      # baseline from critic
    A_hat = r_adjusted - V_x                 # advantage

    # --- PPO update phase ---
    pi_old = copy(pi_theta)                  # snapshot old policy
    for each (x, y, A_hat) in rollout_batch:
        for each token t:
            ratio = pi_theta(y_t|x,y_<t) / pi_old(y_t|x,y_<t)
            clipped = clip(ratio, 1-epsilon, 1+epsilon)
            surrogate = min(ratio * A_hat[t], clipped * A_hat[t])
        loss_ppo = -mean(surrogate)           # maximise (ascent)
        loss_vf  = mean( (r_adjusted - V_x)**2 )  # value function
        loss     = loss_ppo + c_vf * loss_vf
        pi_theta = pi_theta - lr * grad(loss)

OUTPUT: pi_theta  -- the RLHF-aligned policy
```

---

## 7 · PyTorch reference implementation

```python
"""
RLHF core components: RM loss and PPO loss with KL penalty.
Illustrates the key loss computations; omits distributed training scaffolding.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Stage 2: Reward Model Training Loss ───────────────────────────────────────

class RewardModel(nn.Module):
    """Wraps a language model backbone with a scalar reward head."""

    def __init__(self, backbone: nn.Module, hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns scalar reward for each sequence.  Shape: (batch,)"""
        hidden = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        # Pool at the last non-padding token (EOS position)
        last_idx = attention_mask.sum(dim=1) - 1            # (batch,)
        last_hidden = hidden[torch.arange(hidden.size(0)), last_idx]  # (batch, hidden)
        return self.reward_head(last_hidden).squeeze(-1)    # (batch,)


def bradley_terry_loss(
    reward_model: RewardModel,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Bradley-Terry pairwise preference loss.

    L = -E[ log sigma( r(x, y_chosen) - r(x, y_rejected) ) ]

    Args:
        chosen_ids / chosen_mask:   tokenised chosen responses, shape (B, T).
        rejected_ids / rejected_mask: tokenised rejected responses, shape (B, T).

    Returns:
        Scalar loss to minimise.
    """
    r_chosen   = reward_model(chosen_ids,   chosen_mask)    # (B,)
    r_rejected = reward_model(rejected_ids, rejected_mask)  # (B,)

    # log sigma(r_w - r_l) = -log(1 + exp(-(r_w - r_l)))
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss


# ── Stage 3: PPO Loss with KL Penalty ─────────────────────────────────────────

def compute_kl_penalty(
    log_probs_policy: torch.Tensor,   # (B, T)
    log_probs_ref:    torch.Tensor,   # (B, T)
    attention_mask:   torch.Tensor,   # (B, T)
) -> torch.Tensor:
    """
    Per-sequence KL divergence: KL(pi_theta || pi_ref).

    KL = sum_t [ pi_theta(y_t|.) * (log pi_theta - log pi_ref) ]

    When using per-token log probs (not full distributions), this reduces to:
        KL ≈ sum_t (log pi_theta(y_t) - log pi_ref(y_t))

    This is the reverse-KL sample approximation used in practice.
    Returns shape: (B,)
    """
    kl_per_token = log_probs_policy - log_probs_ref          # (B, T)
    kl_per_token = kl_per_token * attention_mask             # mask padding
    return kl_per_token.sum(dim=1)                           # (B,)


def ppo_loss(
    log_probs_policy:  torch.Tensor,   # (B, T)  current policy
    log_probs_old:     torch.Tensor,   # (B, T)  old policy (rollout snapshot)
    log_probs_ref:     torch.Tensor,   # (B, T)  frozen SFT reference
    rewards:           torch.Tensor,   # (B,)    reward model scores
    values:            torch.Tensor,   # (B,)    value function estimates
    attention_mask:    torch.Tensor,   # (B, T)
    beta:              float = 0.1,    # KL penalty coefficient
    epsilon:           float = 0.2,    # PPO clip ratio
    value_coef:        float = 0.5,    # value function loss coefficient
) -> dict[str, torch.Tensor]:
    """
    PPO objective with KL penalty:

        L = -E[ min(r_t(theta) * A_hat,
                    clip(r_t(theta), 1-eps, 1+eps) * A_hat) ]
            + beta * KL(pi_theta || pi_ref)

    Args:
        log_probs_policy: log pi_theta(y_t | x, y_<t)
        log_probs_old:    log pi_old(y_t | x, y_<t)  (from rollout phase)
        log_probs_ref:    log pi_ref(y_t | x, y_<t)  (frozen SFT reference)
        rewards:          scalar reward from RM, shape (B,)
        values:           critic estimate V(x), shape (B,)
        attention_mask:   1 for real tokens, 0 for padding

    Returns:
        dict with keys: 'loss', 'policy_loss', 'value_loss', 'kl'
    """
    # ── KL penalty ──────────────────────────────────────────────────────────
    kl = compute_kl_penalty(log_probs_policy, log_probs_ref, attention_mask)  # (B,)

    # ── Adjusted reward (total return) ──────────────────────────────────────
    r_adjusted = rewards - beta * kl                          # (B,)

    # ── Advantage estimate (simple; in practice use GAE) ────────────────────
    advantages = (r_adjusted - values).detach()               # (B,)

    # Broadcast advantage from (B,) to (B, T) for per-token update
    T = log_probs_policy.size(1)
    adv_expanded = advantages.unsqueeze(1).expand(-1, T)      # (B, T)

    # ── PPO clipped surrogate ────────────────────────────────────────────────
    # probability ratio r_t(theta) = pi_theta / pi_old
    ratio = torch.exp(log_probs_policy - log_probs_old)       # (B, T)

    surr1 = ratio * adv_expanded
    surr2 = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * adv_expanded
    policy_loss = -torch.min(surr1, surr2)                    # (B, T)

    # Mask padding and average
    policy_loss = (policy_loss * attention_mask).sum() / attention_mask.sum()

    # ── Value function loss ──────────────────────────────────────────────────
    value_loss = F.mse_loss(values, r_adjusted.detach())

    # ── Total loss ───────────────────────────────────────────────────────────
    total_loss = policy_loss + value_coef * value_loss

    return {
        "loss":         total_loss,
        "policy_loss":  policy_loss,
        "value_loss":   value_loss,
        "kl":           kl.mean(),
        "r_adjusted":   r_adjusted.mean(),
    }


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T = 4, 20

    # Simulate log-probs as small negative values
    log_probs_policy = torch.randn(B, T) - 3.0
    log_probs_old    = log_probs_policy + 0.05 * torch.randn(B, T)  # slight drift
    log_probs_ref    = log_probs_policy + 0.10 * torch.randn(B, T)  # more drift
    rewards          = torch.tensor([1.8, 0.9, 2.1, 1.3])
    values           = torch.tensor([1.2, 1.0, 1.5, 1.1])
    mask             = torch.ones(B, T)
    mask[:, -5:] = 0  # last 5 tokens are padding

    result = ppo_loss(log_probs_policy, log_probs_old, log_probs_ref,
                      rewards, values, mask, beta=0.1, epsilon=0.2)

    print("PPO loss components:")
    for k, v in result.items():
        print(f"  {k:15s}: {v.item():.4f}")
```

---

## 8 · Worked numerical example

This section traces through concrete numbers for all three stages.

### Stage 2 — RM training on three preference pairs

**Setup:** The reward model produces scalar scores. We observe three preference pairs from human annotators:

| Pair | $r_\phi(y_w)$ | $r_\phi(y_l)$ | $\Delta r$ | $P(y_w \succ y_l)$ | BT loss |
|------|--------------|--------------|----------|------------------|---------|
| 1    | 2.5          | 0.8          | 1.7      | $\sigma(1.7) = 0.8455$ | $-\ln(0.8455) = 0.1678$ |
| 2    | 1.9          | 1.1          | 0.8      | $\sigma(0.8) = 0.6900$ | $-\ln(0.6900) = 0.3711$ |
| 3    | 3.1          | 0.5          | 2.6      | $\sigma(2.6) = 0.9309$ | $-\ln(0.9309) = 0.0716$ |

**Mean BT loss:** $(0.1678 + 0.3711 + 0.0716) / 3 = 0.2035$

The gradient of the loss w.r.t. $r_\phi(y_w)$ for pair 1:

$$\frac{\partial \mathcal{L}}{\partial r_\phi(y_w)} = -(1 - P(y_w \succ y_l)) = -(1 - 0.8455) = -0.1545$$

This pushes $r_\phi(y_w)$ upward, increasing the gap and making the model more confident in the correct preference.

---

### Stage 3 — PPO reward signal at a single step

**Inputs:**
- Raw reward model score: $r_\phi(x, y) = 1.8$
- Policy log-prob sum over response tokens: $\log \pi_\theta(y|x) = -3.147$
- Reference log-prob sum over same tokens: $\log \pi_0(y|x) = -3.270$
- KL penalty coefficient: $\beta = 0.1$

**Step 1 — Compute per-sequence KL divergence:**

The per-sequence KL approximation (summing token-level log-ratio):

$$\mathrm{KL}(\pi_\theta \| \pi_0) \approx \sum_t \left[\log \pi_\theta(y_t|x, y_{<t}) - \log \pi_0(y_t|x, y_{<t})\right]$$

Using the policy and reference logit distributions over the vocabulary at one representative position:

| Token | $\pi_\theta$ | $\pi_0$ | $\pi_\theta \log(\pi_\theta / \pi_0)$ |
|-------|-------------|--------|--------------------------------------|
| "the" | 0.6133      | 0.5089 | $0.6133 \times \ln(1.205) = 0.1143$ |
| "a"   | 0.2494      | 0.3087 | $0.2494 \times \ln(0.808) = -0.0530$ |
| "an"  | 0.0917      | 0.1136 | $0.0917 \times \ln(0.807) = -0.0199$ |
| other | 0.0456      | 0.0689 | $0.0456 \times \ln(0.661) = -0.0188$ |

Sum $= 0.1143 - 0.0530 - 0.0199 - 0.0188 \approx 0.0226$

Aggregating across all tokens in the response: $\mathrm{KL} \approx 0.0229$

**Step 2 — Adjusted reward:**

$$\tilde{r} = r_\phi(x, y) - \beta \cdot \mathrm{KL} = 1.8 - 0.1 \times 0.0229 = 1.7977$$

**Step 3 — Advantage estimate:**

With value baseline $V(x) = 1.2$:

$$\hat{A} = \tilde{r} - V(x) = 1.7977 - 1.2 = 0.5977$$

**Step 4 — PPO probability ratio:**

$$r_t(\theta) = \frac{\pi_\theta(y|x)}{\pi_\text{old}(y|x)} = e^{-3.147 - (-3.270)} = e^{0.123} = 1.131$$

**Step 5 — Clipped surrogate (with $\varepsilon = 0.2$):**

Since $r_t(\theta) = 1.131 < 1 + 0.2 = 1.2$, the ratio is not clipped:

$$\mathcal{L}^\text{PPO} = -\min(1.131 \times 0.5977,\; 1.131 \times 0.5977) = -0.6764$$

(Both terms equal; clipping is inactive when the ratio lies within $[0.8, 1.2]$.)

**Step 6 — Total loss (policy + value):**

With $c_\text{vf} = 0.5$ and value loss $(1.7977 - 1.2)^2 = 0.3572$:

$$\mathcal{L}_\text{total} = -0.6764 + 0.5 \times 0.3572 = -0.4978$$

The optimizer takes a gradient step to minimise this loss, increasing the probability of generating response $y$ for prompt $x$.

---

## 9 · Interview drill — follow-up questions

<details>
<summary><b>Q: Why does RLHF use PPO specifically, and not simpler policy gradient methods like REINFORCE?</b></summary>

REINFORCE uses the raw policy gradient $\nabla_\theta \log \pi_\theta(y|x) \cdot R$, which has high variance (the reward $R$ can vary wildly across rollouts) and no mechanism to limit the size of each update. Large updates in language model training can irreversibly damage the policy — for example, collapsing token diversity or converging to reward-hacking behaviors in a single large step.

PPO addresses this with two mechanisms: (1) **clipping** the probability ratio $r_t(\theta)$ to $[1-\varepsilon, 1+\varepsilon]$, which prevents any single update from changing the policy too drastically; (2) **the KL penalty** (or alternatively, the KL constraint formulation), which penalises drift from the reference. Together, these stabilise training significantly.

REINFORCE also lacks a value function baseline, which PPO's actor-critic architecture provides to reduce variance in advantage estimates.
</details>

<details>
<summary><b>Q: What happens if you remove the KL penalty from Stage 3?</b></summary>

Without the KL penalty, the policy is free to exploit any weakness in the reward model. This is a form of **Goodhart's Law** ("When a measure becomes a target, it ceases to be a good measure"): the policy will find inputs that receive high RM scores without those inputs actually being high-quality.

Common failure modes observed without the KL constraint: (1) **length hacking** — the policy learns that the RM rewards long responses, so it generates extremely verbose, repetitive text; (2) **sycophancy** — the policy learns to agree with any premise in the prompt regardless of truth; (3) **format exploitation** — the policy generates unusual character repetitions or special tokens that happened to correlate with high reward during RM training; (4) **mode collapse** — the policy degenerates into always generating a single high-reward phrase regardless of the prompt.

The KL penalty keeps the policy anchored to the SFT distribution, where these pathological behaviors have low probability.
</details>

<details>
<summary><b>Q: How does the reward model training differ from standard classification fine-tuning?</b></summary>

Standard classification fine-tuning trains on absolute labels (each response is good or bad). The RM trains on relative preferences (response A is better than response B). This distinction matters because:

(1) **Consistency:** Humans are better at comparing two options than assigning absolute quality scores. Relative judgements have lower variance and higher inter-annotator agreement.

(2) **Scalability:** Collecting ranked pairs is faster than generating high-quality labeled examples from scratch.

(3) **The loss:** Cross-entropy classification loss treats each example independently. Bradley-Terry loss explicitly models the joint probability of the pairwise relationship, encoding the comparative nature of the signal.

(4) **No reward scale issue:** Absolute rewards require calibrating the scale of the reward signal, which is hard. Bradley-Terry rewards are only meaningful as differences, so the absolute scale cancels out in the loss.
</details>

<details>
<summary><b>Q: What is the "alignment tax" and how does RLHF affect it?</b></summary>

The alignment tax is the degradation of base model capabilities (e.g., reasoning, knowledge retrieval, code generation) that can occur during post-training. RLHF can contribute to this in several ways: (1) the KL penalty constrains the policy to stay close to the SFT distribution, preventing it from exploring the full capability space; (2) the reward model may not capture performance on capability benchmarks, so PPO inadvertently optimises away from them; (3) the SFT data may not cover the full distribution of capability-testing prompts.

Mitigations include: using capability-aware reward models or multi-objective reward functions; monitoring MMLU/GSM8K/HumanEval throughout RLHF training; applying the KL penalty carefully (lower $\beta$ during early RLHF when the policy has not yet drifted far); and mixing SFT-style supervised loss into the PPO objective.
</details>

<details>
<summary><b>Q: How many human preference annotations did InstructGPT use for each stage?</b></summary>

InstructGPT (Ouyang et al., 2022) used: **Stage 1 SFT** — approximately 13,000 demonstration examples; **Stage 2 RM** — approximately 33,000 human preference comparisons (each comparison covering $k=4$–$9$ responses per prompt, so significantly more candidate responses were generated); **Stage 3 PPO** — approximately 31,000 prompts for the PPO rollout buffer (no additional human annotations; the RM provides the reward signal). The total human annotation budget was modest by modern standards, demonstrating that data quality (prompt diversity, annotator calibration) matters more than raw annotation volume.
</details>

<details>
<summary><b>Q: Can RLHF be applied directly to the base model without SFT? Why or why not?</b></summary>

In principle, yes — but in practice it fails. The base model's outputs are highly variable and often incoherent on instruction-following prompts. When the reward model receives these outputs, its scores are noisy because the distribution is far from the preference data distribution used to train the RM. PPO's gradients are therefore very noisy, and the policy tends to diverge before learning anything useful.

SFT serves as a form of **behavioral cloning** that moves the policy into a region of prompt-response space where the RM provides reliable signal. The KL penalty is also more meaningful when anchored to an SFT policy rather than a raw pretrained model: the SFT model represents "reasonable instruction-following behavior," which is exactly what you want to stay near.

Some recent work (e.g., Reinforcement Learning with Human Feedback from the Base Model) has explored starting from the base model with carefully designed reward shaping, but this is significantly harder to stabilise.
</details>

---

## 10 · Common misconceptions

| Misconception | Reality |
|---|---|
| "RLHF teaches the model new facts." | RLHF shapes behavior — it adjusts which outputs the model prefers to generate, not what knowledge it has. Knowledge comes from pretraining. RLHF can, however, cause the model to stop surfacing knowledge it has (sycophancy, excessive caveats). |
| "The reward model is a binary classifier." | The RM produces a scalar reward (a real number), not a binary label. It is trained on pairwise comparisons via Bradley-Terry, but outputs a continuous score used directly as a reward signal. |
| "The KL penalty goes to zero at convergence." | At convergence, the policy settles at a balance between reward maximisation and KL constraint. Unless $\beta = 0$ or the policy has found the exact reference distribution, the KL term at convergence is typically a small positive constant, not zero. |
| "RLHF requires online human feedback during PPO." | The PPO stage uses a trained reward model as a proxy — no live human feedback is needed. Humans annotate only in Stage 2 to train the RM. This is what makes RLHF scalable. |
| "PPO in RLHF is identical to PPO in games." | The language model setting has key differences: discrete token-level actions (vocabulary-sized), sparse rewards (usually one score per sequence, not per token), extremely long episodes (full response length), and a critical KL regularisation term that is absent in game settings. |
| "Higher reward model score always means a better response." | Reward models are imperfect proxies. Overoptimising against them (using very small $\beta$ or running PPO too long) leads to reward hacking — responses that score high on the RM but are low quality by human standards. |

---

## 11 · Connections to other concepts

**DPO (Direct Preference Optimization):** DPO (Rafailov et al., 2023) eliminates the separate reward model training step by showing that the optimal RLHF policy has a closed-form expression in terms of the preference data. The Bradley-Terry RM loss is reparameterised directly in terms of the policy log-ratios. This removes the instability of the PPO stage at the cost of being an offline method.

**Inverse Reinforcement Learning (IRL):** RLHF is closely related to IRL — both infer a reward function from observed behavior (here, human preferences). The Bradley-Terry model is one specific IRL approach; maximum entropy IRL and regret-based approaches are alternatives.

**Proximal Policy Optimization (PPO):** PPO was designed for continuous control tasks but is particularly well-suited to RLHF because its clipping mechanism prevents the catastrophic policy collapse that can occur in language model fine-tuning.

**Reward Hacking / Goodhart's Law:** Any time you optimise a learned proxy reward, you risk overoptimising past the point where the proxy is faithful. The KL penalty is the primary defense, but careful reward model training (diverse data, ensemble methods) and monitoring are also necessary.

**Constitutional AI (RLAIF):** Bai et al. (2022) from Anthropic replaces human annotators in Stage 2 with AI-generated critiques based on a set of principles (the "constitution"), making the pipeline more scalable. This is called RL from AI Feedback (RLAIF).

**GRPO (Group Relative Policy Optimization):** Used in DeepSeek-R1, GRPO eliminates the value function critic by computing advantages relative to a group of sampled responses. This simplifies the architecture and reduces memory, at the cost of needing multiple rollouts per prompt.

**SFT and the LIMA hypothesis:** Stage 1 SFT is the foundation. The LIMA hypothesis (Zhou et al., 2023) argues that even 1,000 carefully curated SFT examples suffice to unlock most of a pretrained model's capabilities, calling into question whether RLHF Stages 2 and 3 are always necessary for alignment.

---

## 12 · One-screen summary

> **Problem:** Pretraining optimises for next-token prediction, not for helpfulness, honesty, or harmlessness.
>
> **Solution — three stages:**
>
> 1. **SFT:** Fine-tune the base model on $\sim$10K human-written (prompt, response) pairs. Initialises the policy $\pi_0$.
>
> 2. **Reward Model:** For each prompt, generate $k$ responses and have humans rank them. Train a scalar RM via Bradley-Terry loss: $\mathcal{L}_\text{RM} = -\mathbb{E}[\log \sigma(r_\phi(y_w) - r_\phi(y_l))]$.
>
> 3. **PPO:** Optimise $\pi_\theta$ to maximise $r_\phi(x, y) - \beta \,\mathrm{KL}(\pi_\theta \| \pi_0)$. The KL penalty prevents reward hacking and catastrophic deviation from the SFT distribution.
>
> **Key equations:**
> - Bradley-Terry: $P(y_w \succ y_l) = \sigma(r_\phi(y_w) - r_\phi(y_l))$
> - RLHF objective: $\max_\theta \mathbb{E}[r_\phi(x,y) - \beta \,\mathrm{KL}(\pi_\theta \| \pi_0)]$
> - PPO clipping: $\mathcal{L} = \mathbb{E}[\min(r_t(\theta)\hat{A},\; \mathrm{clip}(r_t(\theta), 1{-}\varepsilon, 1{+}\varepsilon)\hat{A})]$
>
> **Introduced by:** Christiano et al. (2017) for RL; Ouyang et al. (2022) / InstructGPT at LLM scale.
>
> **Successor:** DPO (Rafailov et al., 2023) eliminates the RM and PPO stages by reparameterising the RLHF objective directly in terms of policy log-ratios.

---

## 13 · Five-minute refresher

**Why RLHF?** Pretraining produces capable models that are not aligned with user intent. SFT alone is limited by the quality and coverage of labeled data. RLHF enables the model to improve beyond demonstrated behavior by optimising a learned human preference signal.

**Stage 1 — SFT in one sentence:** Fine-tune on human demonstrations to get a starting policy that generates coherent instruction-following responses.

**Stage 2 — RM in one sentence:** Train a scalar scorer on human pairwise preferences using $-\log \sigma(r_w - r_l)$ loss (Bradley-Terry).

**Stage 3 — PPO in one sentence:** Optimise the policy against the RM score minus a KL penalty to the SFT reference, using PPO's clipped surrogate objective to prevent catastrophic updates.

**The KL penalty's purpose:** It prevents the policy from exploiting reward model weaknesses (reward hacking) by penalising how far it drifts from the SFT model.

**What can go wrong:** Reward hacking (policy learns to game the RM), length bias (RM rewards verbosity), sycophancy (RM rewards agreement), alignment tax (capability degradation), and mode collapse (policy converges to repetitive outputs).

**Modern alternatives:** DPO (offline, no explicit RM), GRPO (no value function), RLAIF (AI-generated preferences), online DPO (iterative policy updates on freshly generated data).

---

## 14 · Further reading

| # | Citation |
|---|---|
| 1 | Christiano, P. et al. — **Deep Reinforcement Learning from Human Preferences.** *NeurIPS 2017.* The foundational paper introducing RLHF: applied to Atari and robotics, introduced the key idea of learning a reward model from pairwise human comparisons. |
| 2 | Ouyang, L. et al. — **Training language models to follow instructions with human feedback** (InstructGPT). *NeurIPS 2022 / arXiv:2203.02155.* The canonical application of RLHF to large language models: codified the three-stage SFT → RM → PPO pipeline; introduced the KL penalty formulation. |
| 3 | Schulman, J. et al. — **Proximal Policy Optimization Algorithms.** *arXiv:1707.06347, 2017.* The PPO paper: derives the clipped surrogate objective and actor-critic architecture used in Stage 3 of RLHF. |
| 4 | Stiennon, N. et al. — **Learning to summarize from human feedback.** *NeurIPS 2020 / arXiv:2009.01325.* First large-scale application of RLHF to a language task (abstractive summarisation); demonstrated that RLHF can surpass SFT even on tasks where demonstrations are available. |
| 5 | Bai, Y. et al. — **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.** *arXiv:2204.05862, 2022.* Anthropic's HH-RLHF paper: detailed analysis of the RLHF pipeline for helpfulness and harmlessness, including the HH dataset and reward model design choices. |
| 6 | Rafailov, R. et al. — **Direct Preference Optimization: Your Language Model is Secretly a Reward Model.** *NeurIPS 2023 / arXiv:2305.18290.* DPO: shows that the RLHF optimal policy can be expressed in closed form and the RM can be eliminated, simplifying the pipeline to a single supervised stage. |
| 7 | Gao, L., Schulman, J. and Hilton, J. — **Scaling Laws for Reward Model Overoptimization.** *arXiv:2210.10760, 2022.* Empirical study of how PPO performance degrades as the policy drifts further from the RM training distribution; introduces the KL-distance framework for overoptimisation analysis. |

---

<!-- ============================ BOTTOM NAV ============================ -->
<div align="center">

[⬅️ Q4‑01 — SFT & LIMA](./q01-sft-lima.md) &nbsp;|&nbsp; [📚 Back to Section 4](./README.md) &nbsp;|&nbsp; [🏠 Home](../../README.md) &nbsp;|&nbsp; [Q4‑03 — Reward Model ➡️](./q03-reward-model.md)

<sub>Found an error or have a sharper intuition? See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a> — answers follow the <a href="../../_TEMPLATE.md">answer template</a>.</sub>

</div>
