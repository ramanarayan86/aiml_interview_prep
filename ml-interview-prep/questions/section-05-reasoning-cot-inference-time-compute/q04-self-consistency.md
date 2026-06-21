<!-- ============================ TOP NAV ============================ -->
<div align="center">

[🏠 Home](../../README.md) &nbsp;•&nbsp; [📚 Section 5 — Reasoning &amp; CoT](./README.md) &nbsp;•&nbsp; [⬅️ Q5‑03](./q03-zero-shot-cot-mechanism.md) &nbsp;•&nbsp; [Q5‑05 ➡️](./q05-tree-of-thoughts.md)

</div>

---

# Q5‑04 · Explain self-consistency decoding. When does it help vs hurt?

<div align="center">

![Section](https://img.shields.io/badge/Section_5-Reasoning_%26_CoT-1f4fa3)
![Level](https://img.shields.io/badge/Level-Basic-0e8a6e)
![Tags](https://img.shields.io/badge/tags-self--consistency·majority--voting·sampling·reasoning-c77a12)
![Read](https://img.shields.io/badge/read_time-~15_min-5b6675)

</div>

> [!IMPORTANT]
> **The 20-second answer.** Self-consistency (Wang et al., 2023) replaces the greedy single-path decode of chain-of-thought with **majority voting over N stochastically sampled reasoning chains**. Given a prompt, you sample N completions at temperature T > 0, extract the final answer from each, and return the most common answer. On GSM8K with GPT-3 code-davinci-002, single-pass CoT reaches 56.5%; self-consistency at N=40 reaches 78.5% — a +22.0 pp gain with no additional training. It helps whenever **multiple valid reasoning paths converge on the same correct answer**, making wrong-path noise cancel out. It hurts when the model has a **systematic bias** (all paths repeat the same mistake), when N is too small for majority to be reliable, or when open-ended answers cannot be easily compared.

---

## Table of contents

1. [First principles: why single-path decoding fails](#1--first-principles-why-single-path-decoding-fails)
2. [The self-consistency algorithm](#2--the-self-consistency-algorithm)
3. [Figure 1 — Self-consistency pipeline](#3--figure-1--self-consistency-pipeline)
4. [Step-by-step worked example](#4--step-by-step-worked-example)
5. [Figure 2 — Accuracy vs. number of samples N](#5--figure-2--accuracy-vs-number-of-samples-n)
6. [Algorithm / pseudocode](#6--algorithm--pseudocode)
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

## 1 · First principles: why single-path decoding fails

Chain-of-thought prompting works by eliciting an explicit reasoning trace before producing an answer. This trace dramatically improves accuracy on multi-step problems. However, standard CoT uses **greedy decoding** (temperature T = 0): the model always picks the most probable next token, following a single deterministic path through reasoning space.

This single-path approach has a fundamental weakness: **one wrong step anywhere in the chain derails the entire answer**. Consider a 5-step arithmetic problem. If the model's greedy path makes a mistake at step 2, every downstream step is contaminated. Yet there may be many other correct reasoning paths — different valid orderings of the same algebraic manipulations, different levels of decomposition — that all arrive at the right answer.

The key insight behind self-consistency: **for well-posed problems with unique answers, correct paths outnumber incorrect paths in the model's sampling distribution**. At temperature T > 0, the model will sometimes take wrong paths, but if the correct answer is sufficiently probable in aggregate, it will receive the most votes across many independent samples.

This is analogous to ensemble learning: a single high-variance classifier can be wrong on any given example, but averaging over many classifiers reduces variance. Self-consistency does the same thing, but in the sampling dimension rather than the model dimension — and at inference time, not training time.

---

## 2 · The self-consistency algorithm

**Algorithm (Wang et al., 2023, ICLR 2023, arXiv:2203.11171):**

1. Fix a prompt $P$ — either few-shot CoT examples or a zero-shot CoT trigger
2. Sample $N$ reasoning chains $\{c_1, c_2, \ldots, c_N\}$ from the LLM with temperature $T > 0$ (typically $T = 0.7$)
3. Extract the final answer $a_i = f(c_i)$ from each chain $c_i$ using regex or a second-stage extraction prompt
4. Return the majority vote:

$$\hat{a} = \arg\max_{a} \sum_{i=1}^{N} \mathbf{1}[f(c_i) = a]$$

The key design choices are:

| Choice | Typical value | Reason |
|--------|---------------|--------|
| Temperature $T$ | 0.7 | High enough for path diversity; low enough for coherence |
| Number of samples $N$ | 10–40 | Accuracy plateaus around N=40; diminishing returns beyond |
| Answer extractor $f$ | Regex on last number/letter | Must be deterministic and consistent across chains |
| Voting rule | Plurality majority | Simple, works for discrete answers |

The method requires **no additional training**, no external verifier, and no model modification — only the ability to run the same model N times.

---

## 3 · Figure 1 — Self-consistency pipeline

<div align="center">
<img src="../../assets/s5q04/fig1-self-consistency-pipeline.svg" alt="Diagram of the self-consistency pipeline. A few-shot CoT prompt feeds into N parallel sampling runs at temperature 0.7. Five example paths are shown: paths 1, 2, 3, and 5 correctly compute 144 times 12 plus 36 times 8 equals 2016; path 4 incorrectly computes average price times total items to get 1800. A majority vote box collects all final answers and selects 2016 with 4 votes over 1800 with 1 vote. The correct answer 2016 is output. Below the figure the formula shows a-hat equals argmax over a of the sum from i=1 to N of the indicator function f(c_i) equals a. GPT-3 code-davinci-002 on GSM8K improves from 56.5% with greedy CoT to 78.5% with self-consistency at N=40." width="94%">
<br><sub><b>Figure 1.</b> The self-consistency pipeline. N independent reasoning chains are sampled at temperature T=0.7. Each chain is decoded to a final answer by the extractor function f. The majority vote selects the most frequently occurring answer. Wrong paths (path 4, red, using a flawed averaging heuristic) are outvoted by the 4 correct paths. The formula at the bottom gives the formal majority-vote aggregation rule.</sub>
</div>

---

## 4 · Step-by-step worked example

**Problem:** "A store sold 144 items at \$12 each and 36 items at \$8 each. What was the total revenue?"

**Ground truth:** $144 \times 12 + 36 \times 8 = 1728 + 288 = 2016$.

**Greedy CoT (T=0, N=1):** The model follows one path. If it makes a slip (e.g., computes $144 \times 12 = 1718$ due to a carry error), it returns \$2006 — incorrect.

**Self-consistency (T=0.7, N=5):**

| Path | Reasoning trace | Answer |
|------|-----------------|--------|
| 1 | "144×12=1728. 36×8=288. 1728+288=2016." | \$2016 ✓ |
| 2 | "Revenue = 144×12 + 36×8 = 1728+288 = 2016." | \$2016 ✓ |
| 3 | "12×144=1728. 8×36=288. 1728+288=2016." | \$2016 ✓ |
| 4 | "144+36=180. Average price=(12+8)/2=10. 180×10=1800." | \$1800 ✗ |
| 5 | "12×100=1200, 12×44=528, so 144×12=1728. 8×36=288. Total=2016." | \$2016 ✓ |

**Vote tally:** \$2016 = 4 votes, \$1800 = 1 vote.

**Self-consistency correctly selects \$2016.**

Path 4 made a conceptual error (weighted average instead of sum of revenues), but it was outvoted 4:1. The diversity of correct paths — computing the same arithmetic in different orders and decompositions — provides a natural noise floor against occasional bad chains.

**Verification:**

```python
# Verify: 144 * 12 + 36 * 8
revenue_A = 144 * 12   # = 1728
revenue_B = 36 * 8     # = 288
total = revenue_A + revenue_B  # = 2016
print(total)  # 2016
```

---

## 5 · Figure 2 — Accuracy vs. number of samples N

<div align="center">
<img src="../../assets/s5q04/fig2-accuracy-vs-N.svg" alt="Line chart showing GSM8K accuracy on the y-axis from 40 to 80 percent, and number of sampled reasoning paths N on the x-axis from 1 to 40. A teal curve starts at 56.5 percent for N=1 (same as greedy CoT, shown as a dashed navy horizontal baseline), rises steeply to approximately 65 percent at N=5, then to 71 percent at N=10, 76 percent at N=20, and plateaus at 78.5 percent at N=40. A callout box in the upper left highlights diminishing returns: N=1 to N=5 gives plus 8.5 percentage points, while N=20 to N=40 gives only plus 2.5 percentage points. Data is from Wang et al. 2023 using GPT-3 code-davinci-002." width="94%">
<br><sub><b>Figure 2.</b> Self-consistency accuracy on GSM8K as a function of N (Wang et al., 2023). The dashed navy line marks the greedy CoT baseline (56.5%). The teal curve shows majority-vote accuracy. The gain is steep at first — just 5 samples recovers most of the eventual improvement — and flattens to diminishing returns beyond N=20. The approximate +8.5 pp gain from N=1 to N=5 compresses to +2.5 pp from N=20 to N=40, informing practical tradeoffs between cost and accuracy.</sub>
</div>

---

## 6 · Algorithm / pseudocode

```
SELF-CONSISTENCY(prompt P, model M, N, temperature T, extractor f)
───────────────────────────────────────────────────────────────────
  Input:  P — few-shot or zero-shot CoT prompt
          M — language model
          N — number of samples (default 40)
          T — sampling temperature (default 0.7)
          f — answer extractor (regex / extraction prompt)

  Output: â — majority-vote answer

  Step 1: Sample N chains
    For i = 1 to N:
      c_i ← M.generate(P, temperature=T, do_sample=True)

  Step 2: Extract answers
    For i = 1 to N:
      a_i ← f(c_i)            # e.g., last number, last letter

  Step 3: Majority vote
    counts ← defaultdict(0)
    For i = 1 to N:
      counts[a_i] += 1

    â ← argmax_a counts[a]    # ties broken by first occurrence

  Return â
───────────────────────────────────────────────────────────────────

Complexity:
  Compute: N × (cost of one chain)    -- parallelizable
  Memory:  O(N × chain_length)
  Latency: O(chain_length) if parallel; O(N × chain_length) if serial
```

**Practical note on answer extraction:** The extractor $f$ is the most brittle part. For math, extract the last number matching a currency or integer pattern. For multiple-choice, extract the last `(A)/(B)/(C)/(D)` token. For code generation, extract the function body delimited by triple backticks. When answers cannot be normalised to a canonical form (e.g., open-ended text), self-consistency degenerates into soft similarity scoring, which is a different (harder) problem.

---

## 7 · PyTorch reference implementation

```python
import re
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable

def self_consistency_decode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    n: int = 40,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    answer_extractor: Callable[[str], str] | None = None,
) -> tuple[str, Counter]:
    """
    Self-consistency decoding (Wang et al., 2023).

    Args:
        model:            HuggingFace causal LM
        tokenizer:        matching tokenizer
        prompt:           few-shot CoT prompt string
        n:                number of sampled chains
        temperature:      sampling temperature (>0 for stochasticity)
        max_new_tokens:   max tokens per chain
        answer_extractor: function str -> str that extracts the final answer;
                          defaults to extracting the last integer

    Returns:
        (majority_answer, vote_counter)
    """
    if answer_extractor is None:
        def answer_extractor(text: str) -> str:
            # Extract the last integer or decimal in the completion
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
            return numbers[-1] if numbers else text.strip()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # Move to model device
    input_ids = input_ids.to(next(model.parameters()).device)

    answers = []
    # Run N independent samples
    # In production: batch all N calls together for efficiency
    with torch.no_grad():
        for _ in range(n):
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            new_tokens = output_ids[0, input_ids.shape[1]:]
            chain_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer = answer_extractor(chain_text)
            answers.append(answer)

    # Majority vote
    vote_counter = Counter(answers)
    majority_answer, _ = vote_counter.most_common(1)[0]

    return majority_answer, vote_counter


# ── Batched version (more efficient) ──────────────────────────────────────────
def self_consistency_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    n: int = 40,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> str:
    """
    Batched self-consistency: process b samples at once to amortise overhead.
    Repeat the prompt n times, split into batches of batch_size.
    """
    tokenizer.padding_side = "left"
    prompts = [prompt] * n
    all_answers = []

    for start in range(0, n, batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded.input_ids.to(next(model.parameters()).device)
        attention_mask = encoded.attention_mask.to(input_ids.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        for i, out in enumerate(output_ids):
            new_tokens = out[input_ids.shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
            answer = numbers[-1] if numbers else text.strip()
            all_answers.append(answer)

    return Counter(all_answers).most_common(1)[0][0]


# ── Usage example ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal smoke test with a dummy model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Q: What is 2 + 3? A: Let me think step by step."
    answer, votes = self_consistency_decode(
        model, tokenizer, prompt, n=5, temperature=0.9, max_new_tokens=32
    )
    print(f"Majority answer: {answer}")
    print(f"Vote distribution: {votes}")
```

> [!NOTE]
> For production use, consider vLLM's `SamplingParams(n=N, temperature=T)` which generates N completions in a single forward pass (via beam grouping / sampling groups), reducing overhead significantly compared to N separate model.generate() calls.

---

## 8 · Worked numerical example

**Problem:** "A store sold 144 items at \$12 each and 36 items at \$8 each. What was the total revenue?"

**Correct answer:** $144 \times 12 + 36 \times 8$.

**Verify arithmetic:**

$$144 \times 12 = 100 \times 12 + 44 \times 12 = 1200 + 528 = 1728$$

$$36 \times 8 = 288$$

$$1728 + 288 = 2016 \checkmark$$

**Self-consistency with N=5 paths:**

| Path | Final answer | Correct? |
|------|-------------|---------|
| 1 | \$2016 | ✓ |
| 2 | \$2016 | ✓ |
| 3 | \$2016 | ✓ |
| 4 | \$1800 | ✗ |
| 5 | \$2016 | ✓ |

**Vote tally:**

| Answer | Votes | Fraction |
|--------|-------|---------|
| \$2016 | 4 | 4/5 = 80% |
| \$1800 | 1 | 1/5 = 20% |

**Majority vote:** $\hat{a} = \$2016$ ✓

**Expected accuracy gain formula:** If each path independently has probability $p$ of being correct, the probability that the majority of $N$ paths is correct follows the binomial majority:

$$P(\text{majority correct}) = \sum_{k=\lceil N/2 \rceil}^{N} \binom{N}{k} p^k (1-p)^{N-k}$$

For $p = 0.565$ (GPT-3 code-davinci-002 greedy CoT accuracy) and $N = 40$:

```python
from scipy.stats import binom
import math

p = 0.565
N = 40
# P(majority correct) = P(X >= 21) where X ~ Binomial(40, 0.565)
p_majority_correct = 1 - binom.cdf(N // 2, N, p)
print(f"P(majority correct) = {p_majority_correct:.3f}")
# Output: P(majority correct) = 0.745
```

The actual gain (56.5% → 78.5%) is larger than this independent-path estimate because the model's correct paths are more similar to each other (they all follow valid arithmetic), while incorrect paths are more diverse (each wrong path fails in a different way). This systematic diversity amplifies the majority vote's effectiveness.

---

## 9 · Interview drill — follow-up questions

<details>
<summary><b>Q: Why does self-consistency work better than just increasing temperature and taking the best-of-N by some metric?</b></summary>

Best-of-N requires an external scoring function to identify the best sample — typically a verifier or reward model, which adds cost and may itself be imperfect. Self-consistency is completely self-contained: the majority vote is the only aggregation needed. For tasks with extractable discrete answers (arithmetic results, multiple-choice letters, yes/no), majority voting is a perfectly valid and cheap scoring function. Best-of-N is strictly more powerful when a reliable verifier exists (e.g., code execution for programming tasks) but self-consistency is the right baseline when no verifier is available.

</details>

<details>
<summary><b>Q: What happens to self-consistency if the model is systematically biased?</b></summary>

Self-consistency completely fails under systematic bias. If the model consistently makes the same type of error across all sampled paths — for example, it always applies an incorrect formula, or it has been exposed to memorised wrong solutions — then all N paths converge on the same wrong answer, which receives all votes. The majority vote then amplifies the bias rather than correcting it. This is the critical failure mode: self-consistency can only correct random errors, not systematic ones. Detection: if all N paths give identical or very similar answers despite T > 0, suspect systematic bias.

</details>

<details>
<summary><b>Q: How does self-consistency scale with N? Is there an optimal stopping point?</b></summary>

Accuracy follows a log-linear curve in N, with diminishing returns. The Wang et al. (2023) data on GSM8K shows: N=1 → 56.5%, N=5 → ~65%, N=10 → ~71%, N=20 → ~76%, N=40 → 78.5%. The marginal gain from N=1 to N=5 is ~8.5 pp; from N=20 to N=40 is only ~2.5 pp. Practically, N=10 to N=20 is the sweet spot: it recovers ~80–90% of the total gain at 25–50% of the cost of N=40. For latency-constrained applications, N=5 is often sufficient. The theoretical bound approaches the probability that the most common answer is correct under the model's distribution, which is fixed — adding more samples cannot exceed it.

</details>

<details>
<summary><b>Q: Does self-consistency work for open-ended generation?</b></summary>

Not directly. Majority voting requires that answers can be compared for equality or near-equality. For discrete tasks (arithmetic, multiple-choice, yes/no, short factual answers), the extractor function $f$ can normalise answers to a canonical form and exact matching works. For open-ended generation (essays, explanations, code), there is no natural notion of "same answer" — two correct explanations of the same concept may use entirely different words. Approximate approaches include: (1) embedding similarity clustering, (2) using an LLM to judge equivalence, (3) restricting to the extractable atomic claims in the answer. None are as clean as exact majority voting.

</details>

<details>
<summary><b>Q: How does self-consistency compare to the verifier / best-of-N approach (Cobbe et al., 2021)?</b></summary>

Both are inference-time compute scaling strategies. The key differences:

| Property | Self-Consistency | Best-of-N + Verifier |
|----------|-----------------|---------------------|
| External model needed | No | Yes (verifier) |
| Training required | No | Yes (verifier training) |
| Works without discrete answer | No | Yes (verifier scores free-form) |
| Cost at N=40 | 40 generations | 40 generations + 40 verifier calls |
| Failure mode | Systematic bias | Verifier reward hacking |

Self-consistency is the no-verifier baseline. A well-trained verifier (process reward model) outperforms self-consistency, especially when the model has systematic biases — but requires significant additional engineering.

</details>

<details>
<summary><b>Q: Self-consistency with N=40 costs 40x inference. How do you make it practical?</b></summary>

Three strategies: (1) **Parallelise** — the N chains are independent and can run simultaneously on multiple GPUs or instances; wall-clock latency is the same as a single chain. (2) **Early stopping** — if a clear majority (e.g., >70% of paths agree) emerges after N=10, stop early; otherwise continue to N=20 or N=40. (3) **Adaptive N** — use a low N for easy queries and high N only when the model's initial outputs show high variance. Some systems combine self-consistency with chain difficulty estimation to allocate compute adaptively.

</details>

---

## 10 · Common misconceptions

| Misconception | Reality |
|---|---|
| "Self-consistency always improves over greedy CoT." | It can hurt when N is very small (N=2–3) and the wrong path happens to win the coin flip. Practically, N ≥ 5 is needed for reliable gains. |
| "Higher temperature always gives better self-consistency." | Extreme temperatures (T > 1.0) produce incoherent chains where the answer extractor fails. T=0.7 is empirically optimal; values above 1.0 typically hurt. |
| "Self-consistency fixes bad few-shot examples." | Majority voting cannot compensate for systematically misleading exemplars that push all paths toward the same wrong reasoning pattern. Fix the prompt first. |
| "Self-consistency requires the same model as CoT." | True — it uses the same base model. But the temperature must be > 0. Using T=0 (greedy) produces the same chain N times, giving zero diversity and no benefit. |
| "You need N=40 to see gains." | No. Most of the gain (80–90%) is captured at N=10–20. N=40 is the Wang et al. evaluation setting, not a required minimum. |
| "Self-consistency is the same as ensemble decoding." | Self-consistency uses one model sampled N times. Ensemble decoding averages logits from multiple independently trained models. Both exploit variance reduction but via different mechanisms. |

---

## 11 · Connections to other concepts

**Tree-of-Thoughts (Q5-05):** ToT generalises the linear reasoning chain to a tree, adding a state evaluator that prunes bad branches. Self-consistency samples many complete chains and votes at the end; ToT evaluates partial chains during generation and focuses compute on promising branches. For problems where most paths fail (e.g., Game of 24 with 4% CoT success), ToT's targeted search vastly outperforms self-consistency's uniform sampling.

**Zero-Shot CoT (Q5-03):** Self-consistency works with both few-shot and zero-shot CoT prompts. In the original Wang et al. (2023) paper, it was evaluated with few-shot CoT. Kojima et al. (2022) show that "Let's think step by step" zero-shot CoT also benefits significantly from self-consistency.

**Process Reward Models / Verifiers (Q5-08):** Best-of-N with a process reward model (Lightman et al., 2023) can outperform self-consistency because it evaluates step-by-step correctness rather than just final-answer agreement. Self-consistency is the no-verifier special case.

**Inference-time compute scaling (Q5-01):** Self-consistency is the canonical example of trading inference-time compute for accuracy. It demonstrates that a weaker model with N=40 samples can match or exceed a stronger model with N=1.

**Majority voting in NLP (Q5-10):** Self-consistency is a form of output-space ensembling. Related to classical majority voting in boosting and committee machines, adapted to the discrete answer extraction setting of LLM reasoning tasks.

---

## 12 · One-screen summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SELF-CONSISTENCY DECODING — ONE-SCREEN SUMMARY               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ALGORITHM                                                                       │
│    1. Sample N reasoning chains at temperature T > 0 (default T=0.7, N=40)     │
│    2. Extract final answer a_i from each chain via regex / extraction prompt    │
│    3. Return majority vote: â = argmax_a Σ_i 1[f(c_i) = a]                     │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  KEY RESULTS (Wang et al., 2023 — GPT-3, GSM8K)                                 │
│    CoT single-pass:  56.5%                                                       │
│    N=5:             ~60%   (+13.1 pp)                                            │
│    N=10:            ~67%   (+20.1 pp)                                            │
│    N=40:             78.5% (+22.0 pp)  ← plateau                                │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHEN IT HELPS                          WHEN IT HURTS                           │
│  ✓ Multiple valid reasoning paths       ✗ Systematic model bias                 │
│  ✓ Discrete extractable answers         ✗ Open-ended generation                 │
│  ✓ High-variance model (mid-size)       ✗ N < 5 (too noisy)                     │
│  ✓ High-stakes, cost acceptable         ✗ Real-time latency constraints          │
│  ✓ Arithmetic, commonsense QA, MCQA     ✗ Extreme temperatures (T > 1)          │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COST ANALYSIS                                                                   │
│    Compute: N × (one chain cost)  — parallelisable                              │
│    Tokens:  N × (chain length)  e.g. N=40, 200 tok/chain → 8,000 tokens        │
│    Latency: same as single chain IF parallel; 40× slower if serial             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 13 · Five-minute refresher

**What problem does it solve?** Greedy chain-of-thought follows one path. One wrong step → wrong answer. Self-consistency exploits the fact that many different correct reasoning paths all end at the same answer, while wrong paths scatter across many wrong answers.

**How does it work?** Sample N reasoning chains at T=0.7. Extract the final answer from each. Return the most common answer (majority vote).

$$\hat{a} = \arg\max_{a} \sum_{i=1}^{N} \mathbf{1}[f(c_i) = a]$$

**Key number:** GPT-3 code-davinci-002 GSM8K: 56.5% (greedy CoT) → 78.5% (self-consistency, N=40). That is a +22.0 pp gain with zero additional training.

**Diminishing returns:** The gain from N=1→5 is ~13 pp; from N=20→40 is only ~2.4 pp. Use N=10–20 in practice.

**When it fails:** Systematic bias (all paths make the same mistake), open-ended answers that can't be compared by equality, N too small.

**Cost:** N forward passes, but they're independent and parallelisable. Wall-clock latency can be the same as N=1 with sufficient hardware.

---

## 14 · Further reading

- **Wang et al. (2023), "Self-Consistency Improves Chain of Thought Reasoning in Language Models," ICLR 2023:** The founding paper. Section 3 has all the main benchmarks; Appendix A discusses temperature sensitivity and answer extraction heuristics.
- **Wei et al. (2022), "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," NeurIPS 2022:** The CoT foundation that self-consistency builds on. Establishes the few-shot CoT paradigm.
- **Li et al. (2022), "On the Advance of Making Language Models Better Reasoners," arXiv:2206.02336:** Proposes diversity-of-thought and step-aware verifiers as extensions of self-consistency.
- **Cobbe et al. (2021), "Training Verifiers to Solve Math Word Problems," arXiv:2110.14168:** The competing best-of-N + verifier approach. Useful for understanding when a verifier outperforms majority voting.
- **Lightman et al. (2023), "Let's Verify Step by Step," arXiv:2305.20050:** Process reward models that score each reasoning step. A more powerful inference-time compute strategy when systematic bias is a concern.

---

## 15 · References

1. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D. — **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** *ICLR 2023 / arXiv:2203.11171.* — Introduces self-consistency; reports GPT-3 code-davinci-002 GSM8K 56.5% → 78.5% at N=40.

2. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E., Le, Q., Zhou, D. — **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.** *NeurIPS 2022 / arXiv:2201.11903.* — Establishes few-shot CoT; the baseline self-consistency improves upon.

3. Li, Y., Lin, Z., Zhang, S., Fu, Q., Chen, B., Lou, J.-G., Chen, W. — **On the Advance of Making Language Models Better Reasoners.** *arXiv:2206.02336, 2022.* — Extends self-consistency with diversity-of-thought and step-aware verification.

4. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., Schulman, J. — **Training Verifiers to Solve Math Word Problems.** *arXiv:2110.14168, 2021.* — Best-of-N with trained verifier; complementary to self-consistency.

5. Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., Iwasawa, Y. — **Large Language Models are Zero-Shot Reasoners.** *NeurIPS 2022 / arXiv:2205.11916.* — Zero-shot CoT ("Let's think step by step"); self-consistency also benefits zero-shot CoT chains.

---

<!-- ============================ BOTTOM NAV ============================ -->
<div align="center">

[⬅️ Q5‑03](./q03-zero-shot-cot-mechanism.md) &nbsp;|&nbsp; [📚 Back to Section 5](./README.md) &nbsp;|&nbsp; [🏠 Home](../../README.md) &nbsp;|&nbsp; [Q5‑05 — Tree-of-Thoughts ➡️](./q05-tree-of-thoughts.md)

<sub>Found an error or have a sharper intuition? See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a> — answers follow the <a href="../../_TEMPLATE.md">answer template</a>.</sub>

</div>
