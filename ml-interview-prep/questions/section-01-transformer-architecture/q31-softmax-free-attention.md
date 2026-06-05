<!-- ============================ TOP NAV ============================ -->
<div align="center">

[🏠 Home](../../README.md) &nbsp;•&nbsp; [📚 Section 1 — Transformer Architecture](./README.md) &nbsp;•&nbsp; [⬅️ Q30 — Hyena / RetNet / RWKV](./q30-hyena-retnet-rwkv.md) &nbsp;•&nbsp; [Q32 — Attention Turing Complete ➡️](./q32-attention-turing-complete.md)

</div>

---

# Q31 · The role of softmax in attention — is softmax-free attention viable?

<div align="center">

![Section](https://img.shields.io/badge/Section_1-Transformer_Architecture-1f4fa3)
![Level](https://img.shields.io/badge/Level-Advanced-b5321a)
![Tags](https://img.shields.io/badge/tags-softmax·linear--attention·kernel--methods·SOFT·cosine--attention·quality--gap-0e8a6e)
![Read](https://img.shields.io/badge/read_time-~15_min-5b6675)

</div>

> [!IMPORTANT]
> **The 20-second answer.** Softmax in attention serves three purposes: (1) it normalizes raw dot-product scores into a probability distribution so they sum to 1, (2) it provides a non-linear "sharpening" that concentrates weight on the most relevant tokens (spiky, low-entropy attention), and (3) it is numerically bounded (outputs always in $[0,1]$). Softmax-free attention is possible and has been demonstrated — notably via kernel feature maps (linear attention, Performers), Gaussian kernels (SOFT), and cosine similarity (Swin v2) — but all currently suffer a meaningful quality gap on language tasks compared to softmax attention. The core problem is that removing softmax loses the normalization invariance and the data-dependent spikiness that makes attention so effective at retrieving specific tokens.

---

## Table of contents

1. [Why softmax? — first principles](#1--why-softmax--first-principles)
2. [What breaks without softmax](#2--what-breaks-without-softmax)
3. [Linear attention — the kernel trick](#3--linear-attention--the-kernel-trick)
4. [SOFT — Gaussian kernel attention](#4--soft--gaussian-kernel-attention)
5. [Cosine attention in Swin v2](#5--cosine-attention-in-swin-v2)
6. [Hardmax and sparsemax](#6--hardmax-and-sparsemax)
7. [Quality gap analysis](#7--quality-gap-analysis)
8. [Comparison table](#8--comparison-table)
9. [When softmax-free attention is viable](#9--when-softmax-free-attention-is-viable)
10. [Interview drill](#10--interview-drill)
11. [Common misconceptions](#11--common-misconceptions)
12. [One-screen summary](#12--one-screen-summary)
13. [References](#13--references)

---

## 1 · Why softmax? — first principles

Raw scaled dot-product scores:

$$
e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}
$$

are unbounded real numbers. Softmax converts them to a probability distribution:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{j'} \exp(e_{ij'})}
$$

This gives three properties:
1. **Normalization:** $\sum_j \alpha_{ij} = 1$ — the output is a convex combination of values.
2. **Non-linear sharpening:** The exponential magnifies differences; a score difference of $\Delta e$ leads to a probability ratio of $e^\Delta$. This creates "spiky" distributions that concentrate on a few tokens.
3. **Boundedness:** All weights lie in $(0,1)$; gradients are well-behaved.

> [!NOTE]
> **The spikiness property matters.** Empirical analysis (arXiv:2310.11685) shows that softmax attention has consistently low-entropy (spiky) weight distributions. This is what allows a single attention head to retrieve a specific token from a long context. Removing softmax tends to produce higher-entropy (flatter) weights, which blurs retrieval.

The $\sqrt{d}$ scaling was introduced by Vaswani et al. (2017) because without it, large $d$ causes $q \cdot k$ to grow large, pushing softmax into saturation (near-zero gradients).

---

## 2 · What breaks without softmax

<div align="center">
<img src="../../assets/q31/fig1-softmax-role.svg" alt="Side-by-side bar charts: spiky attention weights with softmax vs flat weights without softmax, with properties listed" width="92%">
<br><sub><b>Figure 1.</b> Softmax produces spiky, low-entropy attention weights (left) that concentrate on relevant tokens. Without softmax (right), weights are flatter and unbounded — blurring retrieval, especially at long context.</sub>
</div>

Without softmax, the output $\sum_j e_{ij} \cdot v_j$ (raw dot products times values) is:
- **Unbounded:** A single large score can dominate with arbitrary magnitude.
- **Non-normalized:** The output is no longer a convex combination; it can have arbitrary scale.
- **Linear in scores:** No non-linear sharpening — the model cannot easily concentrate attention.

These issues are not fatal, but they require mitigation via alternative mechanisms.

---

## 3 · Linear attention — the kernel trick

**Paper:** "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
**Authors:** Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret
**Venue:** ICML 2020
**arXiv:** 2006.16236

### The key idea

Softmax attention can be written as:

$$
\text{Attn}(Q, K, V)_i = \frac{\sum_j \text{sim}(q_i, k_j) \cdot v_j}{\sum_j \text{sim}(q_i, k_j)}
$$

where $\text{sim}(q,k) = \exp(q \cdot k / \sqrt{d})$ is a kernel function. The paper proposes approximating this with a **feature map** $\phi: \mathbb{R}^d \to \mathbb{R}^r$ such that:

$$
\text{sim}(q, k) \approx \phi(q)^\top \phi(k)
$$

Then:

$$
\text{Attn}(Q, K, V)_i = \frac{\phi(q_i)^\top \left(\sum_j \phi(k_j) v_j^\top\right)}{\phi(q_i)^\top \left(\sum_j \phi(k_j)\right)}
$$

By computing the sums $\sum_j \phi(k_j) v_j^\top$ and $\sum_j \phi(k_j)$ **first** (exploiting associativity of matrix multiplication), the complexity drops from $O(N^2 d)$ to $O(N r d)$, which is $O(N)$ when $r$ is fixed.

### Feature map choice

Katharopoulos et al. used:

$$
\phi(x) = \text{elu}(x) + 1
$$

where elu is the exponential linear unit. This ensures non-negativity (required for the ratio to be valid). The authors note that any non-negative kernel feature map works in principle.

### Causal masking complication

In autoregressive (decoder) settings, the causal mask prevents computing $\sum_j \phi(k_j) v_j^\top$ globally — it depends on the query position. The recurrent formulation resolves this: maintain a running sum $S_i = \sum_{j \leq i} \phi(k_j) v_j^\top$, updated at each step in $O(rd)$ per token.

### Quality gap

Despite $O(N)$ complexity, linear attention suffers a significant quality gap on language modeling:
- Perplexity on WikiText-103: linear attention is 4–6 ppl worse than softmax attention.
- Root cause: $\phi(x) = \text{elu}(x)+1$ does not approximate the softmax kernel well for typical $(q,k)$ magnitudes. The approximation loses the spikiness of softmax.

---

## 4 · SOFT — Gaussian kernel attention

**Paper:** "SOFT: Softmax-free Transformer with Linear Complexity"
**Authors:** Jiachen Lu, Jinghan Yao, Junge Zhang, Xiatian Zhu, Hang Xu, Weiguo Gao, Chunjing Xu, Tao Xiang, Li Zhang
**Venue:** NeurIPS 2021 Spotlight
**arXiv:** 2110.11945 (October 2021)

### Mechanism

SOFT replaces the dot-product softmax similarity with a **Gaussian kernel**:

$$
A_{ij} = \exp\!\left(-\frac{\|q_i - k_j\|^2}{2}\right) = \exp\!\left(-\frac{\|q_i\|^2 + \|k_j\|^2}{2}\right) \cdot \exp(q_i \cdot k_j)
$$

Because the Gaussian kernel is a positive semi-definite kernel, the full attention matrix $A \in \mathbb{R}^{N \times N}$ admits a **low-rank approximation** $A \approx UV^\top$ via the Nyström method or similar. The Moore-Penrose pseudo-inverse of this approximation is computed via Newton-Raphson iteration, giving a stable linear-complexity attention.

The key equation without further normalization:

$$
\text{SOFT}(Q, K, V) \approx A_\text{low-rank} \cdot V \quad \text{(linear in } N \text{)}
$$

### Properties
- No explicit softmax normalization — the Gaussian kernel naturally produces bounded, non-negative weights.
- Achieves linear complexity on vision tasks (ImageNet classification, COCO detection).
- Showed significant efficiency improvement over ViT variants.
- Less extensively validated on language modeling.

---

## 5 · Cosine attention in Swin v2

**Paper:** "Swin Transformer V2: Scaling Up Capacity and Resolution"
**Authors:** Ze Liu, Han Hu, Yutao Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Guodong Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo
**Venue:** CVPR 2022
**arXiv:** 2111.09883

### Mechanism

Swin v2 replaces dot-product attention with **scaled cosine attention**:

$$
\text{Sim}(q_i, k_j) = \frac{q_i \cdot k_j}{\|q_i\| \cdot \|k_j\|} \cdot \tau
$$

where $\tau$ is a learnable temperature scalar (initialized to a small value). The result is passed through softmax:

$$
\alpha_{ij} = \text{softmax}\!\left(\frac{q_i \cdot k_j}{\|q_i\| \cdot \|k_j\| \cdot \tau}\right)
$$

> [!NOTE]
> Swin v2's cosine attention retains the softmax operation — it changes the similarity metric, not the normalization. It is therefore "softmax with cosine similarity" rather than "softmax-free."

**Motivation:** Scaling to 3B parameters caused training instability due to large activation values in the dot products. Cosine similarity is bounded in $[-1, 1]$, preventing logit explosion. Combined with post-norm and log-spaced positional biases, this stabilized 3B-parameter training.

---

## 6 · Hardmax and sparsemax

**Hardmax** replaces softmax with an argmax:

$$
\alpha_{ij} = \begin{cases} 1 & \text{if } j = \arg\max_{j'} e_{ij'} \\ 0 & \text{otherwise} \end{cases}
$$

This is the theoretical basis of the Turing completeness proof (Pérez et al., 2021) but is non-differentiable and unusable in practice.

**Sparsemax** (Martins & Astudillo, 2016) provides a differentiable alternative that returns the Euclidean projection onto the probability simplex:

$$
\text{sparsemax}(z) = \arg\min_{p \in \Delta} \|p - z\|^2
$$

This produces exactly sparse outputs (many exact zeros) but is still differentiable almost everywhere. It is used in some structured prediction tasks but has not displaced softmax in general Transformers.

---

## 7 · Quality gap analysis

Research (arXiv:2310.11685, "Why Softmax Attention Outperforms Linear Attention") identifies two key properties of softmax attention that alternatives lose:

1. **Low-entropy (spiky) weights:** Softmax's exponential function concentrates weight on the top-scoring tokens. Linear attention with feature maps produces higher-entropy distributions.

2. **Dot-product monotonicity:** In softmax attention, if $q \cdot k_1 > q \cdot k_2$, then $\alpha_{i,k_1} > \alpha_{i,k_2}$. Some softmax-free formulations break this monotonicity.

Quantitatively (on language modeling):
- Standard softmax attention: reference perplexity.
- Linear attention (elu+1): +4–6 ppl worse.
- RWKV (WKV, effectively softmax-free): matches perplexity at scale (14B parameters) on diverse benchmarks.
- RetNet (no softmax, decay only): competitive perplexity at matched scale.

The gap narrows significantly at large scale and with architectural improvements (gating, better feature maps, normalization).

---

## 8 · Comparison table

| Method | Replaces softmax with | Complexity | Quality gap (LM) | Notes |
|---|---|---|---|---|
| Softmax attention | — (baseline) | $O(N^2)$ | 0 (reference) | Vaswani et al. 2017 |
| Linear attention | $\phi(q)^\top \phi(k)$ feature map | $O(N)$ | Large (4–6 ppl) | Katharopoulos et al. 2020 |
| Performers (FAVOR+) | Random Fourier features | $O(N r d)$ | Moderate | Choromanski et al. 2021 |
| SOFT | Gaussian kernel + low-rank approx | $O(N)$ | Moderate (vision tasks) | Lu et al. 2021, NeurIPS 2021 Spotlight |
| Cosine attention (Swin v2) | Cosine similarity (keeps softmax) | $O(N^2)$ | ~0 | Liu et al. 2022, CVPR 2022 |
| Hardmax | Argmax (non-differentiable) | $O(N^2)$ | Unusable | Theoretical only |
| Sparsemax | Euclidean projection to simplex | $O(N^2)$ | Small | Martins & Astudillo 2016 |
| RWKV WKV | Time-decay weighted sum | $O(N)$ training | Small at scale | Peng et al. 2023 |
| RetNet retention | Geometric decay | $O(N^2)$ / $O(1)$ | Small at scale | Sun et al. 2023 |

<div align="center">
<img src="../../assets/q31/fig2-softmax-free-spectrum.svg" alt="Quality vs efficiency scatter plot positioning softmax attention, linear attention, Performers, SOFT, RWKV/RetNet, cosine attention" width="92%">
<br><sub><b>Figure 2.</b> Quality–efficiency landscape for softmax and softmax-free attention variants. The ideal region (high quality, high efficiency) is top-right. RWKV/RetNet approach it at scale; linear attention sacrifices quality for O(N) inference; Performers balance both.</sub>
</div>

---

## 9 · When softmax-free attention is viable

**Use softmax-free attention when:**
- Sequence length $N > 10\text{k}$ and memory/compute is the bottleneck.
- Deployment requires constant-time inference per token (streaming, edge).
- Vision tasks with fixed input sizes (patch-based) — the quality gap is smaller here.
- Working at large model scale ($\geq 7\text{B}$ parameters) where the quality gap narrows.

**Stick with softmax attention when:**
- Maximum quality on language modeling or reasoning is required.
- Tasks require sharp associative recall from long contexts.
- Sequence length is moderate ($N \leq 8\text{k}$) where FlashAttention handles it.

---

## 10 · Interview drill

<details><summary><b>Q: What does the ELU+1 feature map do for linear attention?</b></summary>

$\phi(x) = \text{elu}(x) + 1$ maps inputs to non-negative values (since ELU is $\geq -1$, adding 1 makes it $\geq 0$). Non-negativity is required for the denominator $\sum_j \phi(q_i)^\top \phi(k_j)$ to be always positive, ensuring valid normalization. However, this feature map is a poor approximation to the exponential kernel of softmax, which is why the quality gap is significant.
</details>

<details><summary><b>Q: Why does removing softmax hurt autoregressive language modeling more than vision?</b></summary>

In language modeling, sharp retrieval of specific tokens is crucial (e.g., retrieving a pronoun's antecedent). Softmax's spikiness enables this. In vision, attention patterns tend to be smoother — nearby patches have semantically similar content. Softmax-free methods that produce higher-entropy distributions are less harmful in smooth-attention domains.
</details>

<details><summary><b>Q: Can you recover the quadratic-to-linear speedup with causal masking in linear attention?</b></summary>

Yes, but only via a recurrent formulation. Maintain a running state $S_i = \sum_{j \leq i} \phi(k_j) v_j^\top \in \mathbb{R}^{r \times d}$. At each position $i$: update $S_i = S_{i-1} + \phi(k_i) v_i^\top$, then output $o_i = \phi(q_i)^\top S_i / \phi(q_i)^\top z_i$ where $z_i = \sum_{j \leq i} \phi(k_j)$. This is $O(rd)$ per step — linear in sequence length.
</details>

---

## 11 · Common misconceptions

| Misconception | Reality |
|---|---|
| "Linear attention is just as good as softmax." | There is a significant quality gap (4–6 ppl on language tasks) with naive feature maps. |
| "Softmax is needed for the output to sum to 1." | Softmax ensures convex combination, but models can normalize by other means (e.g., dividing by sum of similarities as in linear attention). |
| "Swin v2 is softmax-free." | Swin v2 uses cosine similarity but keeps the softmax normalization — it is not softmax-free. |
| "Hardmax is better because it's sparser." | Hardmax is non-differentiable and cannot be trained by backpropagation in standard settings. |
| "The quality gap has been closed." | As of 2024–2025, softmax attention remains the gold standard for language modeling quality; linear attention alternatives are competitive only at large scale with additional design improvements. |

---

## 12 · One-screen summary

> **Role of softmax:** Normalizes scores to a probability distribution; provides non-linear sharpening (spiky attention) that enables precise token retrieval. **Softmax-free options:** (a) Linear attention via feature maps — $O(N)$, significant quality gap; (b) SOFT via Gaussian kernel + low-rank — $O(N)$, viable for vision; (c) Cosine attention (Swin v2) — keeps softmax, changes similarity metric for stability. **Verdict:** Softmax-free is viable for efficiency-dominated settings at large scale or on vision tasks; for highest-quality language modeling, softmax attention remains the standard.

---

## 13 · References

1. **Vaswani, A., et al.** "Attention Is All You Need." NeurIPS 2017. — Original softmax attention.

2. **Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F.** "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020. arXiv:2006.16236. [https://arxiv.org/abs/2006.16236](https://arxiv.org/abs/2006.16236)

3. **Lu, J., et al.** "SOFT: Softmax-free Transformer with Linear Complexity." NeurIPS 2021 Spotlight. arXiv:2110.11945. [https://arxiv.org/abs/2110.11945](https://arxiv.org/abs/2110.11945)

4. **Liu, Z., et al.** "Swin Transformer V2: Scaling Up Capacity and Resolution." CVPR 2022. arXiv:2111.09883. [https://arxiv.org/abs/2111.09883](https://arxiv.org/abs/2111.09883) — Cosine attention for training stability.

5. **Martins, A., Astudillo, R.** "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification." ICML 2016. — Sparsemax.

6. **Qin, Z., et al.** "Why Softmax Attention Outperforms Linear Attention." arXiv:2310.11685, 2023. — Analysis of the quality gap. [https://arxiv.org/abs/2310.11685](https://arxiv.org/abs/2310.11685)

7. **Choromanski, K., et al.** "Rethinking Attention with Performers." ICLR 2021. arXiv:2009.14794. [https://arxiv.org/abs/2009.14794](https://arxiv.org/abs/2009.14794) — FAVOR+ random feature approximation (see also Q33).

---

<!-- ============================ BOTTOM NAV ============================ -->
<div align="center">

[⬅️ Q30 — Hyena / RetNet / RWKV](./q30-hyena-retnet-rwkv.md) &nbsp;|&nbsp; [📚 Back to Section 1](./README.md) &nbsp;|&nbsp; [🏠 Home](../../README.md) &nbsp;|&nbsp; [Q32 — Attention Turing Complete ➡️](./q32-attention-turing-complete.md)

<sub>Found an error? See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
