<!-- ============================ TOP NAV ============================ -->
<div align="center">

[🏠 Home](../../README.md) &nbsp;•&nbsp; [📚 Section 1 — Transformer Architecture](./README.md) &nbsp;•&nbsp; [⬅️ Q32 — Attention Turing Complete](./q32-attention-turing-complete.md) &nbsp;•&nbsp; [Q34 — Graph Transformers ➡️](./q34-graph-transformers.md)

</div>

---

# Q33 · Attention as a kernel method — Performers, FAVOR+, and the kernel trick for efficient attention

<div align="center">

![Section](https://img.shields.io/badge/Section_1-Transformer_Architecture-1f4fa3)
![Level](https://img.shields.io/badge/Level-Advanced-b5321a)
![Tags](https://img.shields.io/badge/tags-kernel·Performers·FAVOR+·linear--attention·random--Fourier·Gaussian--process·RBF-0e8a6e)
![Read](https://img.shields.io/badge/read_time-~17_min-5b6675)

</div>

> [!IMPORTANT]
> **The 20-second answer.** Softmax attention computes $\text{softmax}(QK^\top/\sqrt{d}) V$, which is equivalent to normalizing a **kernel Gram matrix** $K(q_i, k_j) = \exp(q_i \cdot k_j / \sqrt{d})$ — a scaled exponential (RBF-like) kernel. The kernel view unlocks the key efficiency insight: if you can approximate $K(q, k) \approx \phi(q)^\top \phi(k)$ via a random feature map $\phi$, you can compute $(K'^\top V)$ first (size $r \times d$, independent of $N$) rather than materializing the full $N \times N$ matrix, reducing complexity from $O(N^2 d)$ to $O(N r d) = O(N)$ for fixed $r$. Performers (Choromanski et al., ICLR 2021) realize this with **FAVOR+** — Fast Attention Via positive Orthogonal Random features — achieving unbiased (or nearly-unbiased) estimation of the softmax kernel via random projections, with provable uniform convergence bounds independent of sequence length.

---

## Table of contents

1. [The kernel interpretation of softmax attention](#1--the-kernel-interpretation-of-softmax-attention)
2. [The kernel trick — avoiding the N×N matrix](#2--the-kernel-trick--avoiding-the-nn-matrix)
3. [Linear attention as a kernel method](#3--linear-attention-as-a-kernel-method)
4. [Performers and FAVOR+](#4--performers-and-favor)
5. [The random feature decomposition](#5--the-random-feature-decomposition)
6. [Connection to Gaussian processes and RBF kernels](#6--connection-to-gaussian-processes-and-rbf-kernels)
7. [Comparison table](#7--comparison-table)
8. [Reference implementation](#8--reference-implementation)
9. [Interview drill](#9--interview-drill)
10. [Common misconceptions](#10--common-misconceptions)
11. [One-screen summary](#11--one-screen-summary)
12. [References](#12--references)

---

## 1 · The kernel interpretation of softmax attention

Standard scaled dot-product attention:

$$
\text{Attn}(Q, K, V)_i = \frac{\sum_{j=1}^N \exp\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right) v_j}{\sum_{j=1}^N \exp\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)}
$$

Rewrite using a kernel function $K: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{>0}$:

$$
\text{Attn}(Q, K, V)_i = \frac{\sum_j K(q_i, k_j)\, v_j}{\sum_j K(q_i, k_j)}, \quad K(q, k) = \exp\!\left(\frac{q \cdot k}{\sqrt{d}}\right)
$$

This is a **kernel regression** (Nadaraya-Watson estimator): the output at query $q_i$ is a kernel-weighted average of values, with weights proportional to the kernel $K(q_i, k_j)$.

The kernel $K(q, k) = \exp(q \cdot k / \sqrt{d})$ is related to the **Gaussian RBF kernel**:

$$
K_\text{RBF}(q, k) = \exp\!\left(-\frac{\|q - k\|^2}{2}\right) = \exp\!\left(-\frac{\|q\|^2 + \|k\|^2}{2}\right) \cdot \exp(q \cdot k)
$$

The attention kernel drops the $\|q\|^2, \|k\|^2$ normalization terms (they cancel in the softmax ratio), so it is equivalent to an RBF kernel on the unit sphere when queries and keys are L2-normalized. Without normalization it is a scaled exponential kernel — positive definite but not strictly RBF.

---

## 2 · The kernel trick — avoiding the N×N matrix

In kernel methods, the **kernel trick** avoids computing the full $N \times N$ Gram matrix by working in a feature space where $K(q, k) = \langle \phi(q), \phi(k) \rangle$ for some feature map $\phi: \mathbb{R}^d \to \mathcal{H}$ (possibly infinite-dimensional).

For attention, if $K(q_i, k_j) = \phi(q_i)^\top \phi(k_j)$, then:

$$
\sum_j K(q_i, k_j) v_j = \phi(q_i)^\top \underbrace{\left(\sum_j \phi(k_j) v_j^\top\right)}_{\text{size } r \times d}
$$

$$
\sum_j K(q_i, k_j) = \phi(q_i)^\top \underbrace{\left(\sum_j \phi(k_j)\right)}_{\text{size } r}
$$

Computing $\sum_j \phi(k_j) v_j^\top$ and $\sum_j \phi(k_j)$ first (over all $j$) costs $O(N r d)$ and $O(N r)$ respectively. Then for each query $i$, the output is a dot product costing $O(rd)$. **Total cost: $O(Nrd)$, independent of $N^2$.**

This is the kernel trick applied to attention: instead of $O(N^2 d)$ to form and multiply the full attention matrix, you pay $O(Nrd)$ where $r$ is the number of random features (typically $r \ll N$).

---

## 3 · Linear attention as a kernel method

Katharopoulos et al. (2020; arXiv:2006.16236) formalized this view. They choose:

$$
\phi(x) = \text{elu}(x) + 1 \in \mathbb{R}^d
$$

This is a $d$-dimensional feature map (same dimension as the key/query). The approximation $K(q,k) \approx \phi(q)^\top \phi(k)$ is rough — ELU+1 does not well-approximate the exponential kernel — but the associativity trick still applies, giving $O(N)$ complexity.

**Causal (autoregressive) setting:** Maintain running state $S_i = \sum_{j \leq i} \phi(k_j) v_j^\top \in \mathbb{R}^{d \times d}$ and $z_i = \sum_{j \leq i} \phi(k_j) \in \mathbb{R}^d$:

$$
S_i = S_{i-1} + \phi(k_i) v_i^\top
$$

$$
o_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}
$$

This is an $O(d^2)$ per-step recurrence — an RNN with a $d \times d$ matrix hidden state.

---

## 4 · Performers and FAVOR+

**Paper:** "Rethinking Attention with Performers"
**Authors:** Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller
**Venue:** ICLR 2021 (Oral)
**arXiv:** 2009.14794

### The problem with naive random features

The classic Bochner theorem + random Fourier features (Rahimi & Recht, 2007) approximate shift-invariant kernels like the RBF. The RBF kernel admits:

$$
K_\text{RBF}(q, k) = \mathbb{E}_{\omega \sim \mathcal{N}(0, I_d)}\!\left[\cos(\omega^\top q) \cos(\omega^\top k) + \sin(\omega^\top q) \sin(\omega^\top k)\right]
$$

However, these features can be **negative** (cosine and sine can be negative), which breaks the normalization in attention (the denominator could be zero or negative).

### FAVOR+ — positive orthogonal random features

Performers solve this via a **positive** random feature decomposition. Using Lemma 1 of the paper, the softmax kernel admits:

$$
\text{SM}(q, k) = \exp(q \cdot k) = \mathbb{E}_{\omega \sim \mathcal{N}(0, I_d)}\!\left[\exp\!\left(\omega^\top q - \tfrac{\|q\|^2}{2}\right) \cdot \exp\!\left(\omega^\top k - \tfrac{\|k\|^2}{2}\right)\right]
$$

This leads to the **positive feature map**:

$$
\phi^+_\omega(x) = \exp\!\left(\omega^\top x - \tfrac{\|x\|^2}{2}\right) \in \mathbb{R}_{>0}
$$

With $r$ random projections $\omega_1, \ldots, \omega_r \sim \mathcal{N}(0, I_d)$:

$$
\phi^+(x) = \frac{1}{\sqrt{r}}\left[\exp\!\left(\omega_1^\top x - \tfrac{\|x\|^2}{2}\right), \ldots, \exp\!\left(\omega_r^\top x - \tfrac{\|x\|^2}{2}\right)\right] \in \mathbb{R}^r_{>0}
$$

This approximation is **unbiased**: $\mathbb{E}[\phi^+(q)^\top \phi^+(k)] = \exp(q \cdot k) = \text{SM}(q, k)$.

### Orthogonal random features

FAVOR+ further reduces variance by making the projections $\omega_1, \ldots, \omega_r$ **orthogonal** (using random orthogonal matrices via QR decomposition). Orthogonality reduces the estimator MSE compared to i.i.d. Gaussian projections.

### Complexity and theoretical guarantees

The approximate attention using FAVOR+ has:
- **Time:** $O(Lrd)$ where $L$ is sequence length, $r$ is the number of features, $d$ is dimension.
- **Space:** $O(Lr + Ld + rd)$ — no $L^2$ term.

**Uniform convergence (Theorem 4 of the paper):** For $m = \Theta(d / \delta^2 \cdot \log(4d^{3/4} R / \delta))$ random projections, the approximation achieves $\|\hat{A} - A\|_\infty \leq \varepsilon$ with constant probability, where $R$ bounds the query/key norms. The bound is **independent of sequence length $L$** — the approximation quality depends only on $d$ and the desired precision $\varepsilon$.

### Results

Performers achieved competitive performance with standard attention on:
- **Pixel-level image generation** (CIFAR-10).
- **Text modeling** (LM1B, PG-19).
- **Protein sequence analysis** — with substantial speedups at long sequence lengths.

---

## 5 · The random feature decomposition

The full pipeline for FAVOR+ attention:

$$
\hat{\text{Attn}}(Q, K, V) = \hat{D}^{-1}\!\left(Q'\bigl((K')^\top V\bigr)\right)
$$

where:
- $Q' = \phi^+(Q) \in \mathbb{R}^{L \times r}$ (apply feature map row-wise to queries)
- $K' = \phi^+(K) \in \mathbb{R}^{L \times r}$ (apply feature map row-wise to keys)
- $(K')^\top V \in \mathbb{R}^{r \times d}$ — computed first (the kernel trick step)
- $\hat{D} = \text{diag}(Q'((K')^\top \mathbf{1}_L))$ — normalization diagonal

Computing $(K')^\top V$ costs $O(Lrd)$; computing $Q' \cdot (K')^\top V$ costs $O(Lrd)$. Total $O(Lrd)$ versus $O(L^2 d)$ for standard attention.

---

## 6 · Connection to Gaussian processes and RBF kernels

Attention can be viewed as a **non-parametric Gaussian process regression** step:

- The keys $K = \{k_j\}$ are "training inputs."
- The values $V = \{v_j\}$ are "training outputs."
- The query $q_i$ is the "test input."
- The attention output $\text{Attn}(q_i, K, V)$ is the kernel-regression prediction at $q_i$.

The softmax kernel $\exp(q \cdot k / \sqrt{d})$ is a positive definite kernel (though not strictly stationary — it is not shift-invariant because it depends on $q \cdot k$, not $\|q - k\|$). On the unit sphere (after L2 normalization of queries and keys), it reduces to:

$$
K_\text{sphere}(q, k) = \exp\!\left(\frac{q \cdot k}{\sqrt{d}}\right) = \exp\!\left(\frac{\cos\theta_{q,k}}{\sqrt{d}}\right)
$$

which is related to the **arc-cosine kernel** and the **von Mises–Fisher distribution** on the sphere.

**Practical connection:** The kernel perspective explains why QK-norm (normalizing queries and keys before attention) is beneficial — it constrains the attention kernel to operate on the sphere, making it more like a proper stationary kernel with bounded eigenspectrum.

---

## 7 · Comparison table

| Method | Kernel / similarity | Feature map $\phi$ | Complexity | Unbiased? | Quality |
|---|---|---|---|---|---|
| Softmax attention | $\exp(q \cdot k / \sqrt{d})$ | Exact (no approximation) | $O(N^2 d)$ | Exact | Reference |
| Linear attention (Katharopoulos 2020) | $(\text{elu}(q)+1) \cdot (\text{elu}(k)+1)$ | $\text{elu}(x)+1 \in \mathbb{R}^d$ | $O(Nd^2)$ | No (poor approx) | Significant gap |
| Performers FAVOR+ (Choromanski 2021) | $\exp(q \cdot k)$ approximated | Positive random features $\phi^+ \in \mathbb{R}^r$ | $O(Nrd)$ | Yes (unbiased) | Good at $r \gg d$ |
| FAVOR+ with ORF | Same as above | Orthogonal random features | $O(Nrd)$ | Nearly unbiased | Better variance |
| Random Fourier features (Rahimi 2007) | RBF kernel | $\cos(\omega^\top x + b)$ | $O(Nrd)$ | Yes (but signed) | N/A for attention |
| SOFT (Lu 2021) | Gaussian kernel (low-rank) | Nyström approximation | $O(N)$ | Approximate | Vision tasks |

---

## 8 · Reference implementation

```python
import torch
import torch.nn.functional as F
import math

def favor_plus_feature_map(x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    Positive random feature map for FAVOR+ (Choromanski et al. 2021).
    x: (B, L, d)
    omega: (r, d) random projection matrix (orthogonalized rows)
    Returns: (B, L, r) positive feature matrix
    """
    # Project: (B, L, d) x (d, r) -> (B, L, r)
    proj = x @ omega.T               # (B, L, r)
    # Subtract ||x||^2 / 2 for unbiased estimation
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2   # (B, L, 1)
    # Positive features: exp(omega^T x - ||x||^2 / 2)
    features = torch.exp(proj - x_norm_sq) / math.sqrt(omega.shape[0])
    return features   # always positive

def performers_attention(Q, K, V, num_features=256):
    """
    Approximate softmax attention via FAVOR+ (O(NLd) complexity).
    Q, K, V: (B, L, d)
    """
    B, L, d = Q.shape
    # Sample random orthogonal projections
    omega = torch.randn(num_features, d, device=Q.device)
    omega, _ = torch.linalg.qr(omega.T)   # orthogonalize
    omega = omega.T[:num_features]         # (r, d)

    # Apply positive feature map to Q and K
    Q_prime = favor_plus_feature_map(Q, omega)   # (B, L, r)
    K_prime = favor_plus_feature_map(K, omega)   # (B, L, r)

    # KERNEL TRICK: compute (K'^T V) first -> (B, r, d)
    KtV = torch.bmm(K_prime.transpose(1, 2), V)   # (B, r, d)

    # Output numerator: Q' @ (K'^T V) -> (B, L, d)
    num = torch.bmm(Q_prime, KtV)

    # Normalization: Q' @ (K'^T 1) -> (B, L, 1)
    ones = torch.ones(B, L, 1, device=Q.device)
    Kt1 = torch.bmm(K_prime.transpose(1, 2), ones)   # (B, r, 1)
    denom = torch.bmm(Q_prime, Kt1)                  # (B, L, 1)

    return num / (denom + 1e-6)
```

> [!WARNING]
> This implementation is didactic. Production Performer implementations pre-sample random projections once and reuse them across batches. The QR decomposition for orthogonalization should be done offline. For causal masking (autoregressive), the implementation must use a sequential scan rather than global KtV — see the recurrent linear attention formulation in Q31.

---

## 9 · Interview drill

<details><summary><b>Q: Why must the feature map be non-negative for FAVOR+ to work?</b></summary>

In attention, the kernel weights serve as attention probabilities and appear in a ratio (numerator and denominator). If feature map values can be negative, the sum $\sum_j \phi(q)^\top \phi(k_j)$ can be zero or negative, making normalization undefined or negative. Positive features guarantee the denominator is always positive and the weights are well-defined.
</details>

<details><summary><b>Q: What is the key step in the kernel trick for attention?</b></summary>

The key step is reordering the matrix multiplications. Instead of computing $(Q K^\top) V$ (which requires first forming the $N \times N$ matrix $Q K^\top$), you compute $Q' (K'^\top V)$ — first forming the $r \times d$ matrix $K'^\top V$, then multiplying by $Q' \in \mathbb{R}^{N \times r}$. This exploits the associativity of matrix multiplication and reduces $O(N^2 d)$ to $O(N r d)$.
</details>

<details><summary><b>Q: How many random features r are needed for a good approximation?</b></summary>

The FAVOR+ convergence theorem requires $r = O(d \cdot \log(d/\varepsilon) / \varepsilon^2)$ for $\varepsilon$-uniform approximation, where $d$ is the key/query dimension and $\varepsilon$ is the desired error. In practice, $r = 256$ or $r = 512$ works well for $d = 64$ heads. Crucially, $r$ does not depend on the sequence length $N$.
</details>

<details><summary><b>Q: What is the connection between attention and kernel regression / Gaussian processes?</b></summary>

Attention is a Nadaraya-Watson kernel regression: the output at query $q$ is $\hat{f}(q) = \sum_j K(q, k_j) v_j / \sum_j K(q, k_j)$, a kernel-weighted average of the values $v_j$. This is exactly the GP posterior mean for a noise-free GP with kernel $K$ and training pairs $(k_j, v_j)$. The attention kernel $\exp(q \cdot k)$ is positive definite, making the analogy mathematically precise.
</details>

---

## 10 · Common misconceptions

| Misconception | Reality |
|---|---|
| "The kernel trick removes the quadratic cost entirely." | It reduces $O(N^2 d)$ to $O(Nrd)$. If $r \sim N$ features are needed for high accuracy, the savings diminish. In practice $r \ll N$. |
| "Linear attention (ELU+1) is the same as Performers." | Linear attention uses a fixed feature map ($\phi = \text{elu}+1$); Performers use random feature maps with theoretical approximation guarantees. The quality and approximation type differ significantly. |
| "FAVOR+ is unbiased for the full attention matrix." | FAVOR+ is unbiased for the softmax kernel value $\exp(q \cdot k)$. Due to the ratio structure (numerator/denominator), the final attention output is not exactly unbiased, but the bias is controlled. |
| "Performers replace softmax with something worse." | Performers approximate softmax with a provably accurate kernel approximation. The quality gap is mostly a variance issue (how many random features $r$ you use). |
| "The attention kernel is an RBF kernel." | The attention kernel $\exp(q \cdot k)$ is a scaled exponential kernel, equivalent to RBF only on the unit sphere (after L2 normalization of queries and keys). |

---

## 11 · One-screen summary

> **Kernel view:** Softmax attention = kernel regression with $K(q,k) = \exp(q \cdot k/\sqrt{d})$, related to the RBF kernel. **Kernel trick:** Approximate $K(q,k) = \phi(q)^\top \phi(k)$, compute $(K'^\top V)$ first to avoid the $N \times N$ matrix — cost drops from $O(N^2 d)$ to $O(Nrd)$. **FAVOR+:** Positive orthogonal random features that unbiasedly approximate the softmax kernel with convergence guarantees independent of $N$. **Linear attention:** Fixed feature map ($\phi = \text{elu}+1$), same trick, larger quality gap. **Use when:** $N > 8\text{k}$ and approximation quality $r \sim 256$ is acceptable.

---

## 12 · References

1. **Choromanski, K., et al.** "Rethinking Attention with Performers." ICLR 2021 (Oral). arXiv:2009.14794. [https://arxiv.org/abs/2009.14794](https://arxiv.org/abs/2009.14794)

2. **Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F.** "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020. arXiv:2006.16236. [https://arxiv.org/abs/2006.16236](https://arxiv.org/abs/2006.16236)

3. **Rahimi, A., Recht, B.** "Random Features for Large-Scale Kernel Machines." NeurIPS 2007. — The random Fourier feature foundation for kernel approximation.

4. **Lu, J., et al.** "SOFT: Softmax-free Transformer with Linear Complexity." NeurIPS 2021 Spotlight. arXiv:2110.11945. — Gaussian kernel approach.

5. **Vaswani, A., et al.** "Attention Is All You Need." NeurIPS 2017. — Baseline softmax attention.

6. **Qin, Z., et al.** "Why Softmax Attention Outperforms Linear Attention." arXiv:2310.11685, 2023. — Analysis of the quality gap and spikiness property. [https://arxiv.org/abs/2310.11685](https://arxiv.org/abs/2310.11685)

---

<!-- ============================ BOTTOM NAV ============================ -->
<div align="center">

[⬅️ Q32 — Attention Turing Complete](./q32-attention-turing-complete.md) &nbsp;|&nbsp; [📚 Back to Section 1](./README.md) &nbsp;|&nbsp; [🏠 Home](../../README.md) &nbsp;|&nbsp; [Q34 — Graph Transformers ➡️](./q34-graph-transformers.md)

<sub>Found an error? See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
