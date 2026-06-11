<!-- ============================ TOP NAV ============================ -->
<div align="center">

[🏠 Home](../../README.md) &nbsp;•&nbsp; [📚 Section 1 — Transformer Architecture](./README.md) &nbsp;•&nbsp; [⬅️ Q29 — Differential Transformer](./q29-differential-transformer.md) &nbsp;•&nbsp; [Q31 — Softmax-Free Attention ➡️](./q31-softmax-free-attention.md)

</div>

---

# Q30 · Hyena, RetNet, and RWKV — complexity profiles, mechanisms, and when to use each

<div align="center">

![Section](https://img.shields.io/badge/Section_1-Transformer_Architecture-1f4fa3)
![Level](https://img.shields.io/badge/Level-Advanced-b5321a)
![Tags](https://img.shields.io/badge/tags-Hyena·RetNet·RWKV·sub--quadratic·linear--complexity·long--convolution·retention-0e8a6e)
![Read](https://img.shields.io/badge/read_time-~20_min-5b6675)

</div>

> [!IMPORTANT]
> **The 20-second answer.** All three replace standard $O(N^2)$ attention with sub-quadratic or linear alternatives. **Hyena** (Poli et al., 2023) uses implicitly-parametrized long convolutions computed via FFT: $O(N \log N)$ training, $O(N)$ inference (sequential). **RetNet** (Sun et al., 2023) introduces a "retention" mechanism with three computation modes — parallel ($O(N^2)$) for training, recurrent ($O(1)$ per token) for inference, and chunked ($O(C^2 N/C)$) for throughput balance. **RWKV** (Peng et al., 2023) uses a WKV operator — a time-decay weighted key-value that can be parallelized during training ($O(Td^2)$) and runs as a true RNN at inference ($O(d)$ per token). None fully matches attention quality on associative-recall tasks, but all trade some accuracy for dramatically better inference efficiency.

---

## Table of contents

1. [Motivation — the quadratic wall](#1--motivation--the-quadratic-wall)
2. [Hyena — implicit long convolutions](#2--hyena--implicit-long-convolutions)
3. [RetNet — retention with three modes](#3--retnet--retention-with-three-modes)
4. [RWKV — reinventing RNNs for the Transformer era](#4--rwkv--reinventing-rnns-for-the-transformer-era)
5. [Complexity comparison table](#5--complexity-comparison-table)
6. [When to use each](#6--when-to-use-each)
7. [Algorithm & pseudocode](#7--algorithm--pseudocode)
8. [Reference implementations (sketches)](#8--reference-implementations-sketches)
9. [Worked numerical example](#9--worked-numerical-example)
10. [Where it's used / where it breaks](#10--where-its-used--where-it-breaks)
11. [Cousins & alternatives](#11--cousins--alternatives)
12. [Interview drill](#12--interview-drill)
13. [Common misconceptions](#13--common-misconceptions)
14. [One-screen summary](#14--one-screen-summary)
15. [References](#15--references)

---

## 1 · Motivation — the quadratic wall

Standard multi-head attention:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

requires materializing the $N \times N$ attention matrix, costing $O(N^2 d)$ time and $O(N^2)$ memory. For $N = 100\text{k}$ tokens (a book), the matrix is $10^{10}$ entries — infeasible. FlashAttention reduces memory to $O(N)$ via tiling but does not change the asymptotic time complexity. A fundamentally different mechanism is needed.

The three approaches below replace the attention operation with sub-quadratic primitives while preserving much of the modeling capacity.

---

## 2 · Hyena — implicit long convolutions

<div align="center">
<img src="../../assets/q30/fig1-complexity-comparison.svg" alt="Log-log chart of training cost vs sequence length for Transformer, FlashAttention, Hyena, RetNet, RWKV with crossover at ~6K tokens" width="92%">
<br><sub><b>Figure 1.</b> Training time scaling with sequence length N. Hyena (O(N log N)) and RetNet/RWKV (linear modes) become faster than FlashAttention above ~6 000 tokens, a key motivation for sub-quadratic architectures.</sub>
</div>

**Paper:** "Hyena Hierarchy: Towards Larger Convolutional Language Models"
**Authors:** Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré
**Venue:** ICML 2023
**arXiv:** 2302.10866 (February 2023)

### Core mechanism

The Hyena operator replaces attention with a **recurrence of long convolutions and element-wise gating**. Given input projections $v, z^{(1)}, \ldots, z^{(N_\text{ord})}$ (where $N_\text{ord}$ is the "order" of the operator, typically 2–3), the recurrence is:

$$
y^{(0)} = v
$$

$$
y^{(n)} = z^{(n)} \star \bigl(h^{(n)} * y^{(n-1)}\bigr), \quad n = 1, \ldots, N_\text{ord}
$$

where $\star$ denotes element-wise multiplication, $*$ denotes long convolution, and $h^{(n)}$ is a **filter parametrized by a small feed-forward network** (MLP over positional features). The filters are "implicit" — they are not stored as explicit weight vectors but generated on-the-fly from an MLP, giving sublinear parameter scaling with sequence length.

In pseudocode:

```
v, z1, z2 = input_projections(u)   # three projections of input u
h1 = hyena_filter(L)               # filter at order 1, generated by MLP
h2 = hyena_filter(L)               # filter at order 2
y  = v
y  = z1 * fftconv(h1, y)          # element-wise gating + FFT convolution
y  = z2 * fftconv(h2, y)          # repeat for each order
output = y
```

### Complexity

- **Training:** $O(N_\text{ord} \cdot N \log N)$ via FFT convolution. For $N_\text{ord} = 2$ and long sequences, this beats FlashAttention at $N \gtrsim 6\text{k}$; at $N = 100\text{k}$ Hyena is $\approx 100\times$ faster.
- **Inference:** $O(N \cdot d)$ sequential; no native parallel decoding advantage (though the recurrence can be unrolled).

### Strengths and limitations

| Dimension | Hyena |
|---|---|
| Expressivity | Strictly sub-quadratic; cannot exactly simulate softmax attention |
| Long-context speed | Excellent ($O(N \log N)$) |
| Language modeling | Closes most of the gap with attention at matched FLOP budgets |
| Associative recall | Weaker than attention (finite-depth recurrence) |
| Parameter count | Sublinear in $N$ (implicit filters) |
| Crossover with FlashAttn | $\sim 6\text{k}$ tokens |

---

## 3 · RetNet — retention with three modes

**Paper:** "Retentive Network: A Successor to Transformer for Large Language Models"
**Authors:** Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei
**arXiv:** 2307.08621 (July 2023)

### Core mechanism: the retention function

RetNet replaces softmax attention with a **retention** function based on a geometric decay. For a sequence position $n$ and head $h$, define a decay rate $\gamma_h \in (0,1)$ (different per head — multi-scale retention).

**Parallel representation** (for training):

$$
\text{Retention}(X) = (QK^\top \odot D)\, V
$$

where $D \in \mathbb{R}^{N \times N}$ is the causal decay matrix:

$$
D_{nm} = \begin{cases} \gamma^{n-m} & \text{if } n \geq m \\ 0 & \text{otherwise} \end{cases}
$$

This replaces the softmax with a simple masking + decay operation. No normalization is needed; the decay ensures old tokens have exponentially less influence.

**Recurrent representation** (for $O(1)$ inference):

The state at position $n$ is a matrix $S_n \in \mathbb{R}^{d \times d}$:

$$
S_n = \gamma \cdot S_{n-1} + k_n^\top v_n
$$

$$
\text{Retention}(x_n) = q_n \cdot S_n
$$

This is a pure RNN with constant-size state — $O(d^2)$ per step, $O(1)$ in sequence length.

**Chunkwise recurrent representation** (for throughput balance):

Divide the sequence into chunks of size $C$. Within each chunk, use the parallel representation ($O(C^2)$). Across chunks, pass the state $S$ recurrently. This gives $O(NC)$ time with cache-friendly memory access.

### Multi-scale retention

Each head $h$ uses a distinct $\gamma_h$:

$$
\gamma_h = 1 - 2^{-5 - h}, \quad h = 1, \ldots, H
$$

so heads range from $\gamma \approx 0.97$ (long memory) to $\gamma \approx 0.5$ (short memory).

### Complexity

| Mode | Time | Memory | Use case |
|---|---|---|---|
| Parallel | $O(N^2 d)$ | $O(N^2)$ | Training (GPU-parallel) |
| Recurrent | $O(N d^2)$ | $O(d^2)$ | Autoregressive inference |
| Chunkwise | $O(NC \cdot d)$ | $O(C^2 + d^2)$ | Long-sequence throughput |

### Strengths and limitations

| Dimension | RetNet |
|---|---|
| Training | Parallel, comparable to Transformer |
| Inference | $O(1)$ per token (true RNN) |
| Perplexity | Competitive with Transformer at matched params |
| Associative recall | Weaker (exponential decay loses distant info) |
| Flexible context window | No (fixed decay, no sliding attention) |

---

## 4 · RWKV — reinventing RNNs for the Transformer era

**Paper:** "RWKV: Reinventing RNNs for the Transformer Era"
**Authors:** Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, Rui-Jie Zhu
**Venue:** EMNLP 2023 Findings
**arXiv:** 2305.13048 (May 2023)

### Core mechanism: the WKV operator

RWKV uses two alternating sublayers — **Time-Mixing** (replaces attention) and **Channel-Mixing** (replaces FFN).

**Time-Mixing — the WKV operator:**

For token at position $t$, first compute receptance $r_t$, key $k_t$, value $v_t$ via token-shifted linear projections:

$$
r_t = W_r \cdot (\mu_r \odot x_t + (1-\mu_r) \odot x_{t-1})
$$

$$
k_t = W_k \cdot (\mu_k \odot x_t + (1-\mu_k) \odot x_{t-1})
$$

$$
v_t = W_v \cdot (\mu_v \odot x_t + (1-\mu_v) \odot x_{t-1})
$$

The WKV operator is a time-decay weighted sum:

$$
\text{wkv}_t = \frac{\displaystyle\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} \odot v_i \;+\; e^{u + k_t} \odot v_t}{\displaystyle\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} \;+\; e^{u + k_t}}
$$

where $w \in \mathbb{R}^d$ is a **non-negative learnable time-decay vector** and $u \in \mathbb{R}^d$ is a **learnable bonus for the current token**. The output is:

$$
o_t = W_o \cdot (\sigma(r_t) \odot \text{wkv}_t)
$$

**Channel-Mixing (replaces FFN):**

$$
o'_t = \sigma(r'_t) \odot \bigl(W'_v \cdot \max(k'_t, 0)^2\bigr)
$$

using squared ReLU activation.

### Key design insight

The WKV formula can be computed in two ways:
1. **Parallel (training):** Unroll the sum; it has the structure of a cumulative sum that can be parallelized along the time dimension, similar to a prefix scan. Training cost dominates at $O(BTd^2)$ (matrix multiplications in $W_r, W_k, W_v, W_o$).
2. **Recurrent (inference):** Maintain two running accumulators $a_t$ and $b_t$: $a_t = \gamma a_{t-1} + e^{k_t} v_t$ and $b_t = \gamma b_{t-1} + e^{k_t}$, so $\text{wkv}_t = a_t / b_t$. This is $O(d)$ per token — constant in sequence length.

### RWKV-5 (Eagle) and RWKV-6 (Finch)

The follow-up paper (Peng et al., 2024; arXiv:2404.05892) introduces:
- **RWKV-5 / Eagle:** Multi-headed **matrix-valued** states (replacing scalar accumulation with outer products), reformulated receptance, additional gating.
- **RWKV-6 / Finch:** Data-dependent time-mixing weights (the decay $w$ becomes input-dependent), enabling more expressive context-dependent memory dynamics. 7B Finch improved +5.38% vs 7B Eagle across benchmarks.

### Complexity

| Aspect | Training | Inference |
|---|---|---|
| Time | $O(BTd^2)$ | $O(Td)$ per step |
| Memory | $O(T^2 + Td)$ (scan) | $O(d)$ (state only) |
| Mechanism | Parallel prefix scan | Sequential RNN state update |

---

## 5 · Complexity comparison table

<div align="center">
<img src="../../assets/q30/fig2-retnet-three-modes.svg" alt="RetNet three modes: Parallel O(N²), Recurrent O(d²) per token, Chunked O(NC) training" width="92%">
<br><sub><b>Figure 2.</b> RetNet's three computation modes. Parallel mode (training): O(N²d), identical complexity to standard attention. Recurrent mode (inference): O(d²) per token, constant KV cache. Chunked mode (efficient training): interpolates between the two.</sub>
</div>

| Model | Training time | Training memory | Inference / token | State size |
|---|---|---|---|---|
| Standard Transformer | $O(N^2 d)$ | $O(N^2)$ | $O(N d)$ (KV cache grows) | $O(Nd)$ KV cache |
| FlashAttention | $O(N^2 d)$ | $O(N)$ (tiled) | $O(Nd)$ | $O(Nd)$ KV cache |
| Hyena ($p$ orders) | $O(p \cdot N \log N)$ | $O(N)$ | $O(Nd)$ sequential | $O(d)$ per filter |
| RetNet (parallel) | $O(N^2 d)$ | $O(N^2)$ | — | — |
| RetNet (recurrent) | — | — | $O(d^2)$ | $O(d^2)$ state matrix |
| RetNet (chunked) | $O(NCd)$ | $O(C^2 + d^2)$ | $O(d^2)$ | $O(d^2)$ state matrix |
| RWKV (training) | $O(Td^2)$ | $O(Td)$ | — | — |
| RWKV (inference) | — | — | $O(d)$ | $O(d)$ accumulators |
| Mamba (SSM) | $O(N \cdot d \cdot D)$ | $O(ND)$ | $O(dD)$ | $O(dD)$ state |

$N$ = sequence length, $d$ = model dimension, $T$ = sequence length (RWKV notation), $C$ = chunk size, $D$ = SSM state dimension.

---

## 6 · When to use each

| Scenario | Best choice | Reason |
|---|---|---|
| Long-context language modeling ($N > 10\text{k}$) with training budget | Hyena or RetNet chunked | Sub-quadratic training |
| Streaming inference, edge deployment | RWKV or RetNet recurrent | True $O(1)$ per token RNN |
| High-quality retrieval / associative recall | Transformer + FlashAttention | Attention is still best here |
| Very long sequences ($N > 100\text{k}$) | Hyena | FFT convolution scales best |
| Matching Transformer perplexity at scale | RetNet or RWKV (14B) | Both scale like Transformers in perplexity |
| Multilingual, diverse tasks | RWKV-6 (Finch) | Trained on diverse multilingual corpus |

---

## 7 · Algorithm & pseudocode

**Hyena operator (implicit long convolution):**

```text
INPUT : u     # [B, T, d] — input sequence
        h     # implicit filter (parameterised MLP over positions)
        order # number of projections (default 2)

1.  v1, v2 = linear_proj(u)             # split into two feature streams
2.  k = h(positions 0..T-1)             # [T, d] — learned filter, position-dependent
3.  y = v1 ⊛ k                          # long convolution via FFT: O(T log T)
    # ⊛ is causal convolution
4.  output = v2 ⊙ y                     # element-wise gating
RETURN output                           # [B, T, d]
```

**RetNet parallel (training) form:**

```text
INPUT : X        # [B, T, d_model]
        γ        # per-head decay scalar ∈ (0, 1)

1.  Q, K, V = X @ W_Q, X @ W_K, X @ W_V
2.  # Causal decay mask
    D[i,j] = γ^{i-j}  if i >= j, else 0    # [T, T]
3.  retention = (Q @ K.T) ⊙ D              # [T, T] — decayed attention
4.  output = retention @ V
RETURN output
```

**RetNet recurrent (inference) form:**

```text
INPUT : x_t     # single token [d_model]
        s_{t-1} # recurrent state [d_key, d_value]
        γ       # decay scalar

1.  q_t = x_t @ W_Q,  k_t = x_t @ W_K,  v_t = x_t @ W_V
2.  s_t = γ · s_{t-1} + k_t.T ⊗ v_t    # outer product update
3.  y_t = q_t · s_t                      # retrieval via dot product
RETURN y_t, s_t
```

**RWKV time-mix (WKV operator):**

```text
INPUT : x_t       # [d_model] — current token
        u, w      # time-first (u) and decay (w) vectors, learnable
        num, den  # running numerator and denominator (recurrent state)

1.  k_t = exp(u + w_t + log(x_t))  # key score
2.  v_t = x_t @ W_V
3.  num_t = exp(-w) · num_{t-1} + exp(k_t) · v_t
4.  den_t = exp(-w) · den_{t-1} + exp(k_t)
5.  wkv_t = num_t / den_t
RETURN wkv_t, num_t, den_t
```

---

## 8 · Reference implementations (sketches)

### Hyena (pseudocode)
```python
def hyena_operator(u, order=2):
    # u: (B, L, d)
    projections = [linear(u) for _ in range(order + 1)]  # v, z1, z2, ...
    v = projections[0]
    for n in range(1, order + 1):
        h = mlp_filter(L)          # implicit filter via MLP, shape (L, d)
        v = projections[n] * fft_conv(h, v)  # elem-wise gate + FFT conv
    return linear_output(v)
```

### RetNet recurrent (pseudocode)
```python
def retnet_step(x_t, S, gamma):
    # x_t: (d,), S: (d, d) state matrix
    q = W_Q @ x_t;  k = W_K @ x_t;  v = W_V @ x_t
    S = gamma * S + torch.outer(k, v)   # state update
    y = q @ S                           # retrieval
    return y, S
```

### RWKV inference step (pseudocode)
```python
def rwkv_step(x_t, x_prev, a, b, w, u):
    # Token shift
    k = W_k @ (mu_k * x_t + (1 - mu_k) * x_prev)
    v = W_v @ (mu_v * x_t + (1 - mu_v) * x_prev)
    r = W_r @ (mu_r * x_t + (1 - mu_r) * x_prev)
    # WKV update
    wkv = (a + exp(u + k) * v) / (b + exp(u + k))
    a = exp(-exp(w)) * a + exp(k) * v   # accumulator update
    b = exp(-exp(w)) * b + exp(k)
    out = W_o @ (sigmoid(r) * wkv)
    return out, a, b
```

---

## 9 · Worked numerical example

We trace a single RetNet recurrent step with $d = 4$, $\gamma = 0.9$, and $N = 3$ tokens to make the retention mechanics concrete.

**Setup.** Single-head RetNet, dimension $d = 4$. Input tokens (rows of $X$):

| Step $n$ | Token | $x_n \in \mathbb{R}^4$ |
|---|---|---|
| 1 | "the" | $[1, 0, 0, 0]$ |
| 2 | "cat" | $[0, 1, 0, 0]$ |
| 3 | "sat" | $[0, 0, 1, 0]$ |

Projection matrices simplified to identity: $W_Q = W_K = W_V = I$.

**Recurrent mode** — compute state $S_n = \gamma S_{n-1} + k_n^\top v_n$ and output $y_n = q_n S_n$.

**Step 1 ($n=1$):**

$$k_1 = v_1 = x_1 = [1,0,0,0]$$

$$S_1 = \gamma S_0 + k_1^\top v_1 = 0 + [1,0,0,0]^\top [1,0,0,0] = \begin{bmatrix}1&0&0&0\\0&0&0&0\\0&0&0&0\\0&0&0&0\end{bmatrix}$$

$$y_1 = q_1 S_1 = [1,0,0,0] \cdot S_1 = [1,0,0,0]$$

**Step 2 ($n=2$):**

$$S_2 = 0.9 \cdot S_1 + k_2^\top v_2 = 0.9 \begin{bmatrix}1&0\\0&0\end{bmatrix} + \begin{bmatrix}0&0\\1&0\end{bmatrix} = \begin{bmatrix}0.9&0\\1.0&0\end{bmatrix} \quad (\text{top-left } 2\times2 \text{ shown})$$

$$y_2 = q_2 S_2 = [0,1,0,0] \cdot S_2 = [1.0, 0, \ldots]$$

Token "cat" retrieves the "cat" entry in the state (weight 1.0) and the decayed "the" entry (weight 0.9 on the row it contributed). The decay factor $\gamma = 0.9$ reduces older information.

**Step 3 ($n=3$):**

$$S_3 = 0.9 \cdot S_2 + k_3^\top v_3$$

After 3 steps, $S_1$'s contribution has been multiplied by $0.9^2 = 0.81$. $S_2$'s contribution has been multiplied by $0.9^1 = 0.9$.

**Decay schedule** for $\gamma = 0.9$:

| Steps back | Retention factor | % retained |
|---|---|---|
| 1 | $0.9^1 = 0.900$ | 90% |
| 2 | $0.9^2 = 0.810$ | 81% |
| 5 | $0.9^5 = 0.590$ | 59% |
| 10 | $0.9^{10} = 0.349$ | 35% |
| 20 | $0.9^{20} = 0.122$ | 12% |

**Multi-scale intuition.** With $H$ heads using $\gamma_h = 1 - 2^{-5-h}$:

| Head | $\gamma_h$ | Half-life (steps) |
|---|---|---|
| $h=1$ | $1 - 2^{-6} \approx 0.984$ | $\approx 43$ |
| $h=4$ | $1 - 2^{-9} \approx 0.998$ | $\approx 347$ |
| $h=8$ | $1 - 2^{-13} \approx 0.9999$ | $\approx 5678$ |

Different heads naturally focus on different temporal scales — short-range syntax to long-range topic coherence — without any explicit positional encoding.

---

## 10 · Where it's used / where it breaks

**Deployed or in active use:**

| Model | Method | Context |
|---|---|---|
| **RWKV-4/5/6** | RWKV WKV operator | Open-source 1.5B–14B models; used in edge/CPU inference |
| **Hyena-DNA** | Hyena convolution | Genomic sequence modeling; handles 1M+ base-pair contexts |
| **RetNet** (Microsoft Research) | Retention with three modes | Research model; influenced Mamba-2's SSD formulation |
| **Jamba** (AI21 Labs 2024) | Mamba (retention-adjacent SSM) | Production hybrid; Mamba + attention for long contexts |
| **H3** (Together AI 2023) | SSM-based, Hyena-adjacent | Pre-Mamba; demonstrated competitive perplexity |

**Where these methods break:**

1. **Associative recall / needle-in-a-haystack.** Geometric decay (RetNet, RWKV) attenuates older tokens. For γ=0.9, a token 50 steps back retains only $0.9^{50} \approx 0.5\%$ of its original signal. Sharp retrieval fails.

2. **In-context learning with many examples.** Adding 50 examples to context helps Transformers much more than RetNet/RWKV — the examples are compressed into the decaying state rather than directly accessible.

3. **Tasks requiring position-exact recall.** Repeating the exact content from position $k$ (copy tasks) requires the state to maintain that content without decay — provably impossible for fixed γ < 1 at large $k$.

4. **Short sequences (<256 tokens).** The FFT overhead in Hyena and the scan overhead in RWKV make them slower than FlashAttention at typical lengths. The crossover favoring Hyena is ~6K tokens.

---

## 11 · Cousins & alternatives

| Method | Mechanism | Training | Inference/token | Quality gap |
|---|---|---|---|---|
| **Standard Transformer** | Softmax attention | $O(N^2 d)$ | $O(Nd)$ (KV cache) | Reference |
| **FlashAttention** | IO-aware tiling | $O(N^2 d)$ | $O(Nd)$ | None (exact) |
| **Hyena** (this page) | FFT implicit convolution | $O(pN \log N)$ | $O(Nd)$ sequential | Moderate at <6K |
| **RetNet** (this page) | Retention with γ decay | $O(N^2)$ parallel | $O(d^2)$ recurrent | Small at scale |
| **RWKV** (this page) | WKV time-decay | $O(Nd^2)$ | $O(d)$ | Small at 7B+ |
| **Mamba** (Gu 2023) | Selective scan (input-dep. A) | $O(Nd)$ | $O(d_{\text{state}})$ | Small at scale |
| **Linear attention** (Katharopoulos) | ELU+1 feature map | $O(Nd^2)$ | $O(d^2)$ | Large |
| **Performers** (Choromanski) | FAVOR+ random features | $O(Nrd)$ | $O(rd)$ | Moderate |
| **S4** (Gu 2022) | Fixed-A HiPPO SSM | $O(N \log N)$ | $O(d_{\text{state}})$ | Moderate |
| **GLA** (Yang 2024) | Gated linear attention | $O(Nd^2)$ | $O(d^2)$ | Small |

**Key differentiators between Hyena, RetNet, and RWKV:**
- **Hyena** is primarily a training-efficiency story (FFT); inference is still sequential O(Nd).
- **RetNet** and **RWKV** provide both training parallelism AND O(1)-per-token inference — the more practically valuable property.
- **RWKV-6** adds data-dependent decay, making it the closest non-selective-scan method to Mamba.

---

## 12 · Interview drill

<details><summary><b>Q: Why does RetNet have three computation modes rather than one?</b></summary>

The three modes correspond to three engineering trade-offs. The parallel mode matches Transformer training throughput by using GPU parallelism over the full sequence. The recurrent mode provides $O(1)$ inference cost per token, critical for deployment. The chunkwise mode is a middle ground: within each chunk, computation is parallel (fast); across chunks, it is recurrent (memory efficient). Different deployment scenarios call for different modes.
</details>

<details><summary><b>Q: RWKV claims linear complexity — what is linear in what?</b></summary>

During inference, RWKV is $O(d)$ per token — linear in the model dimension, independent of sequence length. During training, the cost is $O(Td^2)$ — linear in sequence length $T$ (dominated by the $d \times d$ weight matrix multiplications, which scale the same way as a Transformer's FFN, but the sequence scan is $O(T)$ rather than $O(T^2)$).
</details>

<details><summary><b>Q: Where does Hyena fail compared to attention?</b></summary>

Hyena struggles on tasks requiring sharp associative recall — retrieving a specific key-value pair from a long context. The FFT convolution is a fixed linear operation; it cannot route information as flexibly as the data-dependent attention matrix. On copying or induction-head tasks, Hyena lags behind Transformers, especially at small model sizes.
</details>

<details><summary><b>Q: What is the crossover point between Hyena and FlashAttention?</b></summary>

Empirically, at sequence lengths below $\sim 6\text{k}$ tokens, highly optimized FlashAttention is faster. At $N = 100\text{k}$, Hyena is approximately 100× faster. The exact crossover depends on hardware and implementation.
</details>

<details><summary><b>Q: Why can't Hyena use standard convolution rather than FFT-based implicit convolution?</b></summary>

Standard convolution with an explicit filter of length $N$ costs $O(N^2)$ — the same as attention — because computing each of $N$ output positions requires summing over up to $N$ filter taps. Hyena avoids this by parameterizing the filter *implicitly* as the output of a small neural network evaluated at positional coordinates, then computing the convolution via FFT in $O(N \log N)$. The filter is "implicit" in the sense that it is never materialized as a length-$N$ vector; instead, FFT enables the circular convolution in the frequency domain where pointwise multiplication replaces the $O(N^2)$ sliding window sum. This is equivalent to what FlashFFT (Dao et al. 2023) computes for the Hyena operator.
</details>

<details><summary><b>Q: In RWKV, what role does the time bonus $u$ play, and why is it necessary?</b></summary>

The WKV operator computes $\text{wkv}_t = \frac{\sum_{i<t} e^{-(t-1-i)w + k_i} v_i + e^{u+k_t} v_t}{\sum_{i<t} e^{-(t-1-i)w + k_i} + e^{u+k_t}}$. The current-step term $e^{u+k_t} v_t$ (numerator) and $e^{u+k_t}$ (denominator) use the bonus $u$ rather than the time-decay $w$. This is necessary because applying $w = 0$ to the current step would be equivalent to treating the current token as if it arrived an infinite time ago (decay $e^0 = 1$), while using $w > 0$ (the decay for older tokens) would penalize it. The bonus $u$ is a learned parameter that allows the model to weight the current token appropriately regardless of the decay schedule for past tokens. Without $u$, the model would systematically under- or over-weight the present.
</details>

<details><summary><b>Q: What is the key architectural difference between RWKV-4 and RWKV-6, and why does it matter?</b></summary>

RWKV-4 uses a scalar time-decay $w \in \mathbb{R}^d$ (one decay value per channel, shared across positions). RWKV-6 (Finch) introduces **data-dependent recurrence**: $w_t$ and the mixing ratios are functions of the input $x_t$, computed via small linear interpolation (lerp) gates. This is analogous to the step from fixed-A SSMs (S4) to Mamba's input-dependent $\bar{A}$. The practical impact: RWKV-6 can selectively retain or discard information based on content (e.g., retain entity names, discard function words), closing much of the associative recall gap while remaining recurrent at inference time. Benchmark scores show RWKV-6 matching or exceeding Mamba on language modeling at equivalent parameter counts.
</details>

---

## 13 · Common misconceptions

| Misconception | Reality |
|---|---|
| "RetNet is always $O(1)$ complexity." | $O(1)$ per token applies only to the recurrent mode. Parallel training mode is $O(N^2)$. |
| "RWKV cannot be parallelized." | Training uses a parallel prefix scan (like a cumulative sum), which is GPU-parallelizable. |
| "Hyena is $O(N \log N)$ during inference." | Inference runs the recurrence sequentially; only training benefits from FFT-based parallelism. |
| "These models match Transformers on all tasks." | On associative recall and precise retrieval tasks, softmax attention retains a significant quality advantage. |
| "RWKV-4 is the latest version." | As of 2024, RWKV-5 (Eagle) and RWKV-6 (Finch) with matrix-valued states and data-dependent recurrence are the current versions (arXiv:2404.05892). |

---

## 14 · One-screen summary

> **Hyena:** $O(N \log N)$ via implicit FFT convolutions; best for very long sequence training. **RetNet:** Three computation modes (parallel/recurrent/chunked); $O(N^2)$ training, $O(1)$ inference per token via geometric decay state. **RWKV:** WKV time-decay operator; parallelizable at training ($O(Td^2)$), true $O(d)$ RNN at inference. All three sacrifice some associative-recall quality for inference efficiency. Choose based on deployment constraints.
>
> **Interview rule of thumb:** For sequences up to ~6K tokens at training time, FlashAttention is simpler and competitive. Beyond ~6K and especially for streaming/edge inference, RetNet (recurrent mode) or RWKV eliminate the growing KV cache. For ultra-long audio or genomics (>100K), Hyena's $O(N \log N)$ training is compelling. Hybrid (attention + SSM) is the current production best-practice.

---

## 15 · References

1. **Poli, M., Massaroli, S., Nguyen, E., Fu, D.Y., Dao, T., Baccus, S., Bengio, Y., Ermon, S., Ré, C.** "Hyena Hierarchy: Towards Larger Convolutional Language Models." ICML 2023. arXiv:2302.10866. [https://arxiv.org/abs/2302.10866](https://arxiv.org/abs/2302.10866)

2. **Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., Wei, F.** "Retentive Network: A Successor to Transformer for Large Language Models." arXiv:2307.08621, July 2023. [https://arxiv.org/abs/2307.08621](https://arxiv.org/abs/2307.08621)

3. **Peng, B., et al.** "RWKV: Reinventing RNNs for the Transformer Era." EMNLP 2023 Findings. arXiv:2305.13048. [https://arxiv.org/abs/2305.13048](https://arxiv.org/abs/2305.13048)

4. **Peng, B., et al.** "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence." arXiv:2404.05892, April 2024. [https://arxiv.org/abs/2404.05892](https://arxiv.org/abs/2404.05892) — RWKV-5 and RWKV-6.

5. **Gu, A., Dao, T.** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023. — Related SSM for comparison.

---

<!-- ============================ BOTTOM NAV ============================ -->
<div align="center">

[⬅️ Q29 — Differential Transformer](./q29-differential-transformer.md) &nbsp;|&nbsp; [📚 Back to Section 1](./README.md) &nbsp;|&nbsp; [🏠 Home](../../README.md) &nbsp;|&nbsp; [Q31 — Softmax-Free Attention ➡️](./q31-softmax-free-attention.md)

<sub>Found an error or have a sharper intuition? See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a> — answers follow the <a href="../../_TEMPLATE.md">answer template</a>.</sub>

</div>
