<div align="center">

[🏠 Home](./README.md)

</div>

---

# Cheat Sheet

One-liner answers for every answered question. Read this the morning of the interview — not the night before.

> **For depth:** click the question link. This page is designed to jog memory, not teach concepts from scratch.

---

## Section 1 — Transformer Architecture

### Basic (Q1–Q10)

| # | Question | One-liner |
|---|----------|-----------|
| Q1 | What is a Transformer block? | Residual of (self-attn + LN) followed by residual of (FFN + LN); repeated L times |
| Q2 | Why scale by √d_k? | Dot products grow with d_k → softmax saturates → gradients vanish; divide to keep variance ≈ 1 |
| Q3 | LayerNorm vs BatchNorm? | LN normalizes across features (independent of batch size); BN normalizes across batch (fails for variable-length sequences) |
| Q4 | Pre-norm vs post-norm? | Pre-norm (LN before sublayer) = cleaner residual stream, more stable at depth; post-norm = original paper, finicky to train |
| Q5 | Positional encodings? | Sinusoidal → learned abs → relative (RoPE, ALiBi); RoPE encodes relative offset in the attention dot product via rotation |
| Q6 | FFN role? | Two-layer MLP (4× expansion) acting as key-value associative memory; SwiGLU gate filters which dimensions matter |
| Q7 | Teacher forcing? | Feed ground-truth token at step t+1 during training (not the model's own prediction) — fast, but creates exposure bias |
| Q8 | Causal masking? | Upper-triangular −∞ mask on attention logits before softmax; prevents position i from attending to j > i |
| Q9 | Multi-head attention? | H independent attention heads each with d/H dimensions; outputs concatenated then projected |
| Q10 | Encoder-decoder vs decoder-only? | Enc-dec separates understanding (enc) from generation (dec with cross-attn); decoder-only simpler, more scalable, dominant today |

### Intermediate (Q11–Q22)

| # | Question | One-liner |
|---|----------|-----------|
| Q11 | Attention complexity? | O(N²d) time, O(N²) memory for the attention matrix; N is sequence length |
| Q12 | What do different heads learn? | Induction, copying, positional, syntactic heads; most attention weight concentrates in a few heads |
| Q13 | GQA & MQA? | GQA: G groups of Q-heads share 1 KV-head; MQA = extreme (G=1); reduces KV-cache by H/G with little quality loss |
| Q14 | RMSNorm? | LayerNorm minus mean subtraction: γ · x / RMS(x); cheaper and matches LN quality |
| Q15 | SwiGLU / GeGLU? | Gate FFN: Swish(xW) ⊙ (xV); gate selects active dimensions, improves gradient flow |
| Q16 | Weight tying? | Share embedding E ∈ ℝ^(V×d) with output un-embedding (logits = h · Eᵀ); halves params, enforces representation consistency |
| Q17 | Dropout in LLMs? | Attention dropout masks random QKᵀ entries (rows → 0); residual dropout on sublayer output; modern LLMs often use 0 dropout |
| Q18 | QK-norm? ⭐ | Normalize Q and K to unit sphere (or via RMSNorm) + learnable scalar; prevents attention entropy collapse (logit growth → softmax saturation) |
| Q19 | Positional encoding deep dive? | Sinusoidal: absolute, fixed; Learned abs: flexible but no extrapolation; RoPE: relative via rotation; ALiBi: distance penalty |
| Q20 | ALiBi? | Subtract m·(j−i) from attention logit for head with slope m; no position embeddings, good length extrapolation |
| Q21 | Parallel attention + FFN? | Compute attn and FFN in parallel and sum; saves one all-reduce per layer in tensor-parallel, 15% faster |
| Q22 | No-bias LLMs? | Biases add noise in distributed settings and rarely help at scale; Llama/Mistral use no bias in linear layers |

### Advanced (Q23–Q34)

| # | Question | One-liner |
|---|----------|-----------|
| Q23 | RoPE derivation? | Multiply q,k by rotation R(θ_i · pos); qᵀk depends only on relative offset Δpos, not absolute positions |
| Q24 | FlashAttention? | Tile Q,K,V into SRAM blocks; compute softmax online per tile; never write N×N matrix to HBM; O(N) memory, 2–4× faster |
| Q25 | Attention sinks? | First token accumulates outsized attention weight as a "probability dump"; exploited by StreamingLLM for infinite generation |
| Q26 | Sub-quadratic attention? | Linear attention: φ(q)ᵀφ(k) factorization; Sparse/local: attend to subset; Sliding window (Mistral); sliding + global |
| Q27 | Mamba / SSM? | Selective SSM: h_t = A(x_t)h_{t-1} + B(x_t)x_t, y_t = C(x_t)h_t; input-dependent transition → selectively remembers/forgets |
| Q28 | Expressivity gap? | Attention is Turing-complete but in practice length-generalization fails; transformers struggle with strict periodicity and counting |
| Q29 | Differential attention? | Two softmax attention maps subtracted → noise cancels, signal amplifies; pairs of Q,K per head, learnable λ |
| Q30 | Hyena / RetNet / RWKV? | Hyena: long convolutions as implicit attention; RetNet: retention = exponential decay + relative pos; RWKV: WKV cumulative sum with decay |
| Q31 | Softmax-free Transformers? | Replace softmax with linear ops (ReLU, L2 norm); avoids quadratic interaction; trade-off: expressivity loss |
| Q32 | Turing completeness? | Universally quantified (unlimited layers, infinite precision); in practice bounded depth+precision limits decidability |
| Q33 | FAVOR+ / Performer? | Random feature map φ so that E[φ(q)ᵀφ(k)] ≈ exp(qᵀk/√d); decomposes attention into O(Nd²) linear operation |
| Q34 | Graph Transformers? | Replace sequence positions with graph structure; edge features in attention bias; node/edge attention mechanisms |

---

## Section 2 — Tokenization & Embeddings

### Basic (Q2-01–Q2-07)

| # | Question | One-liner |
|---|----------|-----------|
| Q2-01 | What is a token? | Atomic unit of text (typically subword); model sees integer ID → embedding lookup; ~1.3 tokens/word in English |
| Q2-02 | BPE? | Start from characters; iteratively merge most-frequent adjacent pair; greedy deterministic encoding; used by GPT/Llama |
| Q2-03 | WordPiece? | Like BPE but merges pair maximizing count(AB)/(count(A)·count(B)) — likelihood ratio; `##` prefix for continuation; used by BERT |
| Q2-04 | Unigram LM? | Start from all substrings; EM to estimate probabilities; prune tokens that minimize loss increase; Viterbi for encoding |
| Q2-05 | Byte-level BPE? | Base vocab = 256 UTF-8 bytes; BPE merges on top; no UNK token ever emitted; used by GPT-2 and Llama |
| Q2-06 | Embedding layer? | Lookup table E ∈ ℝ^(V×d); forward = row-select (0 FLOPs); weight-tied with output un-embedding; sparse gradients in backward |
| Q2-07 | Token fertility? | Tokens/word for a given language+tokenizer; English ~1.3, Arabic/Thai/CJK 4–8×; high fertility → higher cost, shorter effective context |

---

## Key equations

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$$

$$\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV)$$

$$\text{RoPE}: q_m' = R_{\Theta,m} q_m, \quad (q_m')^\top (k_n') = f(\text{only } m-n)$$

$$\text{ALiBi}: a_{ij} = q_i^\top k_j - m \cdot (i - j)$$

$$\text{Unigram}: P(\mathbf{x}) = \prod_{i=1}^{M} p(x_i), \quad L = -\sum_s \log P(\mathbf{x}^{*(s)})$$

$$\text{BPE merge criterion}: \text{argmax}_{(A,B)} \text{count}(AB)$$

$$\text{WordPiece merge criterion}: \text{argmax}_{(A,B)} \frac{\text{count}(AB)}{\text{count}(A) \cdot \text{count}(B)}$$

---

<div align="center">

[🏠 Home](./README.md) &nbsp;|&nbsp; [📖 Glossary](./GLOSSARY.md) &nbsp;|&nbsp; [🗺️ Study Plans](./STUDY_PLANS.md)

</div>
