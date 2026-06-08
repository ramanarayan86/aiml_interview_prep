<div align="center">

[🏠 Home](./README.md)

</div>

---

# Glossary

A shared reference for terms used across sections. Each entry gives a one-paragraph definition and links to the question where the concept is covered in depth. Alphabetical order.

> **How to use:** `Ctrl/Cmd + F` and search the term. The link takes you to the full first-principles treatment. Terms that appear in multiple sections link to the primary coverage.

---

## A

**Attention entropy collapse**
The failure mode where attention logits grow so large that the softmax saturates to a near one-hot distribution — one position gets all the weight, the rest near zero. Entropy of the row approaches 0. Gradients through the softmax vanish, the loss spikes, and training diverges. Fixed by QK-norm or logit soft-capping.
→ Deep dive: [Q18 — QK-norm](./questions/section-01-transformer-architecture/q18-qk-norm.md)

**Attention sink**
A phenomenon where the first token (often `<bos>`) accumulates a disproportionate fraction of attention weight across all layers, even when semantically irrelevant. Acts as a "garbage collector" for softmax probability mass that must sum to 1. Exploited by StreamingLLM to extend generation beyond training context length without full recomputation.
→ Deep dive: [Q25 — Attention Sinks](./questions/section-01-transformer-architecture/q25-attention-sinks.md)

**ALiBi (Attention with Linear Biases)**
A position encoding method that adds a fixed negative slope × distance penalty directly to attention logits (no learned position embeddings). Allows length extrapolation beyond training context. Slope per head controls locality.
→ Deep dive: [Q20 — ALiBi](./questions/section-01-transformer-architecture/q20-alibi.md)

---

## B

**BPE (Byte-Pair Encoding)**
A subword tokenization algorithm that starts from characters (or bytes) and iteratively merges the most frequent adjacent pair, building a vocabulary of up to V entries. Greedy, deterministic at encoding time, and OOV-free at the byte level. Used by GPT-2/3/4, Llama, Mistral.
→ Deep dive: [Q2-02 — BPE](./questions/section-02-tokenization-and-embeddings/q02-bpe.md)

**Byte-level BPE**
A variant of BPE where the base vocabulary is the 256 possible UTF-8 byte values. Any Unicode string is representable as a byte sequence, making the tokenizer truly OOV-free — no `[UNK]` token is ever emitted. Used by GPT-2 and all Llama variants.
→ Deep dive: [Q2-05 — Byte-level BPE](./questions/section-02-tokenization-and-embeddings/q05-byte-level-bpe.md)

---

## C

**Causal masking**
An upper-triangular mask applied to attention logits so that position $i$ can only attend to positions $\leq i$. Enforces the autoregressive property: the model cannot "see the future" during training or inference. Implemented as `masked_fill(-inf)` before softmax.
→ Deep dive: [Q8 — Causal Masking](./questions/section-01-transformer-architecture/q08-causal-masking.md)

**Context window**
The maximum number of tokens a model can process in a single forward pass. Determined by the positional encoding scheme and the KV-cache memory budget. Effective context (in words) varies with token fertility — a 128K-token window holds far fewer words for high-fertility languages.
→ Related: [Q5 — Positional Encodings](./questions/section-01-transformer-architecture/q05-positional-encodings.md), [Q2-07 — Token Fertility](./questions/section-02-tokenization-and-embeddings/q07-token-fertility.md)

---

## D

**Differential attention**
A Transformer variant that computes two softmax attention maps and takes their difference, cancelling common noise and amplifying signal. Uses pairs of Q,K projections per head with a learnable scalar λ. Reduces attention noise with no extra parameters at inference.
→ Deep dive: [Q29 — Differential Transformer](./questions/section-01-transformer-architecture/q29-differential-transformer.md)

---

## E

**Embedding layer**
A trainable lookup table $E \in \mathbb{R}^{V \times d}$ that maps discrete token IDs (integers) to dense float vectors. The forward pass is a row-select (no FLOPs), not a matrix multiply. Often weight-tied with the output un-embedding matrix to halve parameter count.
→ Deep dive: [Q2-06 — Embedding Layer](./questions/section-02-tokenization-and-embeddings/q06-embedding-layer.md)

**Entropy collapse** → See *Attention entropy collapse*

**Exposure bias**
The training/inference mismatch in autoregressive models: during training the model conditions on ground-truth previous tokens (teacher forcing), but at inference it conditions on its own (potentially wrong) previous outputs. Errors accumulate, producing repetition or topic drift in long generations.
→ Deep dive: [Q7 — Teacher Forcing & Exposure Bias](./questions/section-01-transformer-architecture/q07-teacher-forcing.md)

---

## F

**FAVOR+**
Fast Attention Via positive Orthogonal Random features. An approximation to softmax attention using random feature maps that decompose the kernel into a low-rank product, reducing attention from $O(N^2 d)$ to $O(N d^2)$. The basis for the Performer architecture.
→ Deep dive: [Q33 — Attention as a Kernel Method](./questions/section-01-transformer-architecture/q33-attention-kernel-method.md)

**Feed-forward network (FFN)**
The two-layer MLP block after each attention sublayer in a Transformer. Hidden dimension is typically 4× the model dimension ($d_{ff} \approx 4d$). Interpeted as key-value memory, where keys are the first layer weights and values are second layer weights. SwiGLU/GeGLU activations replaced ReLU in modern LLMs.
→ Deep dive: [Q6 — FFN](./questions/section-01-transformer-architecture/q06-ffn.md), [Q15 — SwiGLU/GeGLU](./questions/section-01-transformer-architecture/q15-swiglu-geglu.md)

**Fertility** → See *Token fertility*

**FlashAttention**
An IO-aware exact attention algorithm that tiles Q, K, V to stay in SRAM, computing attention block-by-block. Avoids materializing the full $N \times N$ attention matrix in HBM, reducing memory from $O(N^2)$ to $O(N)$ and runtime by 2–4× on modern GPUs. V2 and V3 added further optimizations for MHA/MQA and Hopper hardware.
→ Deep dive: [Q24 — FlashAttention](./questions/section-01-transformer-architecture/q24-flash-attention.md)

---

## G

**GeGLU / SwiGLU**
Gated activation functions for the FFN block that apply an element-wise gate: $\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV)$. The gate selects which dimensions of the hidden state to pass through, improving gradient flow and empirical performance over plain GELU/ReLU. Used in GPT-4, Llama, PaLM.
→ Deep dive: [Q15 — SwiGLU/GeGLU](./questions/section-01-transformer-architecture/q15-swiglu-geglu.md)

**GQA (Grouped Query Attention)**
A variant of multi-head attention where $G$ groups of query heads share one key-value head. Reduces KV-cache memory by factor $H/G$ vs MHA, with minimal quality degradation. MQA is the extreme case ($G=1$). Used in Llama 2/3, Mistral, Gemma.
→ Deep dive: [Q13 — GQA & MQA](./questions/section-01-transformer-architecture/q13-gqa-mqa.md)

**Glitch tokens**
Tokens in BPE vocabularies that map to unexpected or garbage strings, often because they occurred in training data in weird contexts (e.g., whitespace-only tokens, Reddit usernames, repeated Unicode characters). Their embeddings are poorly trained and can cause erratic model behaviour. `SolidGoldMagikarp` is the canonical example.
→ Deep dive: [Q2-17 — Glitch Tokens](./questions/section-02-tokenization-and-embeddings/README.md) *(coming soon)*

---

## H

**Hyena**
A sub-quadratic attention alternative that replaces the attention matrix with long convolutions parameterized by a Hyena operator — a product of diagonals, permutations, and pointwise nonlinearities. Achieves $O(N \log N)$ complexity and can match Transformer quality on language tasks at scale.
→ Deep dive: [Q30 — Hyena, RetNet, RWKV](./questions/section-01-transformer-architecture/q30-hyena-retnet-rwkv.md)

---

## K

**KV-cache**
During autoregressive inference, the key and value tensors from each attention head are cached for all previously generated tokens. On each new step, only the new query is computed; keys and values are read from the cache. Memory cost is $2 \times B \times S \times H \times d_h \times \text{bytes}$. GQA/MQA reduce this proportionally.
→ Related: [Q13 — GQA/MQA](./questions/section-01-transformer-architecture/q13-gqa-mqa.md)

---

## L

**LayerNorm**
Normalizes activations across the feature dimension (not the batch dimension): $\text{LN}(x) = \gamma \frac{x - \mu}{\sigma + \epsilon} + \beta$. Used instead of BatchNorm in Transformers because it doesn't depend on batch statistics, making it well-behaved at sequence level.
→ Deep dive: [Q3 — LayerNorm vs BatchNorm](./questions/section-01-transformer-architecture/q03-layernorm-batchnorm.md)

**Linear attention**
A family of attention approximations that factorize the softmax kernel as $\phi(q)^\top \phi(k)$, allowing the computation to be reordered to $O(Nd^2)$ instead of $O(N^2d)$. The Performer uses FAVOR+ random features as $\phi$.
→ Deep dive: [Q26 — Sub-quadratic Attention](./questions/section-01-transformer-architecture/q26-subquadratic-attention.md), [Q33 — Attention as Kernel](./questions/section-01-transformer-architecture/q33-attention-kernel-method.md)

---

## M

**Mamba / SSM**
State Space Models (SSMs) that replace attention with learned recurrences over a latent state. Mamba adds input-selective parameters (S4 → S6) so the state transition depends on the current input — making it "selective". Mamba-2 (SSD) unifies SSMs with linear attention under a single framework.
→ Deep dive: [Q27 — SSMs and Mamba](./questions/section-01-transformer-architecture/q27-ssm-mamba.md)

**MFU (Model FLOP Utilization)**
The fraction of peak theoretical GPU FLOP/s actually used during training: $\text{MFU} = \frac{\text{observed throughput (tokens/s)} \times \text{FLOPs per token}}{\text{GPU peak FLOP/s}}$. A well-tuned LLM training run achieves 40–60% MFU; anything below 30% suggests a bottleneck (memory bandwidth, communication, or kernel inefficiency).

**Multi-head attention (MHA)**
The standard attention mechanism with $H$ independent query/key/value projection triples, each operating in $d_h = d/H$ dimensional space. Each head can specialize in different relationship types (induction, copying, positional). Outputs are concatenated and projected.
→ Deep dive: [Q9 — Multi-head Attention](./questions/section-01-transformer-architecture/q09-multi-head-attention.md), [Q12 — MHA Interpretability](./questions/section-01-transformer-architecture/q12-mha-interpretability.md)

---

## P

**Pre-norm / Post-norm**
The placement of LayerNorm relative to the sublayer (attention or FFN). Pre-norm: LN before the sublayer (residual stream is clean); post-norm: LN after the add (original "Attention is All You Need"). Pre-norm dominates modern LLMs due to better gradient flow at depth.
→ Deep dive: [Q4 — Pre-norm vs Post-norm](./questions/section-01-transformer-architecture/q04-prenorm-postnorm.md)

---

## Q

**QK-norm**
Normalizes query and key vectors (to unit length via ℓ₂, or via RMSNorm/LayerNorm over the head dimension) before computing the attention dot product. Replaces uncontrolled logit growth (attention entropy collapse) with a bounded cosine score plus a learnable temperature scalar $g$.
→ Deep dive: [Q18 — QK-norm ⭐](./questions/section-01-transformer-architecture/q18-qk-norm.md)

---

## R

**RetNet**
A recurrent/parallel dual-mode architecture where attention is replaced by a retention mechanism with an exponential decay bias. Supports $O(N)$ parallel training (like Transformers) and $O(1)$ per-step recurrent inference (like RNNs). State-space interpretation exists.
→ Deep dive: [Q30 — Hyena, RetNet, RWKV](./questions/section-01-transformer-architecture/q30-hyena-retnet-rwkv.md)

**RMSNorm**
A simplified LayerNorm that omits the mean-subtraction step: $\text{RMS}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}}$. Cheaper to compute (no mean), empirically matches LayerNorm quality. Used in Llama, Mistral, GPT-NeoX.
→ Deep dive: [Q14 — RMSNorm vs LayerNorm](./questions/section-01-transformer-architecture/q14-rmsnorm-vs-layernorm.md)

**RoPE (Rotary Position Embedding)**
A relative position encoding that multiplies query and key vectors by a rotation matrix whose angle depends on position. Dot products between rotated vectors depend only on the relative offset — giving built-in relative position awareness without explicit relative bias. Long-context extensions: NTK-aware, YaRN, LongRoPE.
→ Deep dive: [Q23 — RoPE Derivation](./questions/section-01-transformer-architecture/q23-rope-derivation.md), [Q19 — Position Encodings](./questions/section-01-transformer-architecture/q19-position-encodings.md)

**RWKV**
A hybrid architecture combining a token-shift (time-mixing) and channel-mixing operator that yields $O(N)$ inference like an RNN but can be trained in parallel like a Transformer. The WKV operator is a key-weighted cumulative sum with exponential decay.
→ Deep dive: [Q30 — Hyena, RetNet, RWKV](./questions/section-01-transformer-architecture/q30-hyena-retnet-rwkv.md)

---

## S

**Scaled dot-product attention**
$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$. The $1/\sqrt{d_k}$ factor prevents dot products from growing with dimension (which would saturate the softmax). The core operation of all Transformer variants.
→ Deep dive: [Q2 — Scaled Attention](./questions/section-01-transformer-architecture/q02-scaled-attention.md)

**SentencePiece**
A language-agnostic tokenization library (Google) that implements both BPE and Unigram LM without requiring pre-tokenization (whitespace splitting). Treats input as a raw byte stream — essential for languages without clear word boundaries (CJK, Thai). Used by T5, mBART, Llama 1/2.
→ Related: [Q2-04 — Unigram LM](./questions/section-02-tokenization-and-embeddings/q04-unigram-lm.md), [Q2-09 — SentencePiece](./questions/section-02-tokenization-and-embeddings/README.md) *(coming soon)*

**Subword tokenization**
A family of tokenization algorithms (BPE, WordPiece, Unigram LM) that split text into units between full words and individual characters. Provides a compact fixed vocabulary with OOV protection and morphological regularity.
→ Deep dive: [Q2-01 — What is a token?](./questions/section-02-tokenization-and-embeddings/q01-what-is-a-token.md)

---

## T

**Teacher forcing**
A training strategy for autoregressive models where the ground-truth token at position $t$ is fed as input at step $t+1$, instead of the model's own prediction. Speeds up training and stabilizes gradients, but creates an exposure bias mismatch at inference time.
→ Deep dive: [Q7 — Teacher Forcing](./questions/section-01-transformer-architecture/q07-teacher-forcing.md)

**tiktoken**
OpenAI's open-source BPE tokenizer library (Rust backend). Uses a regex pre-tokenization step to split text before BPE merges, preventing merges across word boundaries or certain punctuation. Used for GPT-3.5/4 (`cl100k_base`) and GPT-4o (`o200k_base`).
→ Related: [Q2-05 — Byte-level BPE](./questions/section-02-tokenization-and-embeddings/q05-byte-level-bpe.md), [Q2-21 — tiktoken](./questions/section-02-tokenization-and-embeddings/README.md) *(coming soon)*

**Token**
The atomic unit a language model processes — typically a subword string produced by a tokenization algorithm. One word is usually 1–4 tokens. The model never sees raw text; it sees integer token IDs looked up in an embedding table.
→ Deep dive: [Q2-01 — What is a token?](./questions/section-02-tokenization-and-embeddings/q01-what-is-a-token.md)

**Token fertility**
The number of tokens a tokenizer produces per word (or per character) for a given language. A tokenizer trained on English-heavy data assigns low fertility (~1.3 tokens/word) to English but high fertility (4–8 tokens/word) to low-resource scripts. High fertility means higher API cost, shorter effective context, and reduced training signal density.
→ Deep dive: [Q2-07 — Token Fertility](./questions/section-02-tokenization-and-embeddings/q07-token-fertility.md)

---

## U

**Unigram LM tokenizer**
A tokenization algorithm (Kudo, 2018) that starts with a large seed vocabulary (all substrings) and iteratively prunes tokens whose removal minimizes corpus log-likelihood loss. Uses EM for probability estimation and Viterbi for encoding. Supports stochastic segmentation for data augmentation. Implemented in SentencePiece, used by T5 and mBART.
→ Deep dive: [Q2-04 — Unigram LM](./questions/section-02-tokenization-and-embeddings/q04-unigram-lm.md)

---

## V

**Viterbi algorithm**
Dynamic programming over a lattice to find the most probable path (sequence of tokens) under a probabilistic model. Used by Unigram LM tokenizer for encoding: $\alpha[j] = \max_{i<j} (\alpha[i] + \log p(w[i:j]))$. $O(|w|^2)$ per word.
→ Related: [Q2-04 — Unigram LM](./questions/section-02-tokenization-and-embeddings/q04-unigram-lm.md)

---

## W

**Weight tying**
Sharing the embedding table $E \in \mathbb{R}^{V \times d}$ with the output un-embedding matrix: $\text{logits} = h \cdot E^\top$. Halves the embedding parameter count, enforces geometric consistency between input and output token representations. Used in BERT, GPT-2, Llama.
→ Deep dive: [Q16 — Weight Tying](./questions/section-01-transformer-architecture/q16-weight-tying.md), [Q2-06 — Embedding Layer](./questions/section-02-tokenization-and-embeddings/q06-embedding-layer.md)

**WordPiece**
A subword tokenization algorithm that merges the pair maximizing $\frac{\text{count}(AB)}{\text{count}(A) \cdot \text{count}(B)}$ — a likelihood ratio — rather than BPE's raw count. Produces more linguistically informative merges. Uses `##` prefix for continuation tokens. Used by BERT, DistilBERT, ALBERT.
→ Deep dive: [Q2-03 — WordPiece](./questions/section-02-tokenization-and-embeddings/q03-wordpiece.md)

---

<div align="center">

[🏠 Home](./README.md) &nbsp;|&nbsp; [📄 Template](./_TEMPLATE.md) &nbsp;|&nbsp; [🤝 Contributing](./CONTRIBUTING.md)

<sub>Missing a term? Add it via the standard PR process — one term per paragraph, alphabetical placement, link to the primary question page.</sub>

</div>
