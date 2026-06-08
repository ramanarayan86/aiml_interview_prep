<div align="center">

[🏠 Home](./README.md)

</div>

---

# Study Plans

Three structured study plans tailored to different interview types and time budgets. Each plan tells you exactly which questions to read, in what order, and why.

> **How to use:** Pick the plan that matches your interview, follow the sequence, and use the [Cheat Sheet](./CHEATSHEET.md) the morning of the interview as a rapid review.

---

## Plan A — ML Engineer (48-hour sprint)

**Target:** Senior Software Engineer / ML Engineer roles at tech companies. Focus: can you implement and debug LLM components in production?

**Time budget:** 2 full days (≈ 12 hours of focused reading)

### Day 1 — Architecture fundamentals (6 hours)

| Time | Question | Why |
|------|----------|-----|
| 45 min | [Q1 — Transformer block](./questions/section-01-transformer-architecture/q01-transformer-block.md) | Every LLM interview starts here |
| 30 min | [Q2 — √d_k scaling](./questions/section-01-transformer-architecture/q02-scaled-attention.md) | "Why do we divide by √d_k" — guaranteed question |
| 30 min | [Q3 — LayerNorm vs BatchNorm](./questions/section-01-transformer-architecture/q03-layernorm-batchnorm.md) | Normalization is everywhere; understand the why |
| 30 min | [Q4 — Pre-norm vs Post-norm](./questions/section-01-transformer-architecture/q04-prenorm-postnorm.md) | Every modern LLM uses pre-norm; know why |
| 45 min | [Q5 — Positional encodings](./questions/section-01-transformer-architecture/q05-positional-encodings.md) | Sinusoidal → RoPE → ALiBi progression |
| 30 min | [Q6 — FFN role](./questions/section-01-transformer-architecture/q06-ffn.md) | SwiGLU, 4× expansion, key-value memory view |
| 30 min | [Q8 — Causal masking](./questions/section-01-transformer-architecture/q08-causal-masking.md) | Implementation detail every MLE must know |
| 30 min | [Q9 — Multi-head attention](./questions/section-01-transformer-architecture/q09-multi-head-attention.md) | Head specialization, induction heads |
| 30 min | [Q11 — Attention complexity](./questions/section-01-transformer-architecture/q11-attention-complexity.md) | O(n²) memory and compute — when does it hurt? |
| 45 min | [Q13 — GQA & MQA](./questions/section-01-transformer-architecture/q13-gqa-mqa.md) | KV-cache cost, why Llama 2/3 uses GQA |
| 30 min | [Q14 — RMSNorm](./questions/section-01-transformer-architecture/q14-rmsnorm-vs-layernorm.md) | Why modern LLMs dropped the mean subtraction |

### Day 2 — Tokenization + Production (6 hours)

| Time | Question | Why |
|------|----------|-----|
| 45 min | [Q2-01 — What is a token?](./questions/section-02-tokenization-and-embeddings/q01-what-is-a-token.md) | Tokenization basics — surprisingly deep |
| 45 min | [Q2-02 — BPE](./questions/section-02-tokenization-and-embeddings/q02-bpe.md) | The dominant tokenizer; must know the algorithm |
| 30 min | [Q2-06 — Embedding layer](./questions/section-02-tokenization-and-embeddings/q06-embedding-layer.md) | V×d, weight tying, sparse gradients |
| 30 min | [Q2-07 — Token fertility](./questions/section-02-tokenization-and-embeddings/q07-token-fertility.md) | API cost, context window, multilingual fairness |
| 45 min | [Q24 — FlashAttention](./questions/section-01-transformer-architecture/q24-flash-attention.md) | IO-awareness — common in systems-adjacent interviews |
| 30 min | [Q18 — QK-norm ⭐](./questions/section-01-transformer-architecture/q18-qk-norm.md) | Entropy collapse, training stability |
| 45 min | [Q15 — SwiGLU/GeGLU](./questions/section-01-transformer-architecture/q15-swiglu-geglu.md) | Gating in FFN — in every modern LLM |
| 30 min | Review [Cheat Sheet §1](./CHEATSHEET.md#section-1--transformer-architecture) | Lock in key equations before sleep |

**Morning of interview:** Read the [Cheat Sheet](./CHEATSHEET.md) (30 min).

---

## Plan B — Research Scientist (1-week deep dive)

**Target:** Research scientist / research engineer roles at frontier labs (Anthropic, OpenAI, Google DeepMind, Meta AI). Focus: first-principles understanding, ability to derive, extend, and critique.

**Time budget:** 5 days (≈ 30–35 hours total)

### Day 1 — Architecture core (6 hours)

Cover all of Section 1 Basic + Intermediate: Q1–Q22 in order. Focus on derivations, not just intuitions.

| Block | Questions | Focus |
|-------|----------|-------|
| Morning | Q1–Q6 | Forward pass, attention math, norms, position, FFN |
| Afternoon | Q7–Q10 | Teacher forcing, masking, MHA, architectures |
| Evening | Q11–Q16 | Complexity, GQA, RMSNorm, SwiGLU, weight tying |

### Day 2 — Architecture advanced (6 hours)

| Block | Questions | Focus |
|-------|----------|-------|
| Morning | Q17–Q22 | Dropout, QK-norm, RoPE, ALiBi, parallel block, no-bias |
| Afternoon | Q23–Q26 | RoPE derivation (derive it!), FlashAttention, attention sinks, sub-quadratic |
| Evening | Q27–Q30 | Mamba/SSM, expressivity gap, differential Transformer, Hyena/RetNet/RWKV |

### Day 3 — Architecture frontier + deep Tokenization (6 hours)

| Block | Questions | Focus |
|-------|----------|-------|
| Morning | Q31–Q34 | Softmax-free, Turing completeness, FAVOR+, Graph Transformers |
| Afternoon | Q2-01–Q2-05 | Token, BPE, WordPiece, Unigram LM, byte-level |
| Evening | Q2-06–Q2-07 | Embedding layer, fertility + multilingual implications |

### Day 4 — Systems depth (6 hours)

Self-study the following from external sources (these sections are not yet written):
- KV-cache mechanics, continuous batching, speculative decoding (§6)
- LoRA, QLoRA, PEFT adapter architectures (§17)
- Chinchilla scaling laws, compute-optimal training (§3)
- DPO, PPO, RLHF pipeline (§4)

### Day 5 — Synthesis + frontier review (4 hours)

| Block | Activity |
|-------|---------|
| Morning | Re-derive: RoPE rotation, BPE training loop, Viterbi for Unigram LM, selective scan recurrence |
| Afternoon | Practice explaining Q18 (QK-norm) and Q27 (Mamba) to an imaginary 12-year-old |
| Evening | Read [Cheat Sheet](./CHEATSHEET.md) end-to-end |

**Morning of interview:** Open Cheat Sheet §1 and §2. Run through the one-liners. Then close it.

---

## Plan C — Applied Scientist / MLE-Applied (3-day focused plan)

**Target:** Applied scientist, LLM product engineer, ML platform engineer at companies building with LLMs. Focus: system design, RAG, production tradeoffs, debugging.

**Time budget:** 3 days (≈ 18 hours)

### Day 1 — Architecture essentials only (5 hours)

| Time | Question | Why |
|------|----------|-----|
| 45 min | [Q1 — Transformer block](./questions/section-01-transformer-architecture/q01-transformer-block.md) | Baseline — every interviewer assumes this |
| 30 min | [Q2 — √d_k scaling](./questions/section-01-transformer-architecture/q02-scaled-attention.md) | Quick win — clean derivation |
| 30 min | [Q13 — GQA/MQA](./questions/section-01-transformer-architecture/q13-gqa-mqa.md) | KV-cache, throughput — production relevant |
| 30 min | [Q11 — Complexity](./questions/section-01-transformer-architecture/q11-attention-complexity.md) | When does O(n²) actually bite? |
| 45 min | [Q24 — FlashAttention](./questions/section-01-transformer-architecture/q24-flash-attention.md) | IO-awareness — "what is FlashAttention and why?" |
| 30 min | [Q5 — Positional encodings](./questions/section-01-transformer-architecture/q05-positional-encodings.md) | RoPE/ALiBi in production context |
| 30 min | [Q18 — QK-norm](./questions/section-01-transformer-architecture/q18-qk-norm.md) | Training stability — useful for fine-tuning context |

### Day 2 — Tokenization + Production Systems (6 hours)

| Time | Question | Why |
|------|----------|-----|
| 45 min | [Q2-01 — What is a token?](./questions/section-02-tokenization-and-embeddings/q01-what-is-a-token.md) | Tokenization = hidden cost driver |
| 45 min | [Q2-02 — BPE](./questions/section-02-tokenization-and-embeddings/q02-bpe.md) | Algorithm behind every LLM tokenizer |
| 30 min | [Q2-05 — Byte-level BPE](./questions/section-02-tokenization-and-embeddings/q05-byte-level-bpe.md) | OOV-free — multilingual products |
| 30 min | [Q2-07 — Token fertility](./questions/section-02-tokenization-and-embeddings/q07-token-fertility.md) | Cost/context window disparity by language |
| 45 min | Self-study: RAG architecture (retrieval → augmentation → generation pipeline) | Most common applied LLM system design |
| 45 min | Self-study: speculative decoding, prefix caching, continuous batching | "How do you reduce LLM serving latency?" |
| 30 min | Self-study: LoRA — what it is and how it reduces trainable parameters | Fine-tuning a must for applied roles |

### Day 3 — Design practice + review (4 hours)

| Block | Activity |
|-------|---------|
| Morning (2h) | Solve 2 design problems aloud: (1) Design a RAG pipeline for a legal document QA system. (2) A multilingual chatbot is 4× over API budget — investigation plan. |
| Afternoon (2h) | Read [Cheat Sheet](./CHEATSHEET.md) §1 + §2; note anything you can't reproduce |

**Morning of interview:** Cheat Sheet (20 min). Focus on equations for Q1, Q2, Q13.

---

## Quick-reference: questions by interview frequency

Based on patterns across ML/AI interview reports (2023–2025):

| Frequency | Questions |
|-----------|-----------|
| **Almost always asked** | Q1 (Transformer block), Q2 (√d_k), Q5 (position encoding), Q11 (complexity), Q2-02 (BPE), Q2-01 (what is a token) |
| **Often asked at senior+** | Q13 (GQA), Q14 (RMSNorm), Q15 (SwiGLU), Q24 (FlashAttention), Q18 (QK-norm), Q2-07 (fertility) |
| **Advanced / research roles** | Q23 (RoPE derivation), Q27 (Mamba), Q28 (expressivity gap), Q33 (FAVOR+), Q2-04 (Unigram LM) |
| **Applied / systems roles** | Q40 (NaN debugging), Q35 (MFU), speculative decoding, RAG, LoRA |

---

<div align="center">

[🏠 Home](./README.md) &nbsp;|&nbsp; [📋 Cheat Sheet](./CHEATSHEET.md) &nbsp;|&nbsp; [📖 Glossary](./GLOSSARY.md) &nbsp;|&nbsp; [🗺️ Roadmap](./ROADMAP.md)

</div>
