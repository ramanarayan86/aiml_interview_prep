<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 1 — Transformer Architecture](../section-01-transformer-architecture/README.md) &nbsp;|&nbsp; [Section 3 — Pretraining & Scaling Laws ➡️](../section-03-pretraining-and-scaling-laws/README.md)

</div>

---

# Section 2 · Tokenization and Embeddings

<div align="center">

![Questions](https://img.shields.io/badge/questions-32-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F32-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-32%2F32-6c4fe0)

</div>

Tokenization is the hidden tax on every LLM system. Get it wrong and your model pays it on every forward pass — in context waste, vocabulary skew, cross-lingual gaps, and adversarial fragility. This section covers the full stack: **how text becomes token IDs** (BPE, WordPiece, Unigram LM, byte-level), **what embeddings represent** and how they are trained, and the subtle failure modes — glitch tokens, fertility gaps, fertility-based pricing, and the security implications of tokenizer quirks — that separate practitioners from theorists.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (template stub, awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q2‑01 | [What is a token, and why do LLMs operate on tokens rather than characters or words?](./q01-what-is-a-token.md) | tokenization · vocabulary · subword | 📝 |
| Q2‑02 | [Explain Byte-Pair Encoding (BPE) from scratch — the merge algorithm, stopping criterion, and the role of the vocabulary size hyperparameter.](./q02-bpe.md) | BPE · vocabulary · merge-rules | 📝 |
| Q2‑03 | [How does WordPiece differ from BPE? What objective does it optimise and where is it used?](./q03-wordpiece.md) | WordPiece · BERT · likelihood | 📝 |
| Q2‑04 | [What is the Unigram Language Model tokenizer, and how does it differ from BPE in the way it selects the vocabulary?](./q04-unigram-lm.md) | Unigram · SentencePiece · EM | 📝 |
| Q2‑05 | [What is byte-level tokenization, and why do GPT-2 and Llama use it instead of character or word tokenization?](./q05-byte-level-bpe.md) | byte-level · GPT-2 · OOV-free | 📝 |
| Q2‑06 | [What is an embedding layer, and what is the relationship between the vocabulary size, embedding dimension, and the first weight matrix of a Transformer?](./q06-embedding-layer.md) | embeddings · weight-matrix · lookup | 📝 |
| Q2‑07 | [What is token fertility, and why does it matter for multilingual models, cost estimation, and cross-lingual fairness?](./q07-token-fertility.md) | fertility · multilingual · cost | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q2‑08 | [Walk through the BPE training and encoding algorithm end-to-end. What are the computational costs of building and applying the merge table?](./q08-bpe-algorithm.md) | BPE · training · encoding-complexity | 📝 |
| Q2‑09 | [How does SentencePiece differ from Hugging Face tokenizers? What does language-independent tokenization mean in practice?](./q09-sentencepiece.md) | SentencePiece · normalisation · whitespace | 📝 |
| Q2‑10 | [What are the tradeoffs between a small vocabulary (e.g. 32K) and a large vocabulary (e.g. 256K)? How does vocabulary size affect model quality, memory, and throughput?](./q10-vocab-size-tradeoff.md) | vocabulary-size · embedding-table · throughput | 📝 |
| Q2‑11 | [How is the embedding table initialised, and why is initialisation scale important for training stability?](./q11-embedding-init.md) | initialisation · scale · stability | 📝 |
| Q2‑12 | [What is weight tying between the input embedding and the output (un-embedding) matrix? When does it help and when does it hurt?](./q12-weight-tying-embeddings.md) | weight-tying · logits · tied-embeddings | 📝 |
| Q2‑13 | [How does tokenization interact with numbers and arithmetic? Why do LLMs struggle with multi-digit arithmetic and what tokenization choices make it worse?](./q13-tokenization-numbers.md) | numbers · arithmetic · digit-split | 📝 |
| Q2‑14 | [Explain the concept of token healing (prefix caching across tokenisation boundaries). What problem does it solve and how do vLLM and TGI handle it?](./q14-token-healing.md) | token-healing · prefix-caching · boundary | 📝 |
| Q2‑15 | [How do you estimate the number of tokens in a document without running the tokenizer? What heuristics are commonly used and when do they break?](./q15-token-counting-heuristics.md) | token-counting · heuristics · cost | 📝 |
| Q2‑16 | [What is the special token protocol (BOS, EOS, PAD, MASK, SEP, CLS) and how do different model families handle padding and attention masking for batched inference?](./q16-special-tokens.md) | special-tokens · padding · attention-mask | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q2‑17 | [What are glitch tokens? How do they arise from the BPE training process, and what security and behavioural risks do they introduce?](./q17-glitch-tokens.md) | glitch-tokens · SolidGoldMagikarp · adversarial | 📝 |
| Q2‑18 | [Derive the Unigram LM tokenizer objective and the EM algorithm used to optimise it. How does pruning work, and how is the vocabulary size controlled?](./q18-unigram-lm-derivation.md) | Unigram · EM · pruning · marginal-likelihood | 📝 |
| Q2‑19 | [How does the choice of tokenizer affect cross-lingual transfer? Which languages are systematically under-represented in byte-level BPE vocabularies and why?](./q19-multilingual-tokenization.md) | multilingual · fertility · low-resource · fairness | 📝 |
| Q2‑20 | [What is the vocabulary overlap problem when mixing corpora from different domains (code, maths, multilingual)? How do modern tokenizers like Llama-3 (128K) address it?](./q20-vocabulary-overlap.md) | vocabulary · domain-mix · code-maths · Llama-3 | 📝 |
| Q2‑21 | [Explain tiktoken's byte-level regex pre-tokenization. How does the split pattern affect merge frequency, and why does GPT-4o use a different pattern from GPT-4?](./q21-tiktoken-pretokenization.md) | tiktoken · regex · pre-tokenization · GPT-4o | 📝 |
| Q2‑22 | [How do you expand a pretrained tokenizer to add a new language or domain with minimal disruption to existing token IDs and embeddings?](./q22-tokenizer-extension.md) | vocabulary-extension · embedding-init · domain-adaptation | 📝 |
| Q2‑23 | [What is the relationship between the tokenizer and the model's context window in terms of information density? How does tokenization efficiency (bits-per-token) relate to effective context?](./q23-context-window-efficiency.md) | context-window · bits-per-token · information-density | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q2‑24 | [You are building a multilingual product that must serve 40 languages equally well. Walk through how you would design and evaluate the tokenizer.](./q24-multilingual-tokenizer-design.md) | multilingual · design · evaluation | 📝 |
| Q2‑25 | [A model fine-tuned on medical text produces garbled output on drug names and dosage strings. You suspect tokenization. How do you diagnose and fix it?](./q25-medical-tokenization-debug.md) | domain-specific · OOV · fine-tuning | 📝 |
| Q2‑26 | [Your LLM-based product costs 4× more than budgeted because a particular user query type tokenizes poorly. Walk through your investigation and optimisation plan.](./q26-token-cost-optimisation.md) | cost · fertility · prompt-engineering | 📝 |
| Q2‑27 | [How would you validate that a new tokenizer is better than the existing one for your use case before committing to full retraining?](./q27-tokenizer-evaluation.md) | evaluation · fertility · downstream-task | 📝 |
| Q2‑28 | [A red-team finds that carefully crafted token sequences cause the model to output harmful content that a keyword filter misses. What is happening and how do you defend against it?](./q28-adversarial-tokenization.md) | adversarial · red-team · safety · glitch-tokens | 📝 |
| Q2‑29 | [You need to add 10,000 new tokens (a proprietary domain vocabulary) to a 70B model that is already deployed. What is your strategy for the embedding initialisation and fine-tuning?](./q29-vocabulary-expansion-70b.md) | vocabulary-expansion · embedding-init · fine-tuning | 📝 |
| Q2‑30 | [How does tokenization affect retrieval in a RAG system? When can tokenization boundaries cause semantic leakage between chunks?](./q30-tokenization-rag-retrieval.md) | RAG · chunking · semantic-leakage | 📝 |
| Q2‑31 | [A code-generation model produces syntactically broken output whenever it generates identifiers longer than ~15 characters. Root-cause and fix.](./q31-long-identifier-bug.md) | code-generation · long-tokens · fertility | 📝 |
| Q2‑32 | [Design a tokenizer benchmarking suite for an LLM product team. What metrics would you track, what corpora would you use, and how would you report the results?](./q32-tokenizer-benchmark-suite.md) | benchmarking · metrics · evaluation-suite | 📝 |

---

## Why these questions matter

Tokenization questions appear at every interview level because they probe a candidate's **system-level thinking** — understanding that a model is not just its weights but the whole pipeline from raw text to predictions. The questions above cover four recurring interview themes:

| Theme | Key questions |
|-------|--------------|
| **Algorithm derivation** | Q2‑02, Q2‑08, Q2‑18 — show you can rebuild BPE / Unigram from scratch |
| **Design tradeoffs** | Q2‑10, Q2‑19, Q2‑20, Q2‑24 — vocabulary size, multilingual coverage, domain mixing |
| **Failure modes** | Q2‑13, Q2‑17, Q2‑25, Q2‑28 — numbers, glitch tokens, adversarial, domain OOV |
| **Production debugging** | Q2‑26, Q2‑29, Q2‑31 — cost, vocabulary expansion, long identifiers |

---

## Reading order suggestions

**New to tokenization?** Q2‑01 → Q2‑02 → Q2‑05 → Q2‑06 → Q2‑07

**Interview in 48 hours?** Q2‑02 (BPE) · Q2‑10 (vocab size) · Q2‑17 (glitch tokens) · Q2‑07 (fertility) · Q2‑13 (numbers)

**Full track:** Q2‑01 → Q2‑02 → Q2‑03 → Q2‑04 → Q2‑05 → Q2‑06 → Q2‑07 → Q2‑08 → Q2‑09 → Q2‑10 → Q2‑11 → Q2‑12 → Q2‑13 → Q2‑14 → Q2‑15 → Q2‑16 → Q2‑17 → Q2‑18 → Q2‑19 → Q2‑20 → Q2‑21 → Q2‑22 → Q2‑23

**Applied / debugging track:** Q2‑24 → Q2‑25 → Q2‑26 → Q2‑27 → Q2‑28 → Q2‑29 → Q2‑30 → Q2‑31 → Q2‑32

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 1 — Transformer Architecture](../section-01-transformer-architecture/README.md) &nbsp;|&nbsp; [Section 3 — Pretraining & Scaling Laws ➡️](../section-03-pretraining-and-scaling-laws/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
