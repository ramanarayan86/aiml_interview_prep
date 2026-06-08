<div align="center">

[🏠 Home](./README.md)

</div>

---

# Roadmap

This file tracks what's done, what's actively being written, and what's next. Updated with each significant push.

> **Want to contribute?** Pick any 📝 question below, copy [`_TEMPLATE.md`](./_TEMPLATE.md), and open a PR. One question per PR. See [CONTRIBUTING.md](./CONTRIBUTING.md) for the 6-step process.

---

## Current status (as of June 2026)

| Section | Total Qs | Answered | Scaffolded | Progress |
|---------|:--------:|:--------:|:----------:|----------|
| §1 Transformer Architecture | 42 | **34** | 8 | `████████████████████░░░░` 81% |
| §2 Tokenization & Embeddings | 32 | **7** | 25 | `██████░░░░░░░░░░░░░░░░░░` 22% |
| §3 Pretraining & Scaling Laws | 10 | 0 | 0 | not started |
| §4 Post-training: SFT, RLHF, DPO | 10 | 0 | 0 | not started |
| §5 Reasoning, CoT, Inference-Time Compute | 10 | 0 | 0 | not started |
| §6 Efficient Inference & Serving | 10 | 0 | 0 | not started |
| §7 Quantization & Model Compression | 10 | 0 | 0 | not started |
| §8 Vision-Language Models (VLMs) | 10 | 0 | 0 | not started |
| §9 Retrieval, RAG & Long Context | 10 | 0 | 0 | not started |
| §10 Mixture of Experts (MoE) | 8 | 0 | 0 | not started |
| §11 Agents, Tool Use & Code Gen | 10 | 0 | 0 | not started |
| §12 Evaluation, Safety & Interpretability | 10 | 0 | 0 | not started |
| §13 Diffusion, DiT & Generative Modeling | 10 | 0 | 0 | not started |
| §14 Systems & Hardware | 10 | 0 | 0 | not started |
| §15 Mathematical Foundations | 10 | 0 | 0 | not started |
| §16 Distributed Training & Optimization | 10 | 0 | 0 | not started |
| §17 Fine-Tuning, PEFT & Adaptation | 10 | 0 | 0 | not started |
| §18 Speech, Audio & Multimodal | 8 | 0 | 0 | not started |
| §19 Prompt Engineering, ICL & Steering | 10 | 0 | 0 | not started |
| §20 Data Engineering & Curation | 10 | 0 | 0 | not started |
| §21 Frontier Topics & Research Directions | 10 | 0 | 0 | not started |
| §22 Behavioral & Research Leadership | 8 | 0 | 0 | not started |
| §23 Vector Databases & Search | 12 | 0 | 0 | not started |
| §24 Document Digitization & Chunking | 10 | 0 | 0 | not started |
| §25 Embedding Models & Semantic Search | 9 | 0 | 0 | not started |
| §26 Production LLM Systems & Cost | 10 | 0 | 0 | not started |
| §27 Hallucination: Causes, Detection, Mitigation | 9 | 0 | 0 | not started |
| §28 Prompt Security & Adversarial Inputs | 6 | 0 | 0 | not started |
| §29 LLM Application Design Patterns | 10 | 0 | 0 | not started |
| §30 LLM Evaluation in Practice | 9 | 0 | 0 | not started |
| **Total** | **≈305** | **41** | **33** | **13%** |

---

## Active sprint

### Finishing Section 2 — Tokenization & Embeddings

**Intermediate (Q2-08 to Q2-16) — 9 questions**

| Q | Title | Status |
|---|-------|:------:|
| Q2-08 | BPE algorithm complexity end-to-end | 📝 open |
| Q2-09 | SentencePiece vs HuggingFace tokenizers | 📝 open |
| Q2-10 | Vocabulary size tradeoffs (32K vs 128K) | 📝 open |
| Q2-11 | Embedding initialisation and training stability | 📝 open |
| Q2-12 | Weight tying in/out embedding matrix | 📝 open |
| Q2-13 | Tokenization and numbers/arithmetic | 📝 open |
| Q2-14 | Token healing and prefix caching | 📝 open |
| Q2-15 | Token counting heuristics | 📝 open |
| Q2-16 | Special tokens (BOS, EOS, PAD, MASK) | 📝 open |

**Advanced (Q2-17 to Q2-23) — 7 questions**

| Q | Title | Status |
|---|-------|:------:|
| Q2-17 | Glitch tokens (SolidGoldMagikarp) | 📝 open |
| Q2-18 | Unigram LM derivation (EM) | 📝 open |
| Q2-19 | Multilingual tokenization and fertility | 📝 open |
| Q2-20 | Vocabulary overlap across domains | 📝 open |
| Q2-21 | tiktoken regex pre-tokenization | 📝 open |
| Q2-22 | Tokenizer extension for new languages | 📝 open |
| Q2-23 | Context window efficiency (bits-per-token) | 📝 open |

**Applied (Q2-24 to Q2-32) — 9 questions**

| Q | Title | Status |
|---|-------|:------:|
| Q2-24 | Multilingual tokenizer design (40 languages) | 📝 open |
| Q2-25 | Medical domain OOV debugging | 📝 open |
| Q2-26 | Token cost 4× over budget — investigation | 📝 open |
| Q2-27 | Tokenizer evaluation methodology | 📝 open |
| Q2-28 | Adversarial tokenization and red-teaming | 📝 open |
| Q2-29 | Vocabulary expansion for 70B deployed model | 📝 open |
| Q2-30 | Tokenization and RAG retrieval | 📝 open |
| Q2-31 | Long identifier bug in code generation | 📝 open |
| Q2-32 | Tokenizer benchmarking suite design | 📝 open |

---

## Section 1 — remaining Applied questions

| Q | Title | Status |
|---|-------|:------:|
| Q35 | Debugging 35% MFU on 13B model on H100 | 📝 open |
| Q36 | Great training loss but repetitive generation | 📝 open |
| Q37 | Migrating post-norm → pre-norm without full retrain | 📝 open |
| Q38 | Experiments before swapping in sub-quadratic attention | 📝 open |
| Q39 | Extend 4K → 128K context in 2-week budget | 📝 open |
| Q40 | Intermittent NaNs in attention kernels | 📝 stub exists |
| Q41 | GQA-8 vs GQA-4 for 70B — how to decide | 📝 open |
| Q42 | One layer = 40% inference latency — investigate | 📝 open |

---

## Next sections (priority order)

These are ranked by interview frequency and complement to existing content.

| Priority | Section | Reason |
|:--------:|---------|--------|
| 1 | §6 Efficient Inference & Serving | Speculative decoding, continuous batching, vLLM — asked in almost every applied/MLE interview |
| 2 | §9 Retrieval, RAG & Long Context | RAG architecture is the most common production LLM pattern |
| 3 | §7 Quantization & Model Compression | GPTQ/AWQ/GGUF questions common at on-device and startup roles |
| 4 | §17 Fine-Tuning, PEFT & Adaptation | LoRA/QLoRA questions near-universal at fine-tuning/applied roles |
| 5 | §3 Pretraining & Scaling Laws | Chinchilla, emergence, MFU — common at frontier lab research roles |
| 6 | §5 Reasoning, CoT & Inference-Time Compute | o1/o3-style reasoning, MCTS, self-consistency — growing rapidly |
| 7 | §8 Vision-Language Models | LLaVA, InternVL, GPT-4V — common at multimodal roles |
| 8 | §12 Evaluation, Safety & Interpretability | Safety/alignment questions at labs; eval at applied roles |

---

## Contribution bounty board

If you'd like to contribute but aren't sure where to start, these are the highest-value gaps:

| Question | Why it's high-value |
|----------|-------------------|
| Q35 — MFU debugging | Real scenario; tests systems + architecture knowledge |
| Q2-17 — Glitch tokens | Memorable, frequently asked, not well covered elsewhere |
| Q2-13 — Numbers & arithmetic | Concrete failure mode; easy to verify with experiments |
| Q6-xx — Speculative decoding | Widely used in production; little good public explanation |
| Q9-xx — RAG chunking strategies | Most common production LLM architecture question |

---

<div align="center">

[🏠 Home](./README.md) &nbsp;|&nbsp; [🤝 Contributing](./CONTRIBUTING.md) &nbsp;|&nbsp; [📖 Glossary](./GLOSSARY.md)

</div>
