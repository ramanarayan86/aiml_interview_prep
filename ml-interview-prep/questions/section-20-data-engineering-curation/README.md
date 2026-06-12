<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 19 — Prompt Engineering, ICL & Steering](../section-19-prompt-engineering-icl-steering/README.md) &nbsp;|&nbsp; [Section 21 — Frontier Topics & Research ➡️](../section-21-frontier-topics-research/README.md)

</div>

---

# Section 20 · Data Engineering and Curation

<div align="center">

![Questions](https://img.shields.io/badge/questions-27-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F27-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-27%2F27-6c4fe0)

</div>

"Data is the new moat" — and at frontier scale, data engineering is as important as architecture. This section covers **web-scale deduplication**, **quality filtering** (DataComp-LM, FineWeb), **data mixture optimization** (DoReMi, curriculum), **synthetic data**, and the emerging challenges of **copyright, PII, model collapse, and contamination detection**.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q20‑01 | Walk through a pretraining corpus pipeline from web crawl to training-ready tokens. What are the key stages? | pretraining · corpus · pipeline · web-crawl | 📝 |
| Q20‑02 | What is deduplication? Why is it important for both data quality and memorization? | deduplication · memorization · quality · near-duplicate | 📝 |
| Q20‑03 | What is quality filtering? Walk through common heuristics (language detection, perplexity, classifier-based). | quality-filtering · heuristics · perplexity · classifier | 📝 |
| Q20‑04 | What is data mixture / source mixing? Why does the ratio of domains matter at scale? | data-mixture · source-mixing · domain-ratio · scale | 📝 |
| Q20‑05 | What is data contamination? How does it affect benchmark evaluation? | contamination · benchmark · evaluation · leakage | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q20‑06 | Compare DataComp-LM, FineWeb, RedPajama-V2, and DCLM on quality filtering strategy. What is the consensus? | DataComp-LM · FineWeb · RedPajama · DCLM · quality | 📝 |
| Q20‑07 | Explain DoReMi for data mixture optimization. What signal does it use, and what are its assumptions? | DoReMi · data-mixture · proxy-model · importance-weights | 📝 |
| Q20‑08 | Compare MinHash, SimHash, and embedding-based deduplication. What's the precision-recall and cost tradeoff? | MinHash · SimHash · embedding-dedup · precision-recall | 📝 |
| Q20‑09 | Discuss synthetic data for pretraining and fine-tuning. When does it help, and when does it cause model collapse? | synthetic-data · collapse · pretraining · fine-tuning | 📝 |
| Q20‑10 | Walk through how instruction-tuning datasets (FLAN, OpenHermes, Orca, Magpie) differ in quality, diversity, and scale. | instruction-tuning · FLAN · OpenHermes · Magpie · quality | 📝 |
| Q20‑11 | What is per-document curriculum learning? Compare linear, exponential, and competency-based curriculum schedules. | curriculum · per-document · schedule · competency | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q20‑12 | Walk through how you'd design a copyright- and PII-aware data pipeline at scale. What are the legal and technical challenges? | copyright · PII · legal · pipeline · scale | 📝 |
| Q20‑13 | What is model collapse? Derive the mechanism formally, and describe how to prevent it in continuous data pipelines. | model-collapse · synthetic · continuous-pipeline · prevention | 📝 |
| Q20‑14 | Discuss contamination detection at scale. How do you detect when a benchmark has leaked into your training corpus? | contamination-detection · benchmark-leakage · n-gram · scale | 📝 |
| Q20‑15 | Discuss data attribution. How would you trace a model capability or failure back to specific training examples? | data-attribution · TracIn · influence-functions · capability | 📝 |
| Q20‑16 | Critically evaluate open data (CommonCrawl, RedPajama, ROOTS) vs proprietary data. Where is the quality gap, and can it be closed? | open-data · CommonCrawl · proprietary · quality-gap | 📝 |
| Q20‑17 | Walk through how you'd design a continuous-update data pipeline for a model that needs to track world knowledge weekly. | continuous-update · knowledge · weekly · pipeline | 📝 |
| Q20‑18 | Discuss the legal and ethical issues in LLM training data. What does the current litigation landscape (NYT, Getty, Authors Guild) imply for future pipelines? | legal · ethical · NYT · Getty · litigation | 📝 |
| Q20‑19 | Design a synthetic + web data curriculum that maintains quality as the web data distribution degrades over time (due to LLM-generated content flooding the web). | synthetic-web · curriculum · LLM-flood · degradation | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q20‑20 | Your model has memorized specific training examples that appear verbatim in outputs. Walk through your investigation and mitigation. | memorization · verbatim · mitigation · deduplication | 📝 |
| Q20‑21 | You discover that 15% of your fine-tuning dataset is contaminated with benchmark test data. Walk through your response. | contamination · 15-percent · fine-tuning · response | 📝 |
| Q20‑22 | Walk through how you'd build a deduplication pipeline for 100TB of web data. What are the bottlenecks? | dedup-pipeline · 100TB · MinHash · bottlenecks | 📝 |
| Q20‑23 | You're asked to build a data quality classifier that outperforms simple perplexity filtering. Walk through your approach. | quality-classifier · perplexity · fine-grained · training | 📝 |
| Q20‑24 | Walk through how you'd design a data flywheel that uses model outputs to improve training data quality. | data-flywheel · synthetic · quality · self-improvement | 📝 |
| Q20‑25 | Your model performs well on English benchmarks but poorly on non-English tasks. Walk through your data diagnosis and fix. | multilingual · data-imbalance · non-English · diagnosis | 📝 |
| Q20‑26 | Walk through how you'd implement a contamination detection system to validate new benchmarks before releasing them. | contamination-detection · benchmark · validation · n-gram | 📝 |
| Q20‑27 | You are building a domain-specific LLM on 10B tokens of proprietary data. Walk through your complete data pipeline design. | domain-specific · 10B-tokens · proprietary · pipeline | 📝 |

---

## Why these questions matter

Data engineering questions test whether a candidate understands that **model quality is ultimately bounded by data quality** — and whether they can reason about scale, legal constraints, and continuous improvement.

| Theme | Key questions |
|-------|--------------|
| **Pipeline fundamentals** | Q20‑01, Q20‑02, Q20‑03 — crawl, dedup, filtering |
| **Mixture & curriculum** | Q20‑04, Q20‑07, Q20‑11 — DoReMi, domain ratios |
| **Synthetic & collapse** | Q20‑09, Q20‑13, Q20‑19 — synthetic data, model collapse |
| **Legal & contamination** | Q20‑05, Q20‑12, Q20‑14, Q20‑18 — PII, copyright, contamination |

---

## Reading order suggestions

**New to data engineering?** Q20‑01 → Q20‑02 → Q20‑03 → Q20‑05 → Q20‑09

**Interview in 48 hours?** Q20‑07 (DoReMi) · Q20‑09 (synthetic data) · Q20‑13 (model collapse) · Q20‑14 (contamination) · Q20‑15 (attribution)

**Full track:** Q20‑01 → Q20‑04 → Q20‑07 → Q20‑09 → Q20‑12 → Q20‑13 → Q20‑19

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 19 — Prompt Engineering, ICL & Steering](../section-19-prompt-engineering-icl-steering/README.md) &nbsp;|&nbsp; [Section 21 — Frontier Topics & Research ➡️](../section-21-frontier-topics-research/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s20qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
