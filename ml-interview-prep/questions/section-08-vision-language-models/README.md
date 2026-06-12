<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 7 — Quantization & Model Compression](../section-07-quantization-and-model-compression/README.md) &nbsp;|&nbsp; [Section 9 — Retrieval, RAG & Long Context ➡️](../section-09-retrieval-rag-long-context/README.md)

</div>

---

# Section 8 · Vision-Language Models (VLMs)

<div align="center">

![Questions](https://img.shields.io/badge/questions-49-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F49-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-49%2F49-6c4fe0)

</div>

VLMs are the dominant paradigm for multimodal AI. This section covers **CLIP and ViT fundamentals**, **LLaVA-style connector architectures**, **high-resolution strategies** (dynamic resolution, AnyRes), **video understanding**, and the frontier of **VLM-as-judge** systems for evaluating generated visual content.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q8‑01 | Walk through CLIP's training objective. What does the contrastive InfoNCE loss optimize? | CLIP · InfoNCE · contrastive · vision-language | 📝 |
| Q8‑02 | What is a ViT? How does it tokenize images? | ViT · patch-embedding · tokenization · image | 📝 |
| Q8‑03 | Compare early-fusion vs late-fusion multimodal architectures. | early-fusion · late-fusion · multimodal · architecture | 📝 |
| Q8‑04 | Explain the role of the vision encoder, projector, and LLM in a LLaVA-style VLM. | LLaVA · vision-encoder · projector · LLM | 📝 |
| Q8‑05 | What is the difference between contrastive (CLIP), generative (BLIP), and instruction-tuned (LLaVA) VLMs? | CLIP · BLIP · LLaVA · VLM-types | 📝 |
| Q8‑06 | What is patch embedding in ViT, and how does patch size affect compute and quality? | patch-embedding · ViT · patch-size · compute | 📝 |
| Q8‑07 | Compare CNN-based and ViT-based image encoders for VLMs. Why has ViT dominated? | CNN · ViT · encoder · comparison | 📝 |
| Q8‑08 | What is the [CLS] token in ViT and how is it used (or replaced) in VLMs? | CLS-token · ViT · pooling · VLM | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q8‑09 | Walk through LLaVA-1.5's training recipe. Why does the two-stage pretrain → instruction-tune work? | LLaVA-1.5 · two-stage · pretrain · instruction-tune | 📝 |
| Q8‑10 | Discuss the limitations of CLIP for fine-grained understanding. What does SigLIP change and why is the sigmoid loss simpler? | CLIP · SigLIP · sigmoid-loss · fine-grained | 📝 |
| Q8‑11 | Compare Q-Former (BLIP-2), linear projector (LLaVA), and cross-attention (Flamingo) for visual feature injection. What are the tradeoffs? | Q-Former · BLIP-2 · Flamingo · projector · comparison | 📝 |
| Q8‑12 | Explain dynamic resolution / native resolution approaches (Qwen2-VL, InternVL2, NaViT). Why do fixed 224×224 patches lose information? | dynamic-resolution · Qwen2-VL · InternVL2 · NaViT | 📝 |
| Q8‑13 | What is the 'vision token explosion' problem at high resolution? How do approaches like AnyRes, S2, and token pruning address it? | token-explosion · AnyRes · S2 · token-pruning | 📝 |
| Q8‑14 | Discuss how positional encoding works in VLMs across image patches. What is 2D RoPE and how does Qwen2-VL's M-RoPE generalize it? | 2D-RoPE · M-RoPE · Qwen2-VL · positional-encoding | 📝 |
| Q8‑15 | Explain visual instruction tuning data construction. How are LLaVA-Instruct, ShareGPT4V, and Cambrian-7M built? | visual-instruction · LLaVA-Instruct · ShareGPT4V · data-construction | 📝 |
| Q8‑16 | What is the role of the visual encoder's frozen vs unfrozen training? When does unfreezing the encoder help? | frozen · unfrozen · encoder-training · fine-tuning | 📝 |
| Q8‑17 | Compare image-level, region-level, and pixel-level visual understanding tasks. What VLM design choices are needed for each? | image-level · region-level · pixel-level · grounding | 📝 |

---

### 🟠 Intermediate-Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q8‑18 | Discuss VLM-as-judge / VLM-as-verifier. What evaluation properties make a VLM reliable as an automated judge? How do you calibrate against human raters? | VLM-judge · calibration · automated-eval · human-raters | 📝 |
| Q8‑19 | How would you evaluate a VLM's spatial reasoning, text-grounded reasoning, and hallucination rate? Discuss MMMU, MMVet, MathVista, BLINK, HallusionBench. | spatial-reasoning · MMMU · hallucination · benchmarks | 📝 |
| Q8‑20 | Walk through the design of a multi-image VLM (multi-view reference images, video frames). How does positional encoding generalize across images? | multi-image · multi-view · positional-encoding · video | 📝 |
| Q8‑21 | Compare Gemini's native multimodality (interleaved training from scratch) with bolt-on approaches (LLaVA-style). What are the architectural and data implications? | Gemini · native-multimodal · LLaVA · interleaved | 📝 |
| Q8‑22 | Discuss multi-call VLM pipelines (judge → verifier → aggregator). When does multi-call beat single-call, and what are the latency/cost tradeoffs? | multi-call · judge-verifier · pipeline · latency | 📝 |
| Q8‑23 | Walk through the design of a VLM scoring rubric for AI-generated images. How do you decompose 'quality' into independently judgeable dimensions? | scoring-rubric · AI-images · quality-dimensions · VLM | 📝 |
| Q8‑24 | Discuss the role of multi-view reference images in VLM evaluation. How do you handle viewpoint mismatch? | multi-view · reference-images · viewpoint · evaluation | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q8‑25 | Critique CLIP's 'bag of words' failure modes (Winoground, ARO). Why does contrastive pretraining fail at compositional understanding? What do NegCLIP, structure-aware training, and DCI propose? | CLIP · Winoground · compositional · NegCLIP | 📝 |
| Q8‑26 | Design a VLM scoring system for evaluating AI-generated images on dimensions like prompt adherence, anatomical correctness, and aesthetic quality. How do you handle calibration drift? | VLM-scoring · calibration-drift · AI-images · multi-dimension | 📝 |
| Q8‑27 | Discuss visual grounding in modern VLMs. Compare bounding-box prediction (Kosmos-2, GroundingGPT), set-of-marks prompting, and segmentation-aware VLMs (Sa2VA, PixelLM). | visual-grounding · Kosmos-2 · set-of-marks · segmentation | 📝 |
| Q8‑28 | Walk through video understanding architectures — temporal pooling vs temporal attention vs streaming (LongVA, Video-LLaVA). How does temporal redundancy change the token allocation problem? | video · temporal-attention · streaming · token-allocation | 📝 |
| Q8‑29 | Critically evaluate whether VLMs 'see' or whether they're language models with visual conditioning. Discuss probing experiments and why VLMs hallucinate spatial relations. | VLM-perception · hallucination · spatial-relations · probing | 📝 |
| Q8‑30 | Discuss the limits of VLM evaluation benchmarks. Why do MMMU and MMBench scores saturate while real-world behavior remains brittle? | benchmark-saturation · MMMU · MMBench · brittleness | 📝 |
| Q8‑31 | Design a self-improving VLM judge using prompt optimization (OPRO, APO) and preference learning. What's the convergence story? | self-improving · OPRO · preference-learning · VLM-judge | 📝 |
| Q8‑32 | Walk through the design of a 9-layer scoring architecture for visual asset evaluation. What does each layer add, and how do you prevent score collapse? | scoring-architecture · visual-assets · score-collapse | 📝 |
| Q8‑33 | Discuss calibration of VLM judges against a labeled artist dataset. How do you measure inter-rater agreement, judge-vs-human correlation, and rank stability? | calibration · inter-rater · judge-human-correlation | 📝 |
| Q8‑34 | What does it mean for a VLM judge to be 'production-grade'? Discuss latency, cost, drift detection, A/B testing, and human-in-the-loop fallbacks. | production-grade · drift-detection · A/B-testing · human-in-loop | 📝 |
| Q8‑35 | Walk through how you would extend a VLM judge to handle video and 3D asset evaluation. What changes in the encoder, projector, and rubric? | video-judge · 3D-assets · encoder · rubric-extension | 📝 |
| Q8‑36 | Discuss the connection between VLM judges and the LLM-as-a-Verifier framework. What does the verifier framing buy you for visual tasks? | VLM-verifier · LLM-judge · visual-tasks · verification | 📝 |
| Q8‑37 | Critique vision-language pretraining objectives. Is contrastive + generative the right combination, or is something fundamental missing? | pretraining-objectives · contrastive · generative · 3D-causal | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q8‑38 | Your VLM scores high on MMMU but a customer reports it can't reliably read text in screenshots. Walk through how you'd diagnose and fix. | OCR · screenshot · MMMU · debugging | 📝 |
| Q8‑39 | You're asked to deploy a VLM judge for evaluating AI-generated images. Walk through your design from rubric to deployment, including calibration drift guards. | VLM-judge · deployment · rubric · calibration | 📝 |
| Q8‑40 | Walk through how you'd debug a VLM that hallucinates objects not present in the image roughly 5% of the time. | hallucination · object-detection · debugging · VLM | 📝 |
| Q8‑41 | You're given $200K budget for human raters to calibrate a VLM judge. Walk through how you'd allocate it across labeling, double-rating, and evaluator-of-evaluators. | human-raters · budget · calibration · annotation | 📝 |
| Q8‑42 | A new VLM beats yours on benchmarks but loses on internal A/B tests. Walk through what you'd investigate. | benchmark-gap · A/B-test · VLM · investigation | 📝 |
| Q8‑43 | Walk through how you'd design a multi-call VLM pipeline where each call has a specific scoring role. What's your latency-vs-quality tradeoff? | multi-call · scoring · latency · pipeline | 📝 |
| Q8‑44 | Your VLM works well on 1024×1024 images but degrades on 4K images. Walk through your investigation and fix options. | high-resolution · 4K · token-budget · investigation | 📝 |
| Q8‑45 | Walk through how you'd extend a single-image VLM to handle multi-view references for asset evaluation. What architectural and training changes would you make? | multi-view · single-image · architecture · training | 📝 |
| Q8‑46 | Your VLM judge's correlation with human raters dropped from 0.82 to 0.71 over 3 months. Walk through your diagnosis — model drift, data drift, or rater drift? | correlation-drop · drift · VLM-judge · diagnosis | 📝 |
| Q8‑47 | You need to ship a VLM that can detect anatomical errors in generated images of humans. Walk through your data, evaluation, and training plan. | anatomical · error-detection · data · training-plan | 📝 |
| Q8‑48 | A product team wants to use your VLM judge as a hard quality gate. Walk through what concerns you'd raise and what guardrails you'd insist on. | quality-gate · guardrails · reliability · VLM-judge | 📝 |
| Q8‑49 | Walk through how you'd build an interview bible / reference document for a complex production VLM system that a new team member could ramp on in two weeks. | onboarding · documentation · VLM-system · ramp-up | 📝 |

---

## Why these questions matter

VLMs are at the intersection of vision and language research, and VLM-as-judge systems are increasingly used in production pipelines. These questions test whether a candidate can navigate **architecture choices, training recipes, evaluation design, and production deployment**.

| Theme | Key questions |
|-------|--------------|
| **Architecture** | Q8‑03, Q8‑04, Q8‑11, Q8‑12 — connectors, resolution, Q-Former vs projector |
| **Training** | Q8‑09, Q8‑15, Q8‑16, Q8‑21 — LLaVA recipe, data, frozen encoder |
| **Evaluation** | Q8‑18, Q8‑19, Q8‑30, Q8‑33 — VLM-judge, benchmarks, calibration |
| **Production** | Q8‑34, Q8‑39, Q8‑46, Q8‑48 — drift, deployment, guardrails |

---

## Reading order suggestions

**New to VLMs?** Q8‑01 → Q8‑02 → Q8‑04 → Q8‑05 → Q8‑09

**Interview in 48 hours?** Q8‑04 (LLaVA) · Q8‑11 (projector comparison) · Q8‑12 (dynamic resolution) · Q8‑25 (CLIP limits) · Q8‑29 (do VLMs see?)

**Full track:** Q8‑01 → Q8‑08 → Q8‑09 → Q8‑17 → Q8‑18 → Q8‑25 → Q8‑37

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 7 — Quantization & Model Compression](../section-07-quantization-and-model-compression/README.md) &nbsp;|&nbsp; [Section 9 — Retrieval, RAG & Long Context ➡️](../section-09-retrieval-rag-long-context/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s8qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
