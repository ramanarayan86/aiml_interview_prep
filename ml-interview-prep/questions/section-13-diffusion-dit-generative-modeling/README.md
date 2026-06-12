<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 12 — Evaluation, Safety & Interpretability](../section-12-evaluation-safety-interpretability/README.md) &nbsp;|&nbsp; [Section 14 — Systems & Hardware ➡️](../section-14-systems-and-hardware/README.md)

</div>

---

# Section 13 · Diffusion, DiT, and Generative Modeling

<div align="center">

![Questions](https://img.shields.io/badge/questions-37-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F37-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-37%2F37-6c4fe0)

</div>

Diffusion models dominate image, video, and audio generation. This section covers **DDPM fundamentals**, **latent diffusion (Stable Diffusion)**, **flow matching**, **DiT architectures**, **distillation** (LCM, Turbo), and the frontier of **video generation** (Sora/Veo), **3D generation**, and **controllable synthesis**.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q13‑01 | Walk through DDPM. What is the forward process, the reverse process, and the training objective? | DDPM · forward-process · reverse-process · denoising | 📝 |
| Q13‑02 | Why does the simplified ε-prediction objective work? What did the original paper derive vs use in practice? | epsilon-prediction · simplified-objective · DDPM | 📝 |
| Q13‑03 | What is classifier-free guidance? Why is it so effective? | CFG · classifier-free · guidance-scale · quality | 📝 |
| Q13‑04 | Explain latent diffusion (Stable Diffusion). What does the VAE buy you? | latent-diffusion · Stable-Diffusion · VAE · efficiency | 📝 |
| Q13‑05 | Compare DDPM, DDIM, and flow matching at a conceptual level. | DDPM · DDIM · flow-matching · comparison | 📝 |
| Q13‑06 | What is the difference between score-based and denoising diffusion formulations? | score-based · denoising · SDE · equivalence | 📝 |
| Q13‑07 | Explain the role of the noise schedule. Why does cosine schedule beat linear? | noise-schedule · cosine · linear · SNR | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q13‑08 | Walk through the DiT (Diffusion Transformer) architecture. Why did the field move from UNets to Transformers? | DiT · UNet · Transformer · scalability | 📝 |
| Q13‑09 | Explain rectified flow / flow matching. Why is it simpler than diffusion and why has SD3/FLUX adopted it? | rectified-flow · flow-matching · SD3 · FLUX | 📝 |
| Q13‑10 | Discuss MMDiT (used in SD3). How does it handle multimodal conditioning differently from cross-attention? | MMDiT · SD3 · multimodal-conditioning · joint-attention | 📝 |
| Q13‑11 | What is consistency distillation? How does it enable few-step generation? | consistency-distillation · LCM · few-step · distillation | 📝 |
| Q13‑12 | Compare diffusion, autoregressive (Parti, MAR), and masked-generative (MaskGIT, MAGVIT-v2) approaches to image generation. | diffusion · autoregressive · masked-generative · comparison | 📝 |
| Q13‑13 | Discuss the role of the text encoder in T2I models (CLIP, T5, both). Why did SD3 use both? | text-encoder · CLIP · T5 · SD3 · dual-encoder | 📝 |
| Q13‑14 | What is the difference between v-prediction, ε-prediction, and x0-prediction parameterizations? When does each shine? | v-prediction · epsilon · x0 · parameterization | 📝 |
| Q13‑15 | Compare adaLN-zero and cross-attention conditioning in DiT. Why does adaLN-zero help training stability? | adaLN-zero · cross-attention · conditioning · DiT | 📝 |
| Q13‑16 | Discuss progressive distillation and consistency models. How do they reduce sampling steps from 1000s to <10? | progressive-distillation · consistency-model · few-step | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q13‑17 | Derive flow matching from a probabilistic perspective. Why is the conditional flow matching loss tractable when the unconditional one isn't? | flow-matching · conditional · tractable · derivation | 📝 |
| Q13‑18 | Discuss quantization of diffusion transformers. Why is DiT quantization harder than LLM quantization? | DiT · quantization · timestep-dependent · activations | 📝 |
| Q13‑19 | Walk through scaling laws for diffusion models. Are they the same shape as LLM scaling laws? | diffusion-scaling · LLM-comparison · compute · quality | 📝 |
| Q13‑20 | Discuss video diffusion (Sora, Veo, Movie Gen). What architectural choices enable temporal consistency at minute-scale length? | video-diffusion · Sora · Veo · temporal-consistency | 📝 |
| Q13‑21 | Critically evaluate whether autoregressive next-token prediction can match diffusion for image quality. Discuss MAR, VAR, and what changes when you drop fixed token order. | AR-vs-diffusion · MAR · VAR · image-quality | 📝 |
| Q13‑22 | Walk through the design of a 3D-aware diffusion model. How do approaches like Zero-123, Magic3D, and ReconFusion handle multi-view consistency? | 3D-diffusion · Zero-123 · multi-view · consistency | 📝 |
| Q13‑23 | Discuss the Nitro-T / DiT family for low-step generation. What are the architectural and training-recipe innovations? | low-step · DiT · distillation · training-recipe | 📝 |
| Q13‑24 | Compare cascaded diffusion (Imagen) and latent diffusion (SD). What are the quality/efficiency tradeoffs? | cascaded · latent · Imagen · Stable-Diffusion | 📝 |
| Q13‑25 | Walk through how you would distill a 50-step diffusion model into a 4-step one (LCM, SDXL Turbo, DMD). What's the loss formulation? | distillation · LCM · SDXL-Turbo · DMD | 📝 |
| Q13‑26 | Discuss the connection between diffusion models and stochastic optimal control. What does the Schrödinger bridge formulation buy you? | SDE · optimal-control · Schrödinger-bridge · diffusion | 📝 |
| Q13‑27 | Critique the FID metric for evaluating image generation. What does it miss, and what alternatives (CLIPScore, HPSv2, human preference) have emerged? | FID · CLIPScore · HPSv2 · human-preference | 📝 |
| Q13‑28 | Walk through controllable generation — ControlNet, IP-Adapter, T2I-Adapter. What are the architectural and training tradeoffs? | ControlNet · IP-Adapter · T2I-Adapter · controllable | 📝 |
| Q13‑29 | Discuss the safety landscape for image generation models. How do you handle NSFW, deepfakes, and copyrighted content? | NSFW · deepfakes · copyright · safety · T2I | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q13‑30 | Your text-to-image model generates great single objects but fails at compositional prompts. Walk through your investigation and fix. | compositional · T2I · debugging · training | 📝 |
| Q13‑31 | Walk through how you'd reduce sampling steps from 50 to 4 for a deployed DiT without quality loss. | few-step · DiT · distillation · deployment | 📝 |
| Q13‑32 | You're asked to quantize a DiT model for edge deployment. Walk through which layers are sensitive and your bit-allocation strategy. | DiT · quantization · edge · bit-allocation | 📝 |
| Q13‑33 | Your model has temporal inconsistency in video generation (e.g., flickering, identity drift). Walk through your investigation. | video · temporal-consistency · flickering · debugging | 📝 |
| Q13‑34 | Walk through how you'd evaluate a new T2I model against Midjourney and FLUX in a way that produces actionable signal beyond FID. | T2I-eval · Midjourney · FLUX · beyond-FID | 📝 |
| Q13‑35 | Your model fails to handle a specific style or content type that competitors handle well. Walk through your data, training, or architectural fix plan. | style-gap · data · training · competitors | 📝 |
| Q13‑36 | Walk through how you'd add ControlNet-style conditioning to a deployed DiT without retraining from scratch. | ControlNet · DiT · conditioning · efficient-training | 📝 |
| Q13‑37 | Your team is choosing between rectified flow and standard diffusion for the next-gen model. Walk through the experiments you'd run and the decision criteria. | rectified-flow · diffusion · comparison · experiments | 📝 |

---

## Why these questions matter

Diffusion and generative modeling are a distinct specialty track. These questions test whether a candidate understands the **mathematical foundations of diffusion, practical training recipes, distillation techniques, and production evaluation**.

| Theme | Key questions |
|-------|--------------|
| **Diffusion fundamentals** | Q13‑01, Q13‑02, Q13‑05, Q13‑07 — DDPM, schedules, DDIM |
| **Architecture** | Q13‑08, Q13‑09, Q13‑10, Q13‑15 — DiT, flow matching, MMDiT |
| **Distillation & serving** | Q13‑11, Q13‑16, Q13‑25, Q13‑31 — LCM, consistency, 4-step |
| **Evaluation** | Q13‑27, Q13‑34 — FID, beyond-FID, human preference |

---

## Reading order suggestions

**New to diffusion?** Q13‑01 → Q13‑02 → Q13‑03 → Q13‑04 → Q13‑08

**Interview in 48 hours?** Q13‑08 (DiT) · Q13‑09 (flow matching) · Q13‑11 (consistency) · Q13‑20 (video) · Q13‑27 (eval)

**Full track:** Q13‑01 → Q13‑07 → Q13‑08 → Q13‑16 → Q13‑17 → Q13‑20 → Q13‑21

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 12 — Evaluation, Safety & Interpretability](../section-12-evaluation-safety-interpretability/README.md) &nbsp;|&nbsp; [Section 14 — Systems & Hardware ➡️](../section-14-systems-and-hardware/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s13qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
