<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 14 — Systems & Hardware](../section-14-systems-and-hardware/README.md) &nbsp;|&nbsp; [Section 16 — Distributed Training & Optimization ➡️](../section-16-distributed-training-optimization/README.md)

</div>

---

# Section 15 · Mathematical Foundations and Probability

<div align="center">

![Questions](https://img.shields.io/badge/questions-30-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F30-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-30%2F30-6c4fe0)

</div>

Mathematical depth separates senior researchers from engineers. This section covers **information theory** (cross-entropy, KL, mutual information), **optimization theory** (NTK, Fisher, policy gradients), **probability** (concentration inequalities, reparameterization), and the deep connections between **diffusion models, SDEs, and optimal transport**.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q15‑01 | What is cross-entropy loss? Derive it from KL divergence and explain the connection to maximum likelihood. | cross-entropy · KL-divergence · MLE · information-theory | 📝 |
| Q15‑02 | Explain the difference between Jensen-Shannon, KL, and Wasserstein divergences. When is each appropriate? | JS-divergence · KL · Wasserstein · divergences | 📝 |
| Q15‑03 | What is the softmax function, and why is it preferred over argmax for training? | softmax · argmax · differentiability · temperature | 📝 |
| Q15‑04 | Explain the chain rule and how backpropagation uses it. | chain-rule · backpropagation · gradient · computation-graph | 📝 |
| Q15‑05 | What is the bias-variance tradeoff? How does it manifest in deep learning? | bias-variance · overfitting · underfitting · deep-learning | 📝 |
| Q15‑06 | Define perplexity. Why is it used as an LM metric, and what are its limitations? | perplexity · LM · cross-entropy · limitations | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q15‑07 | Derive the log-sum-exp trick. Why is it essential for numerical stability in attention and cross-entropy? | log-sum-exp · numerical-stability · attention · softmax | 📝 |
| Q15‑08 | Explain the reparameterization trick (VAE, Gaussian sampling). Why is it necessary for backprop through stochastic nodes? | reparameterization · VAE · ELBO · stochastic-backprop | 📝 |
| Q15‑09 | What is the Fisher information matrix, and how does it relate to natural gradient descent? | Fisher-information · natural-gradient · curvature · optimization | 📝 |
| Q15‑10 | Walk through the EM algorithm. Where does it appear in modern LLM training (e.g., MoE routing)? | EM-algorithm · MoE · routing · latent-variable | 📝 |
| Q15‑11 | Explain entropy regularization. Why does it appear in RL policies and routing distributions? | entropy-regularization · RL · routing · exploration | 📝 |
| Q15‑12 | Discuss the role of variance reduction in policy gradient methods (baselines, GAE, control variates). | variance-reduction · baselines · GAE · control-variates | 📝 |
| Q15‑13 | What is mutual information? How does it relate to contrastive learning objectives like InfoNCE? | mutual-information · InfoNCE · contrastive · representation | 📝 |
| Q15‑14 | Derive the gradient of the softmax function. Why is the Jacobian structure important? | softmax-gradient · Jacobian · backpropagation | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q15‑15 | Derive the policy gradient theorem from scratch. What assumptions does it require? | policy-gradient · REINFORCE · theorem · derivation | 📝 |
| Q15‑16 | Discuss neural tangent kernel (NTK) theory. What does it predict about training dynamics in infinite-width networks? | NTK · infinite-width · training-dynamics · linearization | 📝 |
| Q15‑17 | Walk through the connection between diffusion models and stochastic differential equations (SDEs). What does the score-based formulation buy you? | SDE · score-based · diffusion · connection | 📝 |
| Q15‑18 | Explain the Johnson-Lindenstrauss lemma. How is it relevant to random projections, sketching, and quantization? | JL-lemma · random-projection · sketching · dimensionality | 📝 |
| Q15‑19 | Discuss the connection between attention and kernel methods (Performer's positive orthogonal random features). What are the approximation guarantees? | attention · kernel-methods · Performer · FAVOR+ | 📝 |
| Q15‑20 | Walk through the math of incoherence and why rotation-based quantization (QuaRot, SpinQuant) works. What does the incoherence parameter μ control? | incoherence · QuaRot · SpinQuant · rotation · μ | 📝 |
| Q15‑21 | Derive the lower bound on quantization error for a given distribution and bit budget. What's the Lloyd-Max iteration? | Lloyd-Max · quantization-error · lower-bound · bits | 📝 |
| Q15‑22 | Discuss optimal transport in ML. Where does it appear (Wasserstein GAN, Sinkhorn, OT-CFM)? | optimal-transport · Wasserstein · Sinkhorn · OT-CFM | 📝 |
| Q15‑23 | What is the connection between energy-based models, score matching, and diffusion? Walk through the unified view. | EBM · score-matching · diffusion · unified-view | 📝 |
| Q15‑24 | Discuss the role of concentration inequalities in deep learning theory (Chernoff, Bernstein, McDiarmid). | Chernoff · Bernstein · McDiarmid · concentration | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q15‑25 | Walk through how you'd numerically validate that your custom attention kernel implements the right math under FP16/BF16. | numerical-validation · attention-kernel · FP16 · BF16 | 📝 |
| Q15‑26 | Your loss is mathematically equivalent to cross-entropy but training is unstable. Walk through how you'd diagnose the numerical issue. | numerical-stability · cross-entropy · instability · diagnosis | 📝 |
| Q15‑27 | Walk through how you'd derive and validate a new loss function for a new training objective from first principles, including unit tests. | loss-derivation · unit-tests · validation · first-principles | 📝 |
| Q15‑28 | You're asked to verify whether a new optimizer's claimed second-order behavior actually matches the math. Walk through your validation. | optimizer · second-order · validation · Fisher | 📝 |
| Q15‑29 | Walk through how you'd diagnose whether a divergence in your KL-regularized policy is from a bug in the KL computation or a real policy issue. | KL-divergence · policy · debugging · regularization | 📝 |
| Q15‑30 | A paper claims a new method has lower sample complexity. Walk through how you'd reproduce and verify the claim empirically. | sample-complexity · reproduction · empirical · verification | 📝 |

---

## Why these questions matter

Mathematical depth is the distinguishing signal at research-scientist and principal-engineer interviews. These questions probe whether a candidate can **derive, not just recall** — and whether they can connect theory to practical implementation choices.

| Theme | Key questions |
|-------|--------------|
| **Information theory** | Q15‑01, Q15‑02, Q15‑06, Q15‑13 — cross-entropy, KL, mutual information |
| **Optimization** | Q15‑09, Q15‑15, Q15‑16 — Fisher, policy gradients, NTK |
| **Probability & statistics** | Q15‑08, Q15‑12, Q15‑24 — reparameterization, GAE, concentration |
| **Diffusion & OT** | Q15‑17, Q15‑22, Q15‑23 — SDEs, Wasserstein, score matching |

---

## Reading order suggestions

**New to the math?** Q15‑01 → Q15‑03 → Q15‑04 → Q15‑06 → Q15‑07

**Interview in 48 hours?** Q15‑15 (policy gradient) · Q15‑16 (NTK) · Q15‑17 (SDE/diffusion) · Q15‑19 (attention/kernels) · Q15‑20 (quantization math)

**Full track:** Q15‑01 → Q15‑06 → Q15‑07 → Q15‑14 → Q15‑15 → Q15‑17 → Q15‑22

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 14 — Systems & Hardware](../section-14-systems-and-hardware/README.md) &nbsp;|&nbsp; [Section 16 — Distributed Training & Optimization ➡️](../section-16-distributed-training-optimization/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s15qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
