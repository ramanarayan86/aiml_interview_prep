<!-- ============================ TOP NAV ============================ -->
<div align="center">

[🏠 Home](../../README.md) &nbsp;•&nbsp; [📚 Section 4 — Post-training: SFT, RLHF, DPO](./README.md) &nbsp;•&nbsp; [⬅️ Q4‑05 — On-Policy vs Off-Policy RL](./q05-on-policy-vs-off-policy.md) &nbsp;•&nbsp; [Q4‑07 — System Prompt vs User Prompt ➡️](./q07-system-prompt-vs-user-prompt.md)

</div>

---

# Q4‑06 · What is instruction tuning, and how does it differ from chat tuning?

<div align="center">

![Section](https://img.shields.io/badge/Section_4-Post--training%3A_SFT%2C_RLHF%2C_DPO-1f4fa3)
![Level](https://img.shields.io/badge/Level-Basic-0e8a6e)
![Tags](https://img.shields.io/badge/tags-instruction--tuning·chat--tuning·SFT-0e8a6e)
![Read](https://img.shields.io/badge/read_time-~15_min-5b6675)

</div>

> [!IMPORTANT]
> **The 20-second answer.** **Instruction tuning** fine-tunes a pretrained LLM on a diverse set of (instruction, response) pairs — each a single-turn example — so the model learns to follow natural language directives zero-shot on tasks it has never seen. The key insight (Wei et al. 2022, FLAN): training across a *diverse* mixture of tasks in instruction format generalizes to *unseen* tasks far better than training on a narrow set. **Chat tuning** extends this to multi-turn conversations: it adds a **system prompt** that sets persona and constraints, includes the full **conversation history** in the input, and trains on data structured as `[SYSTEM] [USER] [ASSISTANT] [USER] [ASSISTANT]…`. Both paradigms use the same cross-entropy loss, but **only on the response/assistant tokens** — loss on instruction or user tokens wastes gradient signal and hurts performance. Instruction tuning produces a zero-shot task follower; chat tuning produces an interactive assistant.

---

## Table of contents

1. [First principles: why pretrained models can't follow instructions](#1--first-principles-why-pretrained-models-cant-follow-instructions)
2. [Instruction tuning — the core mechanism](#2--instruction-tuning--the-core-mechanism)
3. [Figure 1 — Instruction vs. chat data formats side by side](#3--figure-1--instruction-vs-chat-data-formats-side-by-side)
4. [The FLAN insight: task diversity drives generalization](#4--the-flan-insight-task-diversity-drives-generalization)
5. [Figure 2 — Task generalization curve](#5--figure-2--task-generalization-curve)
6. [Chat tuning — extending to multi-turn conversations](#6--chat-tuning--extending-to-multi-turn-conversations)
7. [Loss masking — the critical implementation detail](#7--loss-masking--the-critical-implementation-detail)
8. [Worked numerical example — loss masking in practice](#8--worked-numerical-example--loss-masking-in-practice)
9. [Key differences — comparison table](#9--key-differences--comparison-table)
10. [Common misconceptions](#10--common-misconceptions)
11. [Interview drill — follow-up questions](#11--interview-drill--follow-up-questions)
12. [Connections to other concepts](#12--connections-to-other-concepts)
13. [One-screen summary](#13--one-screen-summary)
14. [Five-minute refresher](#14--five-minute-refresher)
15. [References](#15--references)

---

## 1 · First principles: why pretrained models can't follow instructions

A pretrained language model is trained to predict the next token given a prefix. Its objective is:

$$\mathcal{L}_{\text{PT}} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

This objective is agnostic to the *intent* behind any particular input. Given the prompt "Translate to French: Hello," a pretrained model might continue with "Goodbye," "German: Hallo," or even another translation example — because it has seen all of these continuations on the internet. It has no notion that the input is an instruction demanding a specific type of output.

**The alignment gap.** The pretrained model has learned *form* (how text looks) but not *function* (how to respond when commanded). Instruction tuning closes this gap by making the model an instruction follower rather than a raw text completor.

The key ingredient is the data format: instead of raw documents, the model sees examples of the form:

```
Input:  <instruction text>
Output: <desired response>
```

After seeing thousands of such pairs across dozens of task types, the model internalizes the *meta-skill* of reading an instruction and producing the appropriate output type.

---

## 2 · Instruction tuning — the core mechanism

Instruction tuning is **supervised fine-tuning (SFT)** on a curated dataset of (instruction, response) pairs. The training objective is still cross-entropy over the response tokens:

$$\mathcal{L}_{\text{IT}} = -\sum_{t \in \text{response}} \log P_\theta(x_t \mid x_{<t})$$

Note the summation is restricted to response token positions — the instruction tokens are in the context but contribute no gradient signal (see Section 7 for why this matters).

**Three hallmark instruction datasets illustrate the design space:**

**FLAN (Wei et al. 2022)** — 62 existing NLP datasets (sentiment, QA, summarization, translation, etc.) reformatted into instruction templates. The insight is that the *same task* can be expressed in 10+ different instruction phrasings, each becoming a separate training example. FLAN trains on 60 task clusters and tests zero-shot on the held-out 2.

**T0 / T0++ (Sanh et al. 2022)** — A similar approach using 35 datasets from PromptSource, with careful attention to avoiding data contamination between train and test splits. T0 11B outperforms GPT-3 175B zero-shot on many SuperGLUE tasks.

**Super-Natural Instructions (Wang et al. 2022)** — 1,616 diverse tasks with natural language definitions and examples. Introduces the *instruction schema*: a definition section, positive/negative examples, and the actual instance.

**Alpaca (Taori et al. 2023)** — 52K instructions generated by GPT-3.5 ("self-instruct" method), used to fine-tune LLaMA 7B. Showed that even a small, synthetic instruction dataset produces a strong instruction follower.

The common thread: train on a **diverse mixture** of tasks in instruction format. Diversity of tasks generalizes; diversity of phrasings within a task is a bonus.

---

## 3 · Figure 1 — Instruction vs. chat data formats side by side

<div align="center">
<img src="../../assets/s4q06/fig1-instruction-vs-chat-format.svg" alt="Two-panel diagram comparing instruction tuning and chat tuning data formats. Left panel shows instruction tuning with single-turn format using FLAN and Alpaca templates — each example is a standalone instruction followed by a response, with no system prompt. Right panel shows chat tuning with ChatML multi-turn format including system, user, and assistant turns. Color coding shows which tokens receive loss: only assistant turns (red) receive gradient signal; system and user turns (navy and teal) are masked out." width="92%">
<br><sub><b>Figure 1.</b> Instruction tuning uses single-turn (instruction, response) pairs with no mandatory system prompt. Chat tuning uses a structured multi-turn format with system, user, and assistant role markers. In both cases, only the model's output tokens (response / assistant turns) contribute to the loss — instruction and user tokens are masked.</sub>
</div>

---

## 4 · The FLAN insight: task diversity drives generalization

The most important empirical finding in the FLAN paper (Wei et al. 2022) is about **zero-shot generalization**:

> Training on more diverse tasks produces better zero-shot performance on held-out tasks, up to a saturation point.

This finding has three important corollaries:

**1. Diversity > volume within a single task.** Adding 1,000 more examples of sentiment analysis does not help zero-shot generalization to translation. Adding 10 new task types does.

**2. The gain is approximately logarithmic.** Going from 10 to 20 task types yields a large improvement; going from 50 to 60 yields a smaller one. Performance saturates around 60–100 task clusters for GPT-scale models (exact saturation point scales with model size).

**3. Instruction format is load-bearing.** The same data without instruction formatting does not produce the same generalization. The model must learn to interpret the *instruction* as an intent signal, not just see raw NLP data.

The intuition: instruction tuning teaches the model to recognize the *pattern* `<instruction> → <output>`, and this pattern generalizes to any instruction it has not seen before — provided the model has seen enough variation during training.

---

## 5 · Figure 2 — Task generalization curve

<div align="center">
<img src="../../assets/s4q06/fig2-task-generalization-curve.svg" alt="Line chart showing zero-shot performance on held-out tasks (y-axis, 0-100 percent) versus the number of task clusters in the training mixture (x-axis, 0-62). The navy FLAN curve rises steeply from about 8 percent at 0 tasks to 65 percent at 20 tasks, then flattens into a saturation regime (shaded blue region) between 40-62 tasks, reaching about 83 percent at 62 task clusters. A dashed gray horizontal line marks the GPT-3 175B zero-shot baseline at approximately 35 percent. An annotation box highlights the key insight: task diversity is more important than task volume. The saturation regime is annotated at the right." width="92%">
<br><sub><b>Figure 2.</b> Zero-shot generalization on held-out NLP tasks as a function of task diversity in the training mixture (FLAN, Wei et al. 2022). Performance rises steeply in the first 20–25 task clusters and enters a saturation regime beyond ~40 clusters. FLAN-LaMDA 137B with 62 tasks significantly surpasses GPT-3 175B (no instruction tuning) — demonstrating that a smaller model with diverse instruction tuning outperforms a much larger raw pretrained model.</sub>
</div>

---

## 6 · Chat tuning — extending to multi-turn conversations

Chat tuning is instruction tuning extended to the conversational setting. It introduces three structural additions not present in basic instruction tuning:

### 6.1 · System prompt

A system prompt is a special prefix that sets the model's **persona, tone, and behavioral constraints** before any user message. Examples:

- `You are a helpful, harmless, and honest assistant.`
- `You are a medical information assistant. Always recommend consulting a doctor for personal medical advice.`
- `You are a coding assistant. Always respond in Python unless asked otherwise.`

System prompts are injected at the start of every conversation and remain invisible to the user in most UIs. The model learns during chat tuning that the system prompt sets inviolable constraints it must respect throughout the conversation.

### 6.2 · Conversation history

Multi-turn conversation means the model's input at turn $t$ contains all prior turns:

$$\text{Input}_t = [\text{SYSTEM}, \text{USER}_1, \text{ASST}_1, \text{USER}_2, \text{ASST}_2, \ldots, \text{USER}_t]$$

The model must generate $\text{ASST}_t$ conditioned on this entire context. This creates requirements that single-turn instruction tuning does not address:

- **Coreference resolution** — knowing that "it" in turn 4 refers to an object mentioned in turn 2.
- **Instruction tracking** — remembering a constraint from turn 1 ("respond only in Spanish") five turns later.
- **Topic coherence** — maintaining the thread of the conversation rather than treating each turn as a fresh zero-shot request.

### 6.3 · Chat markup formats

Chat data uses special tokens to delimit role boundaries. Three dominant formats:

| Format | System marker | User marker | Assistant marker |
|--------|---------------|-------------|-----------------|
| **ChatML** | `<|im_start|>system` | `<|im_start|>user` | `<|im_start|>assistant` |
| **Llama-3** | `<|start_header_id|>system<|end_header_id|>` | `<|start_header_id|>user<|end_header_id|>` | `<|start_header_id|>assistant<|end_header_id|>` |
| **Alpaca/Vicuna** | `SYSTEM: ` | `USER: ` | `ASSISTANT: ` |

These special tokens are added to the tokenizer vocabulary during chat tuning. The model learns to treat these tokens as structural markers, not semantic content.

---

## 7 · Loss masking — the critical implementation detail

Both instruction tuning and chat tuning use **cross-entropy loss exclusively on the model's output tokens**. Instruction tokens, system prompts, and user turns are masked out — they contribute to the forward pass (as context) but receive zero loss weight.

**Why does this matter?**

If you compute loss on instruction tokens, you are penalizing the model for not perfectly predicting the instruction — but the instruction is given as input, not generated. This creates several problems:

1. **Wasted gradient signal.** The model's capacity is used to memorize instruction phrasing rather than learn to follow instructions well.
2. **Instruction leakage.** The model learns that instruction tokens are "predictable" (since they are always present), which can cause it to confuse instruction and response distributions.
3. **Empirical degradation.** Alpaca (Taori et al. 2023) and subsequent ablations show that computing loss on instruction tokens measurably hurts instruction-following benchmark performance.

**Implementation.** In PyTorch, loss masking is implemented by setting the labels of non-response tokens to `-100`, which is the `ignore_index` of `nn.CrossEntropyLoss`:

```python
labels = input_ids.clone()
# Mask everything up to the start of the response
labels[:, :response_start_idx] = -100
loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
# CrossEntropyLoss skips positions where label == -100
```

For multi-turn chat data, this masking must be applied to **every** non-assistant turn (both system and all user turns), while keeping all assistant turns active.

---

## 8 · Worked numerical example — loss masking in practice

**Setup.** A single training example consists of the following token sequence:

| Segment | Role | Tokens | Loss weight |
|---------|------|--------|-------------|
| `<|im_start|>system\nYou are a helpful assistant.<|im_end|>` | system | 8 tokens | 0 (masked) |
| `<|im_start|>user\nWhat is 7 × 8?<|im_end|>` | user | 10 tokens | 0 (masked) |
| `<|im_start|>assistant\nThe answer is 56.<|im_end|>` | assistant | 9 tokens | 1 (active) |

**Total tokens:** 27. **Active tokens:** 9.

**Per-token cross-entropy.** Suppose the model assigns the following log-probabilities to each of the 9 assistant tokens:

| Position | Token | $\log P_\theta$ |
|----------|-------|-----------------|
| 1 | `The` | $-0.18$ |
| 2 | ` answer` | $-0.22$ |
| 3 | ` is` | $-0.15$ |
| 4 | ` 56` | $-0.61$ |
| 5 | `.` | $-0.09$ |
| 6 | `<|im_end|>` | $-0.11$ |
| 7–9 | (padding tokens, masked) | — |

Active positions: 6 (tokens 1–6).

**Average cross-entropy over active tokens:**

$$\mathcal{L} = -\frac{1}{6}(0.18 + 0.22 + 0.15 + 0.61 + 0.09 + 0.11) = -\frac{1.36}{6} \approx 0.227 \text{ nats}$$

Converting to bits: $0.227 / \ln 2 \approx 0.327$ bits per token.

**Comparison: what if we had included all 27 tokens?**

If loss were computed over all 27 tokens and the model assigns high confidence to the input tokens (e.g., average $\log P \approx -0.05$ for predictable system/user text):

$$\mathcal{L}_{\text{wrong}} = -\frac{1}{27}(18 \times 0.05 + 9 \times 1.36/9) \approx -\frac{0.90 + 1.36}{27} \approx 0.084$$

The artificially low loss from the easy-to-predict instruction tokens would dominate, masking the signal from the hard-to-predict assistant tokens and leading the optimizer to allocate capacity toward predicting instructions rather than generating correct responses.

> [!NOTE]
> In practice, the FLAN paper reports that loss masking on instructions (keeping only response loss) improves few-shot performance by 2–4 points across standard NLP benchmarks. Alpaca ablations confirm a similar effect in instruction-following settings.

---

## 9 · Key differences — comparison table

| Aspect | Instruction Tuning | Chat Tuning |
|--------|-------------------|-------------|
| **Turn structure** | Single-turn | Multi-turn |
| **System prompt** | Optional / absent | Typically present |
| **Data format** | `instruction → response` | `[SYS] [USER] [ASST] [USER] [ASST]…` |
| **Goal** | Zero-shot task generalization | Conversational assistant behavior |
| **Coreference** | Not required | Required across turns |
| **Special tokens** | Simple delimiters or none | Role-specific tokens (ChatML, Llama-3) |
| **Loss scope** | Response tokens only | Assistant turn tokens only |
| **Dataset examples** | FLAN, T0, Super-Natural Instructions | ShareGPT, OpenAssistant, WizardLM |
| **Persona control** | Not present | Via system prompt |
| **Training set size** | Thousands to millions of tasks | Hundreds of thousands of conversations |

**The relationship between the two.** Chat tuning is a strict superset of instruction tuning: any single-turn (instruction, response) example is a degenerate chat example with no system prompt and one user+assistant turn. In practice, modern models go through instruction tuning first (or combine both in one SFT stage), then may receive further RLHF/DPO alignment specifically tuned on chat-quality signals.

---

## 10 · Common misconceptions

| Misconception | Reality |
|---|---|
| "Instruction tuning teaches the model new knowledge." | Instruction tuning does not inject new facts — it elicits and reorganizes knowledge already present from pretraining. A model cannot instruction-tune its way to knowing facts it never saw during pretraining. |
| "Chat tuning is just instruction tuning with more examples." | Chat tuning requires structural changes: multi-turn context in the input, role-specific special tokens, and masking of all non-assistant turns. A model trained only on single-turn instruction data will not automatically handle multi-turn coherence. |
| "Loss on instruction tokens is harmless." | Empirically incorrect. Loss on instruction tokens dilutes gradient signal and measurably degrades instruction-following quality. The `-100` label mask is not a minor implementation detail — it is load-bearing. |
| "The system prompt is trained away — the model ignores it." | The opposite: chat tuning explicitly trains the model to respect system prompt constraints throughout the conversation. Jailbreaks that override system prompts succeed not because system prompts are ignored during training but because the model's instruction-following is not perfectly robust. |
| "FLAN proved 62 tasks is the magic number." | 62 was the size of the original FLAN mixture; larger instruction sets (Super-Natural Instructions: 1,616 tasks; FLAN-v2: 1,800+ tasks) continue to improve performance. Saturation happens at different task counts for different model sizes. |
| "Instruction tuning and SFT are different things." | SFT (supervised fine-tuning) is the training procedure; instruction tuning is a specific use case of SFT with instruction-formatted data. Chat tuning is another use case of SFT with multi-turn chat data. |

---

## 11 · Interview drill — follow-up questions

<details>
<summary><b>Q: Why does task diversity matter more than task volume in instruction tuning?</b></summary>

The model needs to learn the *meta-skill* of following instructions, not to memorize the answers for specific tasks. A model that sees 100,000 sentiment analysis examples learns to do sentiment analysis well, but has not generalized the instruction-following pattern. A model that sees 1,000 examples each from 100 different task types must find a common pattern — "this is an instruction demanding a specific output type" — and that pattern generalizes to new task types the model has never seen. This is the fundamental insight of FLAN: diversity in the training tasks is what drives zero-shot generalization, not quantity within any single task.
</details>

<details>
<summary><b>Q: What happens if you compute cross-entropy loss on the instruction tokens during instruction tuning?</b></summary>

Two problems. First, gradient dilution: the model's gradient signal is split between "predict the instruction correctly" and "generate the right response," where the first objective is much easier (instructions are highly structured and repetitive) and will dominate. The model allocates capacity toward memorizing instruction templates rather than learning to follow instructions. Second, the instruction tokens are *given* — they are conditioning context, not something the model should be generating. Treating them as targets confuses the model about the generation task. Empirically, masking instruction tokens and keeping only response-token loss improves benchmark performance by 2–5 points. In code, this is implemented by setting instruction token labels to `-100`, the `ignore_index` value of `nn.CrossEntropyLoss`.
</details>

<details>
<summary><b>Q: A model trained with instruction tuning is given a multi-turn conversation. How does it behave compared to a chat-tuned model?</b></summary>

An instruction-tuned model (without chat tuning) will treat the multi-turn conversation as a long single-turn instruction. It may not understand the role boundaries — it has never seen system prompts or the `[USER] / [ASSISTANT]` delimiter tokens. It might: (1) continue generating as if the conversation history is all context for a single request, (2) fail to follow system prompt constraints (since it was never trained to distinguish system from user text), (3) generate inconsistently across turns (since it lacks the multi-turn coherence training signal). A chat-tuned model explicitly learned from examples where the conversation history is the context and only the current assistant turn is the target.
</details>

<details>
<summary><b>Q: FLAN trains on 60 task clusters and tests on 2 held-out ones. Why is this zero-shot?</b></summary>

Zero-shot here means the model has never seen any examples of the held-out task during fine-tuning. It has seen the *task format* (instruction → response) extensively across 60 other tasks, and it must generalize this pattern to the 2 new tasks without any in-context demonstrations from those tasks. This is distinct from zero-shot in the pretraining sense (where the model has never seen instruction-formatted data at all). FLAN's zero-shot is specifically zero-shot within the instruction-following paradigm.
</details>

<details>
<summary><b>Q: What is the difference between ChatML and the Llama-3 chat format?</b></summary>

Both use the same conceptual structure — role markers that delimit system, user, and assistant turns — but with different special tokens. ChatML (OpenAI / Mistral / Qwen) uses `<|im_start|>role\ncontent<|im_end|>`. Llama-3 (Meta) uses `<|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>`. The semantic meaning is identical; the difference is purely in the tokenizer vocabulary and special token IDs. A model trained on one format cannot be used directly with a conversation structured in another format — the special tokens will be interpreted as unknown tokens or regular text. This is why chat templates (e.g., Hugging Face's `tokenizer.apply_chat_template()`) are important: they handle the format-specific serialization automatically.
</details>

<details>
<summary><b>Q: Instruction tuning does not require human demonstrations — what is self-instruct?</b></summary>

Self-instruct (Wang et al. 2022) is a method that bootstraps instruction data from the model itself. Starting with 175 seed tasks written by humans, the model is prompted to generate new tasks, then to generate responses to those tasks. The generated (instruction, response) pairs are filtered for quality and deduplicated, then used to fine-tune the model. Alpaca uses a variant: 52K instructions generated by GPT-3.5 from 175 seed tasks. The advantage is scale — you can generate millions of instructions without human annotation. The risk is quality degradation through model-generated noise and potential circular reinforcement of model errors.
</details>

---

## 12 · Connections to other concepts

**SFT and RLHF.** Instruction tuning and chat tuning are both SFT stages. In the RLHF pipeline (InstructGPT, Ouyang et al. 2022), SFT on instruction data is Stage 1, before reward model training (Stage 2) and PPO (Stage 3). The SFT stage establishes the baseline instruction-following capability that the RL stage then refines toward human preferences.

**LIMA hypothesis.** Li et al. (2023) showed that 1,000 carefully curated instruction examples produce a model comparable in many respects to models trained on 52K (Alpaca). This "less is more alignment" result suggests that *quality* of instruction-response pairs matters more than quantity — echoing the FLAN result that task *diversity* matters more than task *volume*.

**Prompt engineering.** Instruction tuning changes what kind of prompts work well. An instruction-tuned model responds best to imperative, task-specific prompts ("Summarize…", "Translate…"). A raw pretrained model responds best to few-shot demonstrations (in-context learning). Prompt engineering practice shifted significantly once instruction-tuned models became standard.

**System prompts and safety.** System prompts are the primary mechanism by which operators customize chat-tuned models (e.g., "Do not discuss competitor products"). Safety training (RLHF, Constitutional AI) builds on the chat-tuned model's ability to follow system prompt instructions, teaching it to recognize and comply with safety-relevant constraints specified there.

**DPO and preference learning.** DPO and RLHF both fine-tune from an SFT checkpoint (an instruction- or chat-tuned model). The SFT model provides the reference policy $\pi_{\text{ref}}$ and the initialization point for the RL policy. A weak SFT model leads to weak RLHF/DPO results — the preference learning stage can only improve relative to what the SFT stage established.

---

## 13 · One-screen summary

```
INSTRUCTION TUNING
━━━━━━━━━━━━━━━━━
Input:  <instruction text>
Output: <response>

Loss:   only on response tokens
Goal:   zero-shot task generalization
Data:   FLAN, T0, Super-Natural Instructions, Alpaca
Key insight: diverse tasks > many examples of one task

CHAT TUNING
━━━━━━━━━━━
<SYSTEM> persona + constraints
<USER>   first message
<ASST>   ← loss computed here
<USER>   follow-up
<ASST>   ← loss computed here
...

Loss:   only on assistant turn tokens
Goal:   multi-turn conversational assistant
Data:   ShareGPT, OpenAssistant, WizardLM
Adds:   system prompt, conversation history, role tokens

SHARED MECHANICS
━━━━━━━━━━━━━━━
Both: cross-entropy SFT on top of pretrained model
Both: mask non-output tokens (set labels = -100)
Both: SFT stage in the RLHF pipeline

KEY DIFFERENCE
━━━━━━━━━━━━━
Instruction tuning = single-turn task follower
Chat tuning = multi-turn interactive assistant
                (superset of instruction tuning)
```

---

## 14 · Five-minute refresher

**Why instruction tuning at all?**
A pretrained model continues text — it doesn't follow commands. Instruction tuning teaches it the pattern: "when I see an instruction, produce the appropriate output type." Training on diverse tasks generalizes this pattern to new instructions at test time.

**The FLAN key result.** More diverse tasks → better zero-shot generalization. Not more examples of the same tasks — different *types* of tasks. This plateaus around 60–100 task clusters for GPT-scale models.

**What chat tuning adds.** System prompts (persona/constraints), conversation history (multi-turn context), and role-specific special tokens (`<|im_start|>`, `<|eot_id|>`, etc.). The model learns to track conversation context and respect system-level instructions throughout a dialogue.

**The one implementation rule.** Compute loss *only on the model's own output tokens*. In instruction tuning: only the response. In chat tuning: only the assistant turns. Set instruction/user/system labels to `-100`. Forgetting this measurably degrades performance.

**Where instruction tuning fits in the pipeline.** It is SFT Stage 1. Everything else — reward modeling, RLHF, DPO — builds on top of an instruction- or chat-tuned model. A bad SFT base = bad alignment outcomes downstream.

---

## 15 · References

1. Wei, J. et al. — **Finetuned Language Models Are Zero-Shot Learners** (FLAN). *ICLR 2022 / arXiv:2109.01652.* — Introduces instruction tuning on 62 task clusters; establishes that task diversity drives zero-shot generalization.

2. Sanh, V. et al. — **Multitask Prompted Training Enables Zero-Shot Task Generalization** (T0). *ICLR 2022 / arXiv:2110.08207.* — Parallel to FLAN; uses 35 datasets from PromptSource; T0 11B outperforms GPT-3 175B zero-shot on many benchmarks.

3. Wang, Y. et al. — **Super-Natural Instructions: Generalization via Declarative Instructions on 1600+ NLP Tasks**. *EMNLP 2022 / arXiv:2204.07705.* — Scales instruction tuning to 1,616 tasks; introduces the instruction schema (definition + examples + instance).

4. Ouyang, L. et al. — **Training Language Models to Follow Instructions with Human Feedback** (InstructGPT). *NeurIPS 2022 / arXiv:2203.02155.* — The InstructGPT pipeline: SFT on instruction data → reward model → PPO. Establishes instruction tuning as Stage 1 of RLHF.

5. Taori, R. et al. — **Alpaca: A Strong, Replicable Instruction-Following Model**. *Stanford CRFM blog, March 2023.* — Fine-tunes LLaMA 7B on 52K self-instruct examples; ablations confirm loss masking on instructions is important.

6. Wang, Y. et al. — **Self-Instruct: Aligning Language Models with Self-Generated Instructions**. *ACL 2023 / arXiv:2212.10560.* — Self-instruct methodology for bootstrapping instruction data from the model itself; basis for Alpaca data generation.

7. Li, Y. et al. — **LIMA: Less Is More for Alignment**. *NeurIPS 2023 / arXiv:2305.11206.* — 1,000 high-quality instruction examples match or beat 52K Alpaca examples; establishes the superficial alignment hypothesis.

8. Chung, H. et al. — **Scaling Instruction-Finetuned Language Models** (FLAN-v2 / Flan-T5 / Flan-PaLM). *JMLR 2024 / arXiv:2210.11416.* — Scales FLAN to 1,800+ tasks and 540B parameters; shows continued gains with scale.

---

<!-- ============================ BOTTOM NAV ============================ -->
<div align="center">

[⬅️ Q4‑05 — On-Policy vs Off-Policy RL](./q05-on-policy-vs-off-policy.md) &nbsp;|&nbsp; [📚 Back to Section 4](./README.md) &nbsp;|&nbsp; [🏠 Home](../../README.md) &nbsp;|&nbsp; [Q4‑07 — System Prompt vs User Prompt ➡️](./q07-system-prompt-vs-user-prompt.md)

<sub>Found an error or have a sharper intuition? See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a> — answers follow the <a href="../../_TEMPLATE.md">answer template</a>.</sub>

</div>
