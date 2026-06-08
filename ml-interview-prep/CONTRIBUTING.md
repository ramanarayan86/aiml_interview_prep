<div align="center">

[🏠 Home](./README.md)

</div>

---

# Contributing

This repo is designed so that **adding a question is mechanical**: copy the template, fill the blanks, drop in figures, wire up two nav bars. The structure does the rest.

## Add a new answer in 6 steps

1. **Pick the file path.** `questions/section-XX-slug/qNN-short-slug.md` (e.g. `questions/section-07-quantization/q210-gptq.md`).
2. **Copy the template.** Start from [`_TEMPLATE.md`](./_TEMPLATE.md) and replace every `{{PLACEHOLDER}}`. Delete sections that genuinely don't apply, but keep the *20‑second answer*, *worked example*, and *references* — they are the backbone.
3. **Create figures** under `assets/qNN/`. Prefer **SVG** (sharp, diff‑able). Diagrams that are mostly boxes‑and‑arrows can be inline **Mermaid** instead — no asset file needed.
4. **Wire navigation.** Keep the top and bottom nav bars. Fix the prev/next links and the section‑index link.
5. **Register it** in the section's `README.md` table (flip 📝 → ✅ and link the file) and bump the section's `answered N/total` badge.
6. **Open a PR.** One question per PR keeps review easy.

## House style

- **First principles, always.** Assume a smart beginner. Define terms before using them. The test: *could a motivated newcomer teach this back after one read?*
- **Show, then tell.** Lead with the intuition or a figure, then formalize. Every core claim gets an equation, a figure, or a number — not just prose.
- **Math:** GitHub renders `$inline$` and `$$block$$` LaTeX. Use it.
- **Diagrams:** Mermaid for flow/architecture; hand‑authored SVG for conceptual/geometric figures. Keep a light card background so figures read on both GitHub themes.
- **Callouts:** use GitHub alerts — `> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]` — for plain‑English asides, diagnostics, and gotchas.
- **Accuracy &gt; vibes.** Cite primary sources (papers, official blogs) with arXiv IDs. If a claim is contested or version‑dependent, say so.
- **Tone:** precise, warm, confident. No filler. Write the page you wish you'd had the night before the interview.

## Figure palette (for visual consistency)

| Role | Hex |
|---|---|
| Ink / text | `#16202E` |
| Muted text | `#5B6675` |
| Card border | `#E3E6EC` |
| Accent (neutral highlight) | `#2D6CDF` |
| "Good" / the fix | `#0E8A6E` |
| "Problem" / danger | `#E0322E` |
| Warning / amber | `#C77A12` |

## Asset directory naming convention

Asset files (SVGs, PNGs) live under `assets/` at the repo root. The naming convention going forward is:

```
assets/sXqYY/figure-name.svg
```

where `X` is the section number and `YY` is the zero-padded question number within that section.

**Examples:**
- `assets/s1q01/` — Section 1, Question 1 (Transformer block)
- `assets/s2q07/` — Section 2, Question 7 (Token fertility)
- `assets/s6q03/` — Section 6, Question 3 (Speculative decoding)

**Historical note:** Section 1 assets were created before this convention was established and use the shorter `assets/q01/` … `assets/q34/` form. Do not rename them — the question files reference those paths. All new sections (§2 onward) must use `sXqYY/`.

**Figure filenames** within those directories should be descriptive kebab-case:
- `token-types-tradeoff.svg` ✅
- `bpe-merge-sequence.svg` ✅
- `fig1.svg` ✗ (not descriptive)
- `figure_BPE_TRAINING_LOOP.svg` ✗ (underscores, mixed case)

---

<div align="center">

[🏠 Home](./README.md) &nbsp;|&nbsp; [📄 Answer template](./_TEMPLATE.md)

</div>
