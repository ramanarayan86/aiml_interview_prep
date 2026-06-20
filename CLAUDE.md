# CLAUDE.md — GenAI Interview Prep Repo

## Project overview

This is a first-principles ML/LLM interview question bank at `ml-interview-prep/`.
Each question is a standalone Markdown page with 15 sections, SVG figures, LaTeX equations,
PyTorch code, worked numerical examples, and references.

**66 questions answered** across 3 sections. Sections 4–30 are scaffolded.

## Directory structure

```
ml-interview-prep/
├── questions/section-NN-<name>/   ← question files qNN-<slug>.md
├── assets/s<N>qNN/                ← SVGs: fig1-*.svg, fig2-*.svg
└── README.md                      ← main index (badges, ToC, status tables)
```

Asset naming convention:
- Section 1: `assets/qNN/` (legacy)
- Section 2+: `assets/s2qNN/`, `assets/s3qNN/`, etc.

## Answer template (15 sections)

Every answer file must follow this exact structure:

```
1. Top navigation bar
2. Title + badges (tags, read-time)
3. [!IMPORTANT] callout — the 20-second answer
4. Table of contents
5. ## 1 · First principles
6. ## 2 · The core mechanism
7. ## 3 · Figure 1 — <description>  (img tag referencing assets/sNqNN/fig1-*.svg)
8. ## 4 · Step-by-step worked example
9. ## 5 · Figure 2 — <description>  (img tag referencing assets/sNqNN/fig2-*.svg)
10. ## 6 · Algorithm / pseudocode
11. ## 7 · PyTorch reference implementation
12. ## 8 · Worked numerical example
13. ## 9 · Interview drill — follow-up questions
14. ## 10 · Common misconceptions
15. ## 11 · Connections to other concepts
16. ## 12 · One-screen summary
17. ## 13 · Five-minute refresher
18. ## 14 · Further reading
19. ## 15 · Bottom navigation bar
```

## SVG requirements

Every SVG must have:
- `xmlns="http://www.w3.org/2000/svg"`
- `viewBox="0 0 W H"`
- No HTML entities (`&nbsp;` is invalid XML — use literal spaces or `&#160;`)

Color palette: navy #1f4fa3, teal #0e8a6e, amber #c77a12, purple #6c4fe0, gray #5b6675, red #c0392b

## LaTeX conventions

- Display math: `$$...$$`
- Inline math: `$...$`
- Always verify numerical results in Python before writing them into the file

## Git conventions

- Never add `Co-Authored-By` trailers — sole author is Ramanarayan Mohanty
- Commit messages: short imperative title, blank line, bullet details if needed
- Asset directories must exist before committing SVG files

## Answered question counts (update after each session)

| Section | Answered | Total |
|---------|----------|-------|
| §1 Transformer Architecture | 34 | 42 |
| §2 Tokenization and Embeddings | 16 | 32 |
| §3 Pretraining and Scaling Laws | 16 | 32 |
| §4 Post-training: SFT, RLHF, DPO | 2 | 40 |
| **Total** | **68** | **146** |
<!-- Note: Total of 106 originally; §4 adds 40, so 106+40=146 total questions tracked -->

## Key references

- Chinchilla parametric fit: L(N,D) = E + A/N^α + B/D^β; A=406.4, B=410.7, α=0.34, β=0.28, E≈1.69
- Compute approximation: C ≈ 6ND FLOPs
- Compute-optimal ratio: D*/N* ≈ 20 tokens per parameter
- Adam defaults: β₁=0.9, β₂=0.999, ε=1e-8
- BF16 range: ±3.39×10³⁸; FP16 range: ±65504
