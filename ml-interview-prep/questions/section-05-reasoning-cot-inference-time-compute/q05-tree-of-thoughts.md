<!-- ============================ TOP NAV ============================ -->
<div align="center">

[🏠 Home](../../README.md) &nbsp;•&nbsp; [📚 Section 5 — Reasoning &amp; CoT](./README.md) &nbsp;•&nbsp; [⬅️ Q5‑04](./q04-self-consistency.md) &nbsp;•&nbsp; [Q5‑06 ➡️](./q06-system1-system2.md)

</div>

---

# Q5‑05 · What is tree-of-thoughts? How does it differ from chain-of-thought?

<div align="center">

![Section](https://img.shields.io/badge/Section_5-Reasoning_%26_CoT-1f4fa3)
![Level](https://img.shields.io/badge/Level-Basic-0e8a6e)
![Tags](https://img.shields.io/badge/tags-tree--of--thoughts·search·branching·CoT-c77a12)
![Read](https://img.shields.io/badge/read_time-~16_min-5b6675)

</div>

> [!IMPORTANT]
> **The 20-second answer.** Tree-of-Thoughts (ToT, Yao et al., 2023) generalises chain-of-thought from a **linear sequence of reasoning steps to a tree**, where the model generates multiple candidate "thoughts" at each step, evaluates each one (using the model itself as an evaluator), and uses breadth-first or depth-first search to explore and prune the tree. CoT is a special case of ToT with branching factor b=1. Where CoT on the Game of 24 achieves only 4% success, ToT with BFS and b=5 achieves 74%. The key capabilities ToT adds are **backtracking** (abandoning dead-end branches) and **lookahead** (scoring partial solutions before committing to them). The cost is O(b×d) thought evaluations versus O(1) for CoT.

---

## Table of contents

1. [First principles: what CoT cannot do](#1--first-principles-what-cot-cannot-do)
2. [The tree-of-thoughts framework](#2--the-tree-of-thoughts-framework)
3. [Figure 1 — CoT vs. ToT structure](#3--figure-1--cot-vs-tot-structure)
4. [Step-by-step worked example (Game of 24)](#4--step-by-step-worked-example-game-of-24)
5. [Figure 2 — The four ToT modules](#5--figure-2--the-four-tot-modules)
6. [Algorithm / pseudocode](#6--algorithm--pseudocode)
7. [PyTorch reference implementation](#7--pytorch-reference-implementation)
8. [Worked numerical example](#8--worked-numerical-example)
9. [Interview drill — follow-up questions](#9--interview-drill--follow-up-questions)
10. [Common misconceptions](#10--common-misconceptions)
11. [Connections to other concepts](#11--connections-to-other-concepts)
12. [One-screen summary](#12--one-screen-summary)
13. [Five-minute refresher](#13--five-minute-refresher)
14. [Further reading](#14--further-reading)
15. [References](#15--references)

---

## 1 · First principles: what CoT cannot do

Chain-of-thought prompting works by decomposing reasoning into a sequence of intermediate steps. It dramatically improves accuracy on arithmetic and commonsense problems because these tasks have a natural sequential structure: each step logically follows from the previous one, and the correct path is unique.

But not all problems have this structure. Consider tasks that require **search**:

- **Game of 24:** Given four numbers, find an arithmetic expression that evaluates to 24. Many candidate first moves exist; some lead to dead ends.
- **Planning:** Find a sequence of actions to achieve a goal. Wrong action choices must be undone.
- **Constraint satisfaction:** A crossword or sudoku requires filling values that satisfy global constraints; local decisions must sometimes be revised.

For these tasks, CoT fails because:

1. **No backtracking:** CoT generates tokens left-to-right with no mechanism to undo a wrong choice. Once the model writes "4 + 9 = 13, remaining numbers: {13, 10, 13}", it is committed to that branch even if it leads nowhere.
2. **No lookahead:** CoT cannot evaluate whether a partial reasoning path is promising before continuing. It must commit immediately to the next step.
3. **No explicit exploration:** CoT follows a single path. If the model's most probable first move is wrong, the entire computation fails.

Tree-of-Thoughts directly addresses all three limitations by adding a **tree search** layer on top of the language model.

---

## 2 · The tree-of-thoughts framework

**Framework definition (Yao et al., 2023, NeurIPS 2023, arXiv:2305.10601):**

ToT decomposes reasoning into four modules:

| Module | Role | Key question |
|--------|------|-------------|
| **Thought decomposer** | Define the unit of a "thought" | What constitutes one reasoning step? |
| **Thought generator** | Generate $b$ candidate thoughts at each node | What are the possible next steps? |
| **State evaluator** | Score each partial reasoning path | Is this branch promising or a dead end? |
| **Search algorithm** | Decide which nodes to expand | BFS or DFS? |

**Formal definition:**

- Let $s$ be a problem state (the context so far)
- $G(s)$ generates a set of $b$ candidate next thoughts: $\{t_1, \ldots, t_b\} = G(s)$
- $V(s, t)$ evaluates a candidate thought and returns a score or label from \{sure, maybe, impossible\}
- Search explores the tree of states, pruning branches labelled "impossible"

CoT is the degenerate case with $b = 1$, $V$ always returns "maybe", and no backtracking.

**Key empirical results (Yao et al., 2023):**

| Task | CoT success | ToT (BFS, b=5) | Improvement |
|------|------------|----------------|------------|
| Game of 24 | 4% | 74% | +70 pp |
| Creative writing | lower coherence | higher constraint satisfaction | qualitative |
| Crossword (word-level) | 16% | 60% | +44 pp |

---

## 3 · Figure 1 — CoT vs. ToT structure

<div align="center">
<img src="../../assets/s5q05/fig1-cot-vs-tot-structure.svg" alt="Side-by-side comparison of chain-of-thought and tree-of-thoughts structures. Left panel labeled Chain-of-Thought: a vertical chain of five boxes connected by downward arrows, from Prompt through Thought 1, Thought 2, Thought 3 to Answer. Annotations note no backtracking, one forward pass, O(L) thoughts, and Game of 24 success rate of 4%. Right panel labeled Tree-of-Thoughts: a tree starting from a Prompt root, branching into three level-1 nodes T1-a, T1-b, and T1-c. T1-c is crossed out in red and labeled impossible. T1-a branches further into T2-a1 (labeled sure, with a checkmark) and T2-a2 (labeled impossible, crossed out). T1-b branches into T2-b1 (labeled maybe) and T2-b2 (impossible). T2-a1 leads to an Answer Found box. Annotations note backtracking via DFS and BFS, O(b times d) thought evaluations, explicit self-evaluation, and Game of 24 success rate of 74%." width="94%">
<br><sub><b>Figure 1.</b> Left: CoT follows a single linear path from prompt to answer — no backtracking, no exploration. Right: ToT builds a tree; the evaluator labels each partial state as sure (&#10004;), maybe (&#8776;), or impossible (&#10008;); impossible branches are pruned. Only the most promising branch is expanded to the final answer. CoT is the b=1 special case of ToT with no evaluation or pruning.</sub>
</div>

---

## 4 · Step-by-step worked example (Game of 24)

**Task:** Combine the numbers {4, 9, 10, 13} using arithmetic operations (+, -, ×, ÷) to reach exactly 24. Each number must be used exactly once.

**Ground truth:** $13 - 9 = 4$, then $(10 - 4) \times 4 = 6 \times 4 = 24$ ✓. Or: $4 \times 9 = 36$; $36 - 13 + 1$... let's verify the first: $(10 - 4) \times (13 - 9) = 6 \times 4 = 24$ ✓.

**Verify:** $13 - 9 = 4$. $10 - 4 = 6$. $6 \times 4 = 24$. Uses each of {4, 9, 10, 13} exactly once. ✓

---

**ToT BFS execution (breadth b=3 shown):**

**Level 0 (root):** State = {4, 9, 10, 13}

**Level 1 — generate 3 candidate first operations:**

| Candidate | New state | Evaluator says |
|-----------|-----------|---------------|
| $4 \times 9 = 36$ | {36, 10, 13} | sure — large product, many paths |
| $13 - 9 = 4$ | {4, 4, 10} | sure — small numbers, $(10-4)\times4=24$ |
| $10 - 9 = 1$ | {1, 4, 13} | impossible — hard to reach 24 from {1,4,13} |

The "impossible" branch $\{1, 4, 13\}$ is pruned.

**Level 2 — expand {4, 4, 10}:**

| Candidate | New state | Evaluator says |
|-----------|-----------|---------------|
| $4 \times 4 = 16$ | {16, 10} | maybe — 16+10=26 close but not 24 |
| $10 - 4 = 6$ | {6, 4} | sure — $6 \times 4 = 24$ |
| $4 + 4 = 8$ | {8, 10} | impossible |

Expand {6, 4}: $6 \times 4 = 24$ ✓ **Solution found.**

**Full solution trace:**
$$13 - 9 = 4 \Rightarrow \{4, 4, 10\} \Rightarrow 10 - 4 = 6 \Rightarrow \{6, 4\} \Rightarrow 6 \times 4 = 24 \checkmark$$

CoT would have generated a single left-to-right path. If its first move was $10 - 9 = 1$, it would be stuck with {1, 4, 13} and fail, with no ability to backtrack to try $13 - 9 = 4$ instead.

---

## 5 · Figure 2 — The four ToT modules

<div align="center">
<img src="../../assets/s5q05/fig2-tot-algorithm-steps.svg" alt="Diagram showing the four modules of tree-of-thoughts arranged in a two-by-two grid with a central LLM backbone circle. Top-left box labeled Module 1 Thought Decomposer in navy, explaining that you define the unit of a thought: one arithmetic operation for Game of 24, one paragraph for essay writing, one word fill for crossword, one action step for planning. Top-right box labeled Module 2 Thought Generator in teal, explaining two strategies: Strategy A sample (call LLM b times independently, more diverse, b forward passes per node) and Strategy B propose (generate b distinct next steps in one prompt, cheaper, less diverse); breadth b is a key hyperparameter typically b=5. Bottom-left box labeled Module 3 State Evaluator in purple, explaining that you prompt the model to rate if a partial solution leads to a valid answer using labels sure, maybe, or impossible; uses the LLM itself as the evaluator. Bottom-right box labeled Module 4 Search Algorithm in amber, explaining BFS keeps top-k states at each level, used for Game of 24 with b=5 kept per level, and DFS explores one branch to depth d with backtracking, used for crosswords with pruning on impossible; cost is O(b times d) thought evaluations total. A results strip at the bottom shows Game of 24: CoT 4%, ToT BFS b=5: 74%, improvement plus 70 pp." width="94%">
<br><sub><b>Figure 2.</b> The four ToT modules and their interactions through the shared LLM backbone. Module 1 (navy) defines the thought granularity for the task. Module 2 (teal) generates b candidates per node using sampling or proposing. Module 3 (purple) evaluates each partial path using the LLM prompted to rate sure / maybe / impossible. Module 4 (amber) selects BFS or DFS based on the problem type. All four modules call the same underlying language model.</sub>
</div>

---

## 6 · Algorithm / pseudocode

```
TREE-OF-THOUGHTS(problem, model M, breadth b, depth d, search_type)
────────────────────────────────────────────────────────────────────
  Input:  problem — problem statement
          M       — language model (used as generator AND evaluator)
          b       — branching factor (thoughts per node)
          d       — maximum search depth
          search_type — "BFS" or "DFS"

  Output: solution — the first valid complete path found

  ── BFS VERSION ──────────────────────────────────────────────────
  BFS(problem, M, b, d):
    frontier ← [{state: problem, path: []}]   # initial single state

    For level = 1 to d:
      candidates ← []
      For each state s in frontier:
        thoughts ← GENERATE_THOUGHTS(M, s, b)  # module 2
        For each thought t in thoughts:
          new_state ← s + t                    # append thought
          score ← EVALUATE(M, new_state)       # module 3
          If score == "sure" and COMPLETE(new_state):
            Return new_state.path + [t]        # solution found
          If score != "impossible":
            candidates.append((new_state, score))

      # Keep top-b candidates by score
      frontier ← top_b_by_score(candidates, b)

    Return BEST(frontier)                      # best partial solution

  ── DFS VERSION ──────────────────────────────────────────────────
  DFS(state, path, M, b, d, depth):
    If depth == 0: Return (path, EVALUATE(M, state))
    thoughts ← GENERATE_THOUGHTS(M, state, b)    # module 2
    For each thought t in sorted_by_score(thoughts, M):
      score ← EVALUATE(M, state + t)             # module 3
      If score == "impossible": Continue          # prune
      result ← DFS(state+t, path+[t], M, b, d, depth-1)
      If result is valid: Return result
    Return FAILURE                               # backtrack

  ── THOUGHT GENERATION ───────────────────────────────────────────
  GENERATE_THOUGHTS(M, state, b):
    # Strategy A (sampling):
    thoughts ← [M.sample(state, temperature=0.7) for _ in range(b)]
    # Strategy B (proposing):
    # thoughts ← M.generate(state + "List " + str(b) + " next steps:")
    Return [extract_thought(t) for t in thoughts]

  ── EVALUATION ───────────────────────────────────────────────────
  EVALUATE(M, state):
    prompt ← state + "\nRate this partial solution: sure/maybe/impossible"
    response ← M.generate(prompt, temperature=0.0)  # greedy for evaluation
    Return parse_label(response)  # "sure" | "maybe" | "impossible"

────────────────────────────────────────────────────────────────────
Complexity (BFS):  O(b × d) thought generations + O(b × d) evaluations
Complexity (DFS):  O(b × d) in the worst case; O(d) in the best case
```

**BFS vs. DFS tradeoffs:**

| Property | BFS | DFS |
|----------|-----|-----|
| Guarantees shortest path | Yes | No |
| Memory | O(b^depth) | O(depth) |
| Backtracking | Level-by-level | Step-by-step |
| Best for | Game of 24 | Crosswords |
| Early solution | After full level | Immediately on first leaf |

---

## 7 · PyTorch reference implementation

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ThoughtNode:
    state: str                        # full context including thoughts so far
    thoughts: list[str] = field(default_factory=list)  # path of thoughts
    score: str = "maybe"              # "sure" | "maybe" | "impossible"


def generate_thoughts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    state: str,
    b: int,
    temperature: float = 0.7,
    max_new_tokens: int = 64,
) -> list[str]:
    """
    Strategy A: sample b thoughts independently.
    Returns a list of b thought strings.
    """
    prompt_ids = tokenizer(state, return_tensors="pt").input_ids
    prompt_ids = prompt_ids.to(next(model.parameters()).device)
    thoughts = []
    with torch.no_grad():
        for _ in range(b):
            out = model.generate(
                prompt_ids,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            thought_text = tokenizer.decode(
                out[0, prompt_ids.shape[1]:], skip_special_tokens=True
            )
            # Extract only the first line (one thought)
            thought_text = thought_text.split("\n")[0].strip()
            thoughts.append(thought_text)
    return thoughts


def evaluate_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    state: str,
    eval_prompt_template: str,
) -> str:
    """
    Prompt the model to evaluate the partial state.
    Returns one of: "sure", "maybe", "impossible"
    """
    eval_prompt = eval_prompt_template.format(state=state)
    prompt_ids = tokenizer(eval_prompt, return_tensors="pt").input_ids
    prompt_ids = prompt_ids.to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            do_sample=False,           # greedy for evaluation (deterministic)
            max_new_tokens=8,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        out[0, prompt_ids.shape[1]:], skip_special_tokens=True
    ).strip().lower()
    if "sure" in response:
        return "sure"
    elif "impossible" in response:
        return "impossible"
    return "maybe"


def tree_of_thoughts_bfs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: str,
    b: int = 5,
    depth: int = 3,
    eval_prompt_template: str = (
        "Evaluate this partial reasoning:\n{state}\n"
        "Does this lead to a valid solution? Answer: sure/maybe/impossible"
    ),
    is_complete: Callable[[str], bool] | None = None,
) -> str | None:
    """
    BFS Tree-of-Thoughts: keep top-b nodes per level.

    Args:
        problem:    problem statement string
        b:          branching factor
        depth:      maximum search depth
        is_complete: function to check if a state is a complete solution

    Returns:
        Solution path string, or None if no solution found.
    """
    # Score priority: sure > maybe > impossible
    score_rank = {"sure": 2, "maybe": 1, "impossible": 0}

    frontier: list[ThoughtNode] = [ThoughtNode(state=problem)]

    for level in range(1, depth + 1):
        next_candidates: list[ThoughtNode] = []

        for node in frontier:
            thoughts = generate_thoughts(model, tokenizer, node.state, b)

            for thought in thoughts:
                new_state = node.state + "\n" + thought
                score = evaluate_state(model, tokenizer, new_state, eval_prompt_template)

                if score == "impossible":
                    continue

                new_node = ThoughtNode(
                    state=new_state,
                    thoughts=node.thoughts + [thought],
                    score=score,
                )

                # Check for complete solution
                if is_complete and is_complete(new_state):
                    return new_state

                next_candidates.append(new_node)

        if not next_candidates:
            return None  # All branches pruned

        # Keep top-b by score
        next_candidates.sort(key=lambda n: score_rank[n.score], reverse=True)
        frontier = next_candidates[:b]

    # Return the best remaining node
    return frontier[0].state if frontier else None


# ── Usage sketch for Game of 24 ────────────────────────────────────────────────
GAME24_THOUGHT_PROMPT = """
Numbers: {numbers}
Possible operations with these numbers:
"""

GAME24_EVAL_PROMPT = """
Numbers remaining: {state}
Goal: reach 24 using all remaining numbers.
Rate the likelihood of success: sure / maybe / impossible
"""
```

> [!NOTE]
> The eval prompt is the most task-specific component. For Game of 24, the evaluator is prompted to reason about whether the remaining numbers can reach 24. For planning tasks, it is prompted to evaluate if the current plan prefix is feasible. Yao et al. (2023) report that using a weak evaluator (or no evaluator) significantly degrades ToT performance, confirming that the state evaluation module drives most of the gain.

---

## 8 · Worked numerical example

**Game of 24 with input {4, 9, 10, 13}. Goal: reach 24.**

**Step 1: Verify the solution.**

$$13 - 9 = 4 \Rightarrow \{4, 4, 10\}$$

$$10 - 4 = 6 \Rightarrow \{6, 4\}$$

$$6 \times 4 = 24 \checkmark$$

Check: used 4, 9, 10, 13 each exactly once. ✓ Result equals 24. ✓

**Step 2: Trace CoT failure mode.**

CoT greedy path (typical failure):

```
Step 1: 10 - 9 = 1. Remaining: {1, 4, 13}
Step 2: 13 - 4 = 9. Remaining: {1, 9}
Step 3: 9 × 1 = 9. Remaining: {9} — not 24. FAIL.
```

No mechanism to backtrack and try $13 - 9 = 4$ first.

**Step 3: Trace ToT BFS (breadth b=3).**

| Level | State | Candidates generated | Evaluator | Action |
|-------|-------|---------------------|-----------|--------|
| 0 | {4,9,10,13} | — | — | Root |
| 1 | {4,9,10,13} | $4\times9=36$ → {36,10,13} | sure | Keep |
| 1 | {4,9,10,13} | $13-9=4$ → {4,4,10} | sure | Keep |
| 1 | {4,9,10,13} | $10-9=1$ → {1,4,13} | impossible | Prune |
| 2 | {36,10,13} | $36-13=23$ → {23,10} | maybe | Keep |
| 2 | {36,10,13} | $13\times10=130$ → {36,130} | impossible | Prune |
| 2 | {4,4,10} | $10-4=6$ → {6,4} | sure | Keep |
| 3 | {6,4} | $6\times4=24$ | **sure — solution** | Return |

**Total thought evaluations:** 3 (level 1) + 3 (level 2) + 1 (level 3) = 7.

**CoT cost:** 1 generation (fails).

**ToT cost:** 7 thought generations + 7 evaluations. But success rate improves from 4% to 74%.

**Cost-normalised perspective:** For the 74 out of 100 problems where CoT would have required many retries (or given wrong answers), ToT spends ~7× the inference compute and reliably finds solutions.

---

## 9 · Interview drill — follow-up questions

<details>
<summary><b>Q: What makes Game of 24 specifically hard for CoT?</b></summary>

Game of 24 has all the properties that defeat CoT: (1) **combinatorial search space** — there are many possible first operations, and most lead to dead ends, so a single greedy path is almost certain to fail; (2) **reversibility requirement** — once you commit to an operation, you cannot undo it in a left-to-right generation; (3) **no incremental signal** — you cannot tell whether a partial set of remaining numbers is promising without lookahead; (4) **exact constraint** — the answer must be exactly 24, not approximately 24, so near-misses do not count. These are exactly the conditions that BFS tree search is designed for.

</details>

<details>
<summary><b>Q: The evaluator in ToT is the same LLM — how reliable is self-evaluation?</b></summary>

Self-evaluation quality is the critical bottleneck of ToT. Yao et al. (2023) show that ToT with oracle evaluation (a human or verifier tells you sure/impossible) is significantly better than ToT with LLM self-evaluation. Self-evaluation degrades when: (1) the problem requires computation that the model cannot reliably do in its head (e.g., complex arithmetic); (2) the partial state is too early in the tree to determine feasibility; (3) the LLM has limited problem-specific knowledge. For tasks where self-evaluation is reliable (natural language constraint checking, broad feasibility judgments), ToT works well. For tasks requiring precise computation, a symbolic verifier or code executor is a much better evaluator than another LLM forward pass.

</details>

<details>
<summary><b>Q: How does ToT compare to self-consistency? When do you use which?</b></summary>

They are complementary strategies for different problem types:

**Use self-consistency** when: problems have unique discrete answers, many valid reasoning paths exist, errors are random (not systematic), and you want a simple no-training-required approach.

**Use ToT** when: problems require search (combinatorial, planning), single paths almost always fail, dead-end detection is possible, and you can tolerate higher inference-time cost.

Key structural difference: self-consistency samples complete paths and votes at the end; ToT evaluates partial paths and prunes during generation. For a 4-step problem, self-consistency generates N×4 tokens and then votes; ToT may generate far fewer tokens if early pruning cuts most branches.

</details>

<details>
<summary><b>Q: What are Graph-of-Thoughts and other extensions of ToT?</b></summary>

Several papers extend ToT beyond trees:

- **Graph-of-Thoughts (Besta et al., 2024, AAAI):** Allows thoughts to be combined and merged, not just branched — thought nodes can have multiple parents. Useful for problems that require synthesising insights from different branches (e.g., aggregation, sorting).
- **Cumulative Reasoning (Zhang et al., 2023):** Adds a "proposer" and "verifier" split, where one LLM call generates candidates and another verifies; cleaner separation than self-evaluation.
- **Reasoning via Planning (RAP, Hao et al., 2023):** Uses Monte Carlo Tree Search (MCTS) with a reward signal to guide ToT exploration. Replaces the LLM's self-evaluation with a value function estimated by MCTS rollouts.
- **Long (2023):** Introduces a separate "controller" LLM that guides the ToT search, separating generation from meta-reasoning.

The general trend: as problems get harder, self-evaluation becomes less reliable, and external verifiers or value functions (MCTS, code execution) are needed.

</details>

<details>
<summary><b>Q: What is the actual computational cost of ToT vs. CoT in practice?</b></summary>

For a depth-$d$ tree with branching factor $b$:

- **CoT:** 1 generation of length $L$.
- **ToT (BFS, keeping top-$b$):** $b \times d$ thought generations + $b \times d$ evaluation calls.

For the Yao et al. (2023) Game of 24 experiment with $b=5$, $d=3$: approximately $5 \times 3 = 15$ thought generations + 15 evaluations = 30 LLM calls per problem, versus 1 for CoT. However, the 74% success rate versus 4% means that for 100 problems, CoT needs ~25× the attempts (retrying until it gets right answers) while ToT nearly always solves on the first tree. The amortised cost is similar.

</details>

<details>
<summary><b>Q: Why is the choice of what counts as a "thought" so important?</b></summary>

The thought granularity determines the search space size and the evaluator's ability to assess feasibility. Too fine-grained (e.g., one token per thought): the tree is too deep and wide; evaluation becomes meaningless at that scale. Too coarse-grained (e.g., the entire solution in one thought): ToT collapses back to CoT. The right granularity is problem-specific:

- Game of 24: one arithmetic operation (reduces 4 numbers to 3, then to 2, then to 1)
- Creative writing with constraints: one passage (evaluator checks constraint satisfaction so far)
- Multi-hop QA: one retrieval-reasoning step

In Yao et al. (2023), the thought decomposition is explicitly designed for each task, and the authors note that it requires domain knowledge about the problem structure. This is a design cost that CoT does not incur.

</details>

---

## 10 · Common misconceptions

| Misconception | Reality |
|---|---|
| "ToT always outperforms CoT." | For sequential problems with clear paths (arithmetic, commonsense QA), CoT + self-consistency matches or exceeds ToT at lower cost. ToT's advantage is specific to search-heavy tasks. |
| "ToT requires a separate verifier model." | No. The original ToT uses the same LLM prompted to evaluate partial states. A separate verifier is an extension that improves reliability but is not required. |
| "DFS is better than BFS for ToT." | Neither is universally better. BFS finds optimal solutions; DFS is more memory-efficient and finds solutions faster when the first good branch happens to be correct. Yao et al. use BFS for Game of 24 and DFS for crosswords. |
| "ToT is just self-consistency with pruning." | Different mechanism. Self-consistency samples N complete paths and votes. ToT evaluates partial paths and prunes during generation — it never generates the complete failed branches. |
| "The 74% Game of 24 result is the LLM's inherent capability." | No. The evaluator is the key. Remove the state evaluator (random pruning) and ToT drops back near CoT. The LLM's ability to evaluate "is {6,4} reachable to 24?" is what drives the result. |
| "ToT requires retraining." | No. It is a pure prompting strategy over a frozen LLM. No gradient updates, no fine-tuning, no adapter layers. |

---

## 11 · Connections to other concepts

**Self-Consistency (Q5-04):** Self-consistency samples complete paths; ToT evaluates partial paths. Self-consistency is variance reduction through output voting; ToT is variance reduction through search-space pruning. They target different failure modes: self-consistency handles random single-path errors; ToT handles structural dead ends.

**Chain-of-Thought (Q5-01, Q5-03):** CoT is the b=1, no-evaluation degenerate case of ToT. The full hierarchy is: Standard prompting → Chain-of-Thought → Self-Consistency → Tree-of-Thoughts → MCTS-guided reasoning.

**Inference-time compute scaling (Q5-01):** ToT is a structured way to spend inference-time compute — not randomly (as in self-consistency) but guided by the evaluator. This is analogous to classical best-first search versus random restart.

**Reinforcement Learning / MCTS (Q3-08):** ToT's BFS/DFS can be replaced with Monte Carlo Tree Search for problems with sparse rewards, turning the LLM into the policy and value networks of AlphaGo-style systems. RAP (Hao et al., 2023) makes this connection explicit.

**Graph-of-Thoughts (Besta et al., 2024):** Extends ToT to allow merging of thought branches, enabling aggregation and refinement operations not possible in a tree. The generalisation is: Standard → Chain → Tree → Graph → Hypergraph of Thoughts.

**Reward models and verifiers (Q4-07):** The ToT state evaluator plays the same role as a process reward model — assigning value to intermediate states. A trained process reward model (Lightman et al., 2023) is a stronger evaluator than a prompted LLM, and can be plugged into the ToT framework as Module 3.

---

## 12 · One-screen summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      TREE-OF-THOUGHTS — ONE-SCREEN SUMMARY                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  FRAMEWORK: 4 MODULES                                                            │
│    1. Thought Decomposer   — define unit of reasoning step (task-specific)      │
│    2. Thought Generator    — sample b candidates per node (typically b=5)       │
│    3. State Evaluator      — LLM scores each state: sure/maybe/impossible      │
│    4. Search Algorithm     — BFS (keeps top-b per level) or DFS (backtrack)     │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CoT vs. ToT                                                                     │
│    CoT:  b=1, no evaluation, no backtrack, O(L) cost                            │
│    ToT:  b>1, explicit eval, backtracking, O(b×d) cost                          │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  KEY RESULTS (Yao et al., 2023)                                                  │
│    Game of 24:  CoT 4%  →  ToT 74%   (+70 pp)                                  │
│    Crossword:   CoT 16% →  ToT 60%   (+44 pp)                                  │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHEN TO USE ToT                    WHEN TO USE CoT                             │
│  ✓ Search problems (planning, 24)   ✓ Sequential problems (arithmetic)          │
│  ✓ Dead ends frequent               ✓ Unique clear path exists                  │
│  ✓ Lookahead is informative         ✓ Low latency required                      │
│  ✓ Backtracking needed              ✓ Evaluator unreliable                      │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  EXTENSIONS                                                                      │
│  Graph-of-Thoughts (Besta, AAAI 2024): merge branches, aggregation ops         │
│  RAP (Hao, 2023): MCTS replaces BFS/DFS, LLM as policy + value network        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 13 · Five-minute refresher

**What problem does it solve?** CoT follows a single left-to-right path. For search problems (Game of 24, planning, puzzles), the first path is almost always wrong, and there is no way to backtrack. ToT adds backtracking and lookahead.

**How does it work?** Four modules around the same LLM:
1. Decompose the problem into thought units
2. Generate b candidate thoughts at each node
3. Evaluate each partial state: sure / maybe / impossible
4. BFS (keep best-b per level) or DFS (explore and backtrack)

**Key numbers:** Game of 24: CoT 4% → ToT 74%. Crossword (word-level): CoT 16% → ToT 60%.

**Cost:** O(b×d) thought generations and evaluations. For b=5, d=3: ~15 generation + 15 evaluation calls per problem (vs. 1 for CoT).

**Critical design choice:** What counts as one "thought"? Too fine → tree too deep. Too coarse → back to CoT. Must be matched to problem structure.

**When NOT to use ToT:** Sequential problems where a direct path usually works (arithmetic, commonsense QA). In those cases, self-consistency (Q5-04) at N=10–20 is simpler and nearly as effective.

---

## 14 · Further reading

- **Yao et al. (2023), "Tree of Thoughts: Deliberate Problem Solving with Large Language Models," NeurIPS 2023:** The founding paper. Section 3 defines the four modules; Section 4 has the Game of 24 and creative writing experiments; Section 5 analyses the evaluator's contribution.
- **Wei et al. (2022), "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," NeurIPS 2022:** The CoT foundation that ToT extends. Essential context for understanding what ToT adds.
- **Long (2023), "Large Language Model Guided Tree-of-Thought," arXiv:2305.08291:** Introduces a separate controller LLM that manages the tree search, separating generation from meta-reasoning.
- **Besta et al. (2024), "Graph of Thoughts: Solving Elaborate Problems with Large Language Models," AAAI 2024, arXiv:2308.09687:** Generalises ToT to a directed graph, enabling thought merging, aggregation, and refinement operations.
- **Hao et al. (2023), "Reasoning with Language Model is Planning with World Model," arXiv:2305.14992:** Frames ToT as MCTS with the LLM as policy + world model (RAP). Strongest results on structured planning tasks.

---

## 15 · References

1. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., Narasimhan, K. — **Tree of Thoughts: Deliberate Problem Solving with Large Language Models.** *NeurIPS 2023 / arXiv:2305.10601.* — Introduces the four-module ToT framework; Game of 24: CoT 4% → ToT 74%.

2. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E., Le, Q., Zhou, D. — **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.** *NeurIPS 2022 / arXiv:2201.11903.* — The CoT foundation. ToT is directly positioned as its generalisation.

3. Long, J. — **Large Language Model Guided Tree-of-Thought.** *arXiv:2305.08291, 2023.* — Introduces a separate controller LLM for tree search, extending Yao et al.'s self-evaluation approach.

4. Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Podstawski, M., Gianinazzi, L., Gajda, J., Lehmann, T., Niewiadomski, H., Nyczyk, P., Hoefler, T. — **Graph of Thoughts: Solving Elaborate Problems with Large Language Models.** *AAAI 2024 / arXiv:2308.09687.* — Generalises ToT to directed graphs; enables thought merging and aggregation.

5. Hao, S., Gu, Y., Ma, H., Hong, J. J., Wang, Z., Wang, D. Z., Hu, Z. — **Reasoning with Language Model is Planning with World Model.** *arXiv:2305.14992, 2023.* — RAP: replaces BFS/DFS with MCTS; uses LLM as policy and world model for planning tasks.

6. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D. — **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** *ICLR 2023 / arXiv:2203.11171.* — Self-consistency: the flat-sampling alternative to ToT; essential for understanding the tradeoffs.

---

<!-- ============================ BOTTOM NAV ============================ -->
<div align="center">

[⬅️ Q5‑04 — Self-Consistency](./q04-self-consistency.md) &nbsp;|&nbsp; [📚 Back to Section 5](./README.md) &nbsp;|&nbsp; [🏠 Home](../../README.md) &nbsp;|&nbsp; [Q5‑06 ➡️](./q06-system1-system2.md)

<sub>Found an error or have a sharper intuition? See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a> — answers follow the <a href="../../_TEMPLATE.md">answer template</a>.</sub>

</div>
