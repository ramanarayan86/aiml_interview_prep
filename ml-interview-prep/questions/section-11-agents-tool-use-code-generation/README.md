<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 10 — Mixture of Experts](../section-10-mixture-of-experts/README.md) &nbsp;|&nbsp; [Section 12 — Evaluation, Safety & Interpretability ➡️](../section-12-evaluation-safety-interpretability/README.md)

</div>

---

# Section 11 · Agents, Tool Use, and Code Generation

<div align="center">

![Questions](https://img.shields.io/badge/questions-34-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F34-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-34%2F34-6c4fe0)

</div>

Agents represent the frontier of LLM capability deployment. This section covers **function calling and tool use**, **ReAct and agentic loops**, **SWE-bench and coding agents**, **multi-agent systems**, **code execution sandboxing**, and the deep tension between scaffolding-based vs RL-trained agent paradigms.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q11‑01 | What is a 'tool' in the LLM-agent sense? How do function-calling fine-tunes work? | tool · function-calling · fine-tuning · schema | 📝 |
| Q11‑02 | Walk through the ReAct paradigm. What's the difference between Reason+Act and pure CoT? | ReAct · reason-act · CoT · agentic | 📝 |
| Q11‑03 | What is the difference between an agent and a chatbot, mechanically? | agent · chatbot · tool-use · autonomy | 📝 |
| Q11‑04 | Discuss the typical agent loop (observe, plan, act, reflect). Where does it most often fail? | agent-loop · observe-plan-act · failure-modes | 📝 |
| Q11‑05 | Why do code-generation models benefit from fill-in-the-middle (FIM) training? | FIM · code-generation · infilling · training | 📝 |
| Q11‑06 | What is the difference between an agentic workflow and a fully autonomous agent? | agentic-workflow · autonomous · orchestration · human-in-loop | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q11‑07 | Compare function calling (OpenAI/Anthropic structured), MCP (Model Context Protocol), and free-form tool descriptions. What are the engineering tradeoffs? | function-calling · MCP · tool-descriptions · engineering | 📝 |
| Q11‑08 | Discuss tool selection at scale (100+ tools). What retrieval and prompting strategies prevent tool confusion? | tool-selection · 100-tools · retrieval · confusion | 📝 |
| Q11‑09 | Walk through SWE-bench. Why was it such a difficult benchmark, and how have agents like SWE-agent, AutoCodeRover, and Cognition's Devin made progress? | SWE-bench · coding-agent · SWE-agent · Devin | 📝 |
| Q11‑10 | What is the difference between training-time tool use (ToolLLM, Gorilla) and prompting-time tool use? When does fine-tuning pay off? | training-time · prompting-time · ToolLLM · Gorilla | 📝 |
| Q11‑11 | Discuss multi-agent systems. When does decomposing into multiple agents help (debate, verifier-generator) vs hurt (compounding errors)? | multi-agent · debate · verifier-generator · compounding-errors | 📝 |
| Q11‑12 | Compare agent frameworks (LangChain, AutoGen, CrewAI, smol-agents). What are the abstractions, and where do they leak? | LangChain · AutoGen · CrewAI · smol-agents · abstraction | 📝 |
| Q11‑13 | What is a planner-executor architecture? How does it compare to ReAct-style interleaved planning? | planner-executor · ReAct · planning · architecture | 📝 |
| Q11‑14 | Discuss code-generation benchmarks beyond HumanEval — BigCodeBench, LiveCodeBench, RepoBench. What does each measure that HumanEval misses? | BigCodeBench · LiveCodeBench · RepoBench · HumanEval | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q11‑15 | Design a coding agent that achieves >50% on SWE-bench Verified. Walk through the architecture: retrieval, planning, code execution, test feedback, and revision. | SWE-bench · coding-agent · architecture · test-feedback | 📝 |
| Q11‑16 | Discuss 'agentic reasoning' — how do you train a model end-to-end on multi-turn tool use? Reference Toolformer, LATS, and the recent trend of RL on tool-use trajectories. | agentic-reasoning · Toolformer · LATS · RL-tool-use | 📝 |
| Q11‑17 | Critique the current state of agents. Why do they fail at long-horizon tasks? Discuss compounding error, the lack of stable world models, and credit assignment over long trajectories. | long-horizon · compounding-error · world-models · credit-assignment | 📝 |
| Q11‑18 | Walk through how you'd build a sandboxed code-execution environment for an agent at scale. Security, statefulness, resource isolation, and reproducibility. | sandbox · code-execution · security · isolation | 📝 |
| Q11‑19 | Discuss the 'bitter lesson' applied to agents — is scaffolding (LangChain-style) or end-to-end RL (the o1/R1 paradigm extended to tools) the right paradigm? | bitter-lesson · scaffolding · end-to-end-RL · paradigm | 📝 |
| Q11‑20 | Walk through how you would do RL on tool-use trajectories. What's the action space, the reward, and the credit assignment story? | RL · tool-use · action-space · credit-assignment | 📝 |
| Q11‑21 | Discuss browser-use agents (WebArena, VisualWebArena). What's the gap between LLM-only and VLM-based browser agents? | browser-use · WebArena · VLM-agent · LLM-agent | 📝 |
| Q11‑22 | Critically evaluate Devin, Aider, and Cursor as different points in the coding-agent design space. What does each get right and wrong? | Devin · Aider · Cursor · coding-agent · comparison | 📝 |
| Q11‑23 | Discuss the role of memory in agents. Compare scratchpad, vector memory, and structured memory (e.g., MemGPT). When does each pay off? | agent-memory · scratchpad · vector-memory · MemGPT | 📝 |
| Q11‑24 | Walk through how you would design an evaluation harness for a software-engineering agent that handles repository-scale tasks. What's the right cost model? | evaluation-harness · repo-scale · cost-model · SWE | 📝 |
| Q11‑25 | Discuss the alignment risks specific to agents. How do you bound exploration in a tool-using agent without crippling capability? | alignment-risks · exploration · tool-use · safety | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q11‑26 | Your coding agent achieves 35% on SWE-bench Verified but only 18% on your internal repo. Walk through what you'd investigate. | SWE-bench · internal-repo · gap · investigation | 📝 |
| Q11‑27 | Walk through how you'd build evaluation infrastructure for a coding agent at the scale of 1000+ tasks per day. | evaluation · scale · 1000-tasks · infrastructure | 📝 |
| Q11‑28 | Your agent fails at long-horizon tasks beyond 20 tool calls. Walk through your investigation and architectural options. | long-horizon · 20-tool-calls · compounding-error · architecture | 📝 |
| Q11‑29 | Walk through how you'd debug an agent that occasionally takes dangerous actions (e.g., deleting files, sending emails) despite passing safety evaluations. | dangerous-actions · safety · debugging · evaluation-gap | 📝 |
| Q11‑30 | You're asked to add 50 new tools to an existing agent. Walk through your retrieval, prompting, and evaluation strategy. | tool-expansion · retrieval · prompting · evaluation | 📝 |
| Q11‑31 | Walk through how you'd design a sandboxed code-execution environment for an agent serving 10K users daily. | sandbox · 10K-users · isolation · production | 📝 |
| Q11‑32 | Your agent has a 12% rate of hallucinating tool names that don't exist. Walk through your investigation and fix. | tool-hallucination · 12-percent · debugging · fix | 📝 |
| Q11‑33 | Walk through your approach to teaching an agent to recover from its own errors mid-trajectory. | error-recovery · mid-trajectory · self-correction · agent | 📝 |
| Q11‑34 | You need to ship a browser-use agent. Walk through your architecture decisions — VLM-based, DOM-based, or hybrid — and the evaluation strategy. | browser-use · VLM · DOM · architecture | 📝 |

---

## Why these questions matter

Agent systems are the fastest-growing deployment pattern for LLMs. These questions test whether a candidate understands the **full agent stack** — from tool calling mechanics to RL training to production sandboxing to evaluation.

| Theme | Key questions |
|-------|--------------|
| **Tool use fundamentals** | Q11‑01, Q11‑07, Q11‑10, Q11‑20 — calling, MCP, RL training |
| **Coding agents** | Q11‑05, Q11‑09, Q11‑15, Q11‑22 — FIM, SWE-bench, Devin |
| **Architecture** | Q11‑02, Q11‑11, Q11‑13, Q11‑19 — ReAct, multi-agent, scaffolding vs RL |
| **Production** | Q11‑18, Q11‑24, Q11‑29, Q11‑31 — sandbox, eval, dangerous actions |

---

## Reading order suggestions

**New to agents?** Q11‑01 → Q11‑02 → Q11‑03 → Q11‑04 → Q11‑06

**Interview in 48 hours?** Q11‑09 (SWE-bench) · Q11‑15 (coding agent design) · Q11‑17 (failure modes) · Q11‑19 (scaffolding vs RL) · Q11‑11 (multi-agent)

**Full track:** Q11‑01 → Q11‑06 → Q11‑07 → Q11‑14 → Q11‑15 → Q11‑17 → Q11‑19

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 10 — Mixture of Experts](../section-10-mixture-of-experts/README.md) &nbsp;|&nbsp; [Section 12 — Evaluation, Safety & Interpretability ➡️](../section-12-evaluation-safety-interpretability/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s11qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
