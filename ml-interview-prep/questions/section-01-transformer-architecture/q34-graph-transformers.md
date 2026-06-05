<!-- ============================ TOP NAV ============================ -->
<div align="center">

[🏠 Home](../../README.md) &nbsp;•&nbsp; [📚 Section 1 — Transformer Architecture](./README.md) &nbsp;•&nbsp; [⬅️ Q33 — Attention as Kernel Method](./q33-attention-kernel-method.md)

</div>

---

# Q34 · Transformers on graphs — Graphormer, GraphGPS, and why vanilla Transformers struggle on graph data

<div align="center">

![Section](https://img.shields.io/badge/Section_1-Transformer_Architecture-1f4fa3)
![Level](https://img.shields.io/badge/Level-Advanced-b5321a)
![Tags](https://img.shields.io/badge/tags-Graphormer·GraphGPS·graph--transformer·positional--encoding·MPNN·molecular--property-0e8a6e)
![Read](https://img.shields.io/badge/read_time-~18_min-5b6675)

</div>

> [!IMPORTANT]
> **The 20-second answer.** Vanilla Transformers treat input as a sequence with fixed positional order; graphs are unordered, irregular structures where the inductive bias of "distance" is topological (shortest path), not positional. A Transformer applied naively to graph nodes sees all nodes as equally distant and loses permutation equivariance unless graph structure is explicitly encoded. Two papers define the modern approach: **Graphormer** (Ying et al., NeurIPS 2021) injects graph structure into standard Transformer attention via three learnable encodings — degree centrality, spatial (shortest-path distance), and edge encodings as attention biases. **GraphGPS** (Rampášek et al., NeurIPS 2022) generalizes this into a modular framework combining local message-passing networks (MPNN) + positional/structural encodings + global Transformer attention, achieving linear complexity $O(N+E)$ and state-of-the-art results across 16 graph benchmarks.

---

## Table of contents

1. [Why vanilla Transformers struggle on graphs](#1--why-vanilla-transformers-struggle-on-graphs)
2. [The key design challenge — encoding graph structure](#2--the-key-design-challenge--encoding-graph-structure)
3. [Graphormer — structural bias in Transformer attention](#3--graphormer--structural-bias-in-transformer-attention)
4. [GraphGPS — the modular GPS framework](#4--graphgps--the-modular-gps-framework)
5. [Comparison table](#5--comparison-table)
6. [Key results](#6--key-results)
7. [Intuition and design choices](#7--intuition-and-design-choices)
8. [Reference implementations (sketches)](#8--reference-implementations-sketches)
9. [Interview drill](#9--interview-drill)
10. [Common misconceptions](#10--common-misconceptions)
11. [One-screen summary](#11--one-screen-summary)
12. [References](#12--references)

---

## 1 · Why vanilla Transformers struggle on graphs

### Problem 1: No intrinsic spatial structure

A standard Transformer uses positional encodings (sinusoidal, RoPE, etc.) that encode the **index** of a token in a sequence. Graphs have no natural linear ordering — nodes are indexed arbitrarily, and any permutation of node indices describes the same graph. Applying sequence positional encodings to graph nodes gives encodings that depend on the arbitrary index assignment, breaking **permutation equivariance**.

> [!NOTE]
> **Permutation equivariance** for graph models means: if you permute the node labels (renumber the nodes), the output for each node should be permuted correspondingly. A model that embeds node $i$ differently from node $j$ purely because $i < j$ (sequence positional order) is not permutation equivariant.

### Problem 2: All-to-all attention ignores graph topology

In a vanilla Transformer, every node attends to every other node with equal initial access — the model starts from a fully-connected view. For graphs, many node pairs are structurally far apart (high shortest-path distance or no path at all), and their interaction should be downweighted. Without structural priors, the model must learn topology from scratch, which is data-inefficient.

### Problem 3: Edge features are discarded

Many graphs (molecules, knowledge graphs, citation networks) have rich edge features (bond types, relation labels, citation weights). A sequence Transformer has no natural mechanism to incorporate edge-level information.

### Problem 4: Over-squashing in deep MPNNs (alternative approach fails)

The classical alternative — message-passing GNNs (MPNNs) — aggregate information only from local neighborhoods. At depth $k$, a node sees only its $k$-hop neighbors. For long-range dependencies, this requires very deep networks that suffer from **over-squashing** (exponentially many paths compressed into a fixed-size vector) and **over-smoothing** (node features converge to indistinguishable vectors).

---

## 2 · The key design challenge — encoding graph structure

Any graph Transformer must solve: **how do you encode graph structure (topology, distances, edge features) into the attention mechanism without breaking permutation equivariance?**

The answer has three categories:

| Encoding type | What it captures | How injected |
|---|---|---|
| **Node-level (centrality)** | Node importance / degree | Added to node feature embedding |
| **Pairwise (spatial)** | Distance between node pairs | Added as attention bias $b_{ij}$ |
| **Edge-level** | Features on the edge between nodes | Added as attention bias or feature |

Permutation equivariance is preserved as long as encodings depend on **graph properties** (degree, path length) rather than **node indices**.

---

## 3 · Graphormer — structural bias in Transformer attention

**Paper:** "Do Transformers Really Perform Bad for Graph Representation?"
**Authors:** Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu
**Venue:** NeurIPS 2021
**arXiv:** 2106.05234

### Three structural encodings

#### (1) Centrality encoding — node degree

Each node $v$ in the graph has in-degree $\text{deg}^-(v)$ and out-degree $\text{deg}^+(v)$. Graphormer adds learnable degree embeddings to the input node features:

$$
h_v^{(0)} = x_v + z^-_{\text{deg}^-(v)} + z^+_{\text{deg}^+(v)}
$$

where $x_v$ is the node feature and $z^-, z^+ \in \mathbb{R}^{d}$ are learnable embedding tables indexed by degree. This informs the model about the structural importance of each node (hub nodes have high degree).

#### (2) Spatial encoding — shortest-path distance

For each pair of nodes $(v_i, v_j)$, compute the shortest-path distance $\text{SPD}(v_i, v_j)$ in the graph. Assign a learnable scalar bias $b_\phi$ to the attention logit:

$$
A_{ij} = \frac{(h_i W_Q)(h_j W_K)^\top}{\sqrt{d}} + b_{\phi(v_i, v_j)}
$$

where $\phi(v_i, v_j) = \text{SPD}(v_i, v_j)$ and $b_\phi \in \mathbb{R}$ is a learnable scalar per distance value. For disconnected nodes, a special "unreachable" token is assigned. This encodes the graph's metric structure into attention.

#### (3) Edge encoding — attention bias from edge features

For node pairs $(v_i, v_j)$ connected by a path $\pi_{ij} = (e_1, e_2, \ldots, e_K)$ of $K$ edges, the edge encoding is:

$$
c_{ij} = \frac{1}{K} \sum_{n=1}^K x_{e_n}^\top w_n
$$

where $x_{e_n}$ is the feature of the $n$-th edge on the path and $w_n \in \mathbb{R}^{d_e}$ are learnable weights. This is added as an additional attention bias:

$$
A_{ij} = \frac{(h_i W_Q)(h_j W_K)^\top}{\sqrt{d}} + b_{\phi(v_i, v_j)} + c_{ij}
$$

The combined attention formula thus incorporates node features, pair-wise topology (SPD), and edge features along shortest paths.

### Key insight

Graphormer shows that many popular GNN variants (GCN, GraphSAGE, GIN, etc.) can be expressed as **special cases of Graphormer** by choosing appropriate encodings. This makes Graphormer a strict generalization of the MPNN family.

---

## 4 · GraphGPS — the modular GPS framework

**Paper:** "Recipe for a General, Powerful, Scalable Graph Transformer"
**Authors:** Ladislav Rampášek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, Dominique Beaini
**Venue:** NeurIPS 2022
**arXiv:** 2205.12454

### Motivation

Graphormer's spatial encoding requires precomputing all-pairs shortest paths — $O(N^3)$ preprocessing and $O(N^2)$ memory, prohibiting large graphs. GraphGPS addresses this with a modular framework that achieves **linear complexity $O(N + E)$** by decoupling local and global computation.

### The GPS layer

Each GPS layer combines three components applied in sequence (or parallel):

$$
h^{(\ell+1)} = \text{GPS-Layer}(h^{(\ell)}) = \text{FFN}(\text{GlobalAttn}(h^{(\ell)}) + \text{LocalMPNN}(h^{(\ell)}))
$$

More precisely:

1. **Local Message-Passing (MPNN):** Any standard MPNN (GatedGCN, GINE, etc.) operating on **real edges only** — aggregates information from direct neighbors. Cost: $O(E \cdot d)$ where $E$ = number of edges.

2. **Global Attention:** Any Transformer-style attention operating over **all node pairs** — captures long-range dependencies. Cost: $O(N^2 d)$ for full attention, or $O(Nrd)$ for linear attention (Performers, etc.). When linear attention is used, the whole GPS layer is $O(N + E)$.

3. **FFN:** Standard position-wise feed-forward network.

The key insight is that the MPNN and global attention are **decoupled** — the MPNN handles local topology with exact edge information, and the global attention handles long-range patterns without needing to encode every pairwise distance.

### Positional and structural encodings in GraphGPS

GraphGPS categorizes encodings into three types:

| Type | Example | What it encodes |
|---|---|---|
| **Local** | Node degree, local topology | Immediate neighborhood |
| **Global** | Laplacian eigenvectors (LapPE), random-walk encodings (RWSE) | Graph-global position |
| **Relative** | Shortest-path distance (as in Graphormer) | Pairwise relationships |

The paper evaluates multiple combinations. **Laplacian PE (LapPE):** Use the eigenvectors of the normalized graph Laplacian $L = I - D^{-1/2} A D^{-1/2}$:

$$
L v_k = \lambda_k v_k, \quad \text{PE}_i = [v_1(i), v_2(i), \ldots, v_K(i)]
$$

where $v_k(i)$ is the $i$-th component of the $k$-th eigenvector. This assigns each node a $K$-dimensional encoding reflecting its position in the graph's spectral geometry.

**Random Walk PE (RWSE):** $\text{PE}_i(k) = P^k_{ii}$ — the probability of a random walk returning to node $i$ after $k$ steps. This encodes local graph structure without eigenvector sign ambiguity.

### Complexity

| Component | Cost |
|---|---|
| MPNN | $O(E \cdot d)$ |
| Global attention (full) | $O(N^2 d)$ |
| Global attention (linear/Performer) | $O(N r d)$ |
| FFN | $O(N d^2)$ |
| **Total (with linear attention)** | $O((N + E) d + N d^2)$ |

For sparse graphs ($E \ll N^2$), the total is effectively linear in $N$.

---

## 5 · Comparison table

| Dimension | Vanilla Transformer | Graphormer | GraphGPS |
|---|---|---|---|
| Graph structure input | None | SPD + degree + edge bias | MPNN local + PE global |
| Permutation equivariant | No (sequence PE) | Yes (graph-based PE) | Yes |
| Complexity | $O(N^2 d)$ | $O(N^2 d)$ + $O(N^3)$ SPD preprocessing | $O(N+E)$ with linear attn |
| Edge features | No | Yes (path-averaged) | Yes (via MPNN) |
| Max graph size | N/A | ~100s of nodes | ~10k+ nodes |
| Generalization | Sequences | Generalizes MPNNs | Strictly more expressive than MPNNs |
| Benchmarks | N/A | OGB-LSC winner (2021) | SOTA on 16 benchmarks (2022) |
| Published | Vaswani 2017 | NeurIPS 2021 | NeurIPS 2022 |

---

## 6 · Key results

### Graphormer

- **OGB Large-Scale Challenge (PCQM4M):** Graphormer achieved the best MAE among all entries at NeurIPS 2021, demonstrating that Transformers with structural encodings can outperform specialized GNNs on molecular property prediction.
- **Open Catalyst Challenge:** Graphormer won this competition on modeling catalyst-adsorbate reaction systems.
- **OGB-MolHIV, OGB-MolPCBA:** Competitive with or better than top GNN methods.

### GraphGPS

- **LRGB (Long-Range Graph Benchmarks):** GPS achieves state-of-the-art on 5 out of 5 LRGB datasets designed specifically to test long-range information flow — tasks where MPNNs struggle.
- **16 diverse benchmarks:** GPS with different MPNN backends and PE types consistently outperforms prior methods across molecular, social, citation, and code graphs.
- **Scalability:** By using Performer attention, GPS scales to graphs with tens of thousands of nodes.

---

## 7 · Intuition and design choices

**Why can't you just add sequence positional encodings?**

Sequence PE embeds node $i$ differently from node $j$ because $i \neq j$, but there is no reason a priori that node $1$ is more similar to node $2$ than node $1000$ — graph topology, not index, determines similarity. SPD-based spatial encoding instead embeds node pairs by their *graph distance*, which is permutation invariant.

**Why does Graphormer need $O(N^3)$ preprocessing?**

All-pairs shortest path (APSP) computation costs $O(N^3)$ via Floyd-Warshall or $O(N(N+E))$ via $N$ BFS calls. For molecules (N ~ 10–100 atoms), this is trivial. For large graphs (N ~ 10,000 nodes), it becomes prohibitive.

**Why does GPS add MPNN to global attention rather than replacing it?**

MPNNs explicitly propagate information along real edges, which encodes local topology exactly and efficiently. Global attention captures long-range patterns but is agnostic to whether two nodes are actually connected. The combination gets the best of both: exact local structure + long-range global context.

**Laplacian eigenvectors vs. random walk PE:**

Laplacian eigenvectors are theoretically motivated (they are the "graph Fourier basis") but have a sign ambiguity (eigenvectors are defined up to $\pm 1$), requiring special handling (e.g., sign-equivariant encoders). Random walk PE avoids sign ambiguity and is computationally simpler.

---

## 8 · Reference implementations (sketches)

### Graphormer spatial encoding
```python
def graphormer_attention_bias(spd_matrix, max_dist=20):
    """
    spd_matrix: (N, N) integer shortest-path distances
    Returns attention bias: (N, N)
    """
    # Learnable bias table indexed by SPD
    bias_table = nn.Embedding(max_dist + 2, 1)   # +2 for unreachable token
    spd_clamped = spd_matrix.clamp(0, max_dist)
    spd_clamped[spd_matrix < 0] = max_dist + 1   # unreachable
    return bias_table(spd_clamped).squeeze(-1)    # (N, N)

def graphormer_attention(Q, K, V, spd_bias, edge_bias=None):
    d = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / d**0.5  # (N, N)
    scores = scores + spd_bias
    if edge_bias is not None:
        scores = scores + edge_bias
    return F.softmax(scores, dim=-1) @ V
```

### GPS layer (pseudocode)
```python
class GPSLayer(nn.Module):
    def __init__(self, mpnn, global_attn, d_model, dropout=0.1):
        super().__init__()
        self.mpnn = mpnn          # any MPNN (GatedGCN, GINE, etc.)
        self.attn = global_attn   # any attention (full or Performer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x, edge_index, edge_attr=None):
        # Local: MPNN over real edges
        h_local = self.mpnn(x, edge_index, edge_attr)
        # Global: attention over all nodes
        h_global = self.attn(x)
        # Combine, normalize, FFN
        h = self.norm1(x + h_local + h_global)
        h = self.norm2(h + self.ffn(h))
        return h
```

---

## 9 · Interview drill

<details><summary><b>Q: Why is permutation equivariance important for graph models?</b></summary>

Graphs don't have a canonical node ordering — renaming node 1 as node 2 and vice versa describes the same graph. A model that is not permutation equivariant would give different predictions for the same molecule depending on how atoms are numbered, which is incorrect. Permutation equivariance ensures predictions depend only on the graph structure, not the arbitrary labeling.
</details>

<details><summary><b>Q: How does Graphormer show that GNNs are special cases of Graphormer?</b></summary>

Standard MPNNs (GCN, GIN) aggregate over 1-hop neighbors, which corresponds to Graphormer with spatial encoding restricted to nodes at SPD = 1 (directly connected). The attention to non-adjacent nodes is masked out (or given zero bias). More precisely, GCN's adjacency normalization is equivalent to Graphormer's attention with specific degree-based biases and no long-range connections.
</details>

<details><summary><b>Q: What is the main advantage of GraphGPS over Graphormer for large graphs?</b></summary>

Graphormer requires all-pairs SPD computation ($O(N^3)$ preprocessing) and $O(N^2)$ memory for attention — prohibitive for large graphs. GraphGPS achieves $O(N+E)$ by (1) using MPNN for local edges (cost proportional to edges, not node pairs) and (2) using linear attention (Performers) for global attention (cost linear in $N$). The GPS approach scales to graphs with tens of thousands of nodes.
</details>

<details><summary><b>Q: What tasks specifically benefit most from graph Transformers vs. pure MPNNs?</b></summary>

Tasks requiring **long-range dependencies** — where information must flow across many hops — benefit most. Examples: (1) Molecular property prediction where distant functional groups interact (long-range electronic effects); (2) Graph regression where global topology determines the label; (3) Tasks on social/citation graphs where distant nodes are semantically related. Pure MPNNs over-squash exponentially many paths into fixed-size vectors at deep layers, losing long-range information.
</details>

---

## 10 · Common misconceptions

| Misconception | Reality |
|---|---|
| "Any Transformer can be applied to graphs with just node feature inputs." | Without structural encodings, the model loses permutation equivariance and cannot distinguish graph topology — it treats all nodes as equally distant. |
| "Graphormer requires a special graph neural network backbone." | Graphormer is a standard Transformer; it is made graph-aware entirely through its encoding strategy (degree, SPD, edge biases), not through a modified architecture. |
| "GraphGPS replaces the MPNN with Transformer attention." | GPS **adds** global attention on top of MPNN — both are used together, not one replacing the other. |
| "SPD-based positional encoding is like sequence positional encoding." | SPD encodes topological distance in the graph (permutation invariant). Sequence PE encodes token index (permutation sensitive). They are fundamentally different. |
| "Graphormer won't scale to large graphs." | The original Graphormer is limited by $O(N^2)$ attention and $O(N^3)$ SPD preprocessing. Follow-up work (Graphormer-GD, etc.) addresses scaling for larger graphs. |

---

## 11 · One-screen summary

> **Problem:** Vanilla Transformers can't handle graphs — no permutation equivariance, no awareness of graph topology, no edge features, and quadratic cost is prohibitive for large graphs. **Graphormer (NeurIPS 2021):** Add degree centrality to node embeddings; add SPD-based learnable scalar bias to attention logits; add edge encoding bias — all permutation invariant. State-of-the-art on molecular benchmarks; $O(N^2)$ cost limits scale. **GraphGPS (NeurIPS 2022):** Modular GPS layer = MPNN (local) + global attention + PE; $O(N+E)$ with linear attention; SOTA on 16 benchmarks including long-range tasks. **Key design rule:** Encode graph structure via graph-property-dependent (not index-dependent) features injected into attention as biases.

---

## 12 · References

1. **Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., Liu, T.-Y.** "Do Transformers Really Perform Bad for Graph Representation?" NeurIPS 2021. arXiv:2106.05234. [https://arxiv.org/abs/2106.05234](https://arxiv.org/abs/2106.05234)

2. **Rampášek, L., Galkin, M., Dwivedi, V.P., Luu, A.T., Wolf, G., Beaini, D.** "Recipe for a General, Powerful, Scalable Graph Transformer." NeurIPS 2022. arXiv:2205.12454. [https://arxiv.org/abs/2205.12454](https://arxiv.org/abs/2205.12454) — Code: [https://github.com/rampasek/GraphGPS](https://github.com/rampasek/GraphGPS)

3. **Vaswani, A., et al.** "Attention Is All You Need." NeurIPS 2017. — Baseline Transformer.

4. **Dwivedi, V.P., et al.** "Long Range Graph Benchmark." NeurIPS 2022 Datasets Track. arXiv:2206.08164. — The LRGB benchmark suite GraphGPS was evaluated on.

5. **Xu, K., et al.** "How Powerful are Graph Neural Networks?" ICLR 2019. — GIN, the theoretical foundation for MPNN expressivity. arXiv:1810.00826.

6. **Veličković, P., et al.** "Graph Attention Networks." ICLR 2018. arXiv:1710.10903. — Key prior work on attention in GNNs.

---

<!-- ============================ BOTTOM NAV ============================ -->
<div align="center">

[⬅️ Q33 — Attention as Kernel Method](./q33-attention-kernel-method.md) &nbsp;|&nbsp; [📚 Back to Section 1](./README.md) &nbsp;|&nbsp; [🏠 Home](../../README.md)

<sub>Found an error? See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
