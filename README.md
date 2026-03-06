# Graphon Playground

> **Disclaimer:** This entire project was **vibecoded**. It was built in a single afternoon through high-signal dialogue with an AI assistant. It prioritizes conceptual clarity, rapid experimentation, and mathematical intuition over production-grade engineering.

A lightweight Python toolkit for **exploring graphons, graph limits, and cut distance**.

The goal is not to build a production library, but rather a **small interactive sandbox** useful for reading groups following Lovász’s *Large Networks and Graph Limits*. The focus is on visualization, sampling, and intuition-building experiments.


## Goals

This project should make it easy to:

- Define graphons $W:[0,1]^2 \to [0,1]$
- Visualize graphons
- Sample graphs $G(n,W)$
- Convert graphs into **step graphons**
- Compute **subgraph densities**
- Estimate **cut norm and cut distance**
- Experiment with **vertex relabelings and measure-preserving transformations**

The emphasis is on **clarity and interactivity**, not efficiency.


## Design Philosophy

Graph limit theory is naturally expressed in terms of **matrices and functions**, not graph objects.
For this reason the library is built around **NumPy arrays** rather than graph libraries such as NetworkX.

Core objects:

| Object | Representation |
|------|------|
| Graphon | Python callable `W(x,y)` |
| Graph | adjacency matrix `A` |
| Step graphon | block matrix `B` |
| Vertex ordering | permutation array |

This matches the mathematical notation used throughout *Large Networks and Graph Limits*, where graphs are represented by adjacency matrices $A_{ij}$ and graphons are measurable functions $W(x,y)$.

Using NumPy allows:

- vectorized operations
- efficient dense graph handling
- simple mathematical correspondence with the theory

NetworkX may still be used optionally for visualization or graph algorithms, but it is **not a dependency of the core library**.


## Repository Structure

```
graphlimpy/
│
├── graphlimpy/
│   │
│   ├── __init__.py
│   │
│   ├── graphons.py
│   │   Built-in graphon constructors and examples.
│   │
│   ├── sample.py
│   │   Sampling graphs G(n,W) from graphons.
│   │
│   ├── viz.py
│   │   Visualization utilities for graphons and adjacency matrices.
│   │
│   ├── step.py
│   │   Step-function graphon approximations and block models.
│   │
│   ├── stats.py
│   │   Subgraph densities and simple graph statistics.
│   │
│   ├── cut.py
│   │   Approximate cut norm and cut distance algorithms.
│   │
│   ├── rearrange.py
│   │   Measure-preserving graphon transformations.
│   │
│   └── utils.py
│       Small shared utilities (sampling grids, permutations, helpers).
│
├── demos/
│   │
│   ├── demo_sampling.py
│   ├── demo_stats_convergence.py
│   ├── demo_step_graphon.py
│   ├── demo_cut_reorder.py
│   └── demo_rearrange_graphon.py
│
├── notebooks/
│   │
│   ├── graphon_playground.ipynb
│   └── cut_distance_experiments.ipynb
│
└── README.md
```


## Core Concepts

### Graphon

A graphon is a symmetric measurable function

$$
W : [0,1]^2 \to [0,1]
$$

representing the limit of a dense graph sequence.

In this project, graphons are represented simply as **callable Python functions**:

```python
def W(x, y):
    return (x + y > 1).astype(float)
```

This makes it easy to experiment with arbitrary graphons without introducing additional class abstractions.


## Module Overview

### graphons.py

Graphon constructors and common examples.

Functions:

```
constant(p)
half_graphon()
bipartite(split, p_in, p_out)
sbm(P, splits)
ramp()
rank1(f)
```

Example:

```python
from graphlimpy.graphons import sbm

W = sbm(
    P=[[0.1, 0.5],
       [0.5, 0.2]],
    splits=[0.4, 0.6]
)
```


### viz.py

Visualization tools.

Functions:

```
plot_graphon(W, m=400)
plot_adj(A, order=None)
plot_step(B)
plot_sampling_4panel(W, n=300, k=20)
```

Example:

```python
plot_graphon(W)
```

Produces a heatmap of the graphon.


### sample.py

Sampling graphs from graphons.

Functions:

```
sample_GnW(W, n, seed=None)
```

Algorithm:

1. Sample latent variables  
   $u_i \sim \text{Uniform}[0,1]$

2. Compute probabilities

$$
P_{ij} = W(u_i, u_j)
$$

3. Sample

$$
A_{ij} \sim \text{Bernoulli}(P_{ij})
$$

Example:

```python
A, u = sample_GnW(W, n=300)
```


### step.py

Step graphon approximations.

Given a graph $G$, produce a block-averaged approximation.

Functions:

```
empirical_step_graphon(A, order=None, k=20)
```

Returns a $k \times k$ block density matrix.

This approximates the graphon associated with the graph and is closely related to the **weak regularity lemma**.


### stats.py

Subgraph densities.

Functions:

```
edge_density_graph(A)
triangle_density_graph(A)

edge_density_graphon(W)
triangle_density_graphon(W)
```

Graphon densities are computed using Monte Carlo integration.


### cut.py

Cut norm and cut distance.

Exact computation is NP-hard, so we implement approximate algorithms based on alternating maximization.

Functions:

```
cut_norm(A)

cut_distance_graphs(A, B)

cut_distance_graphons(W1, W2)
```

Graphon distance is estimated by sampling a grid of points.


### rearrange.py

Measure-preserving transformations.

Used to illustrate the fact that graphons are equivalent up to rearrangement.

Functions:

```
rearrange_graphon(W, phi)
swap_intervals(...)
random_rearrangement(...)
```

Example:

```python
W2 = rearrange_graphon(W, phi)
```

Then

```
cut_distance_graphons(W, W2) ≈ 0
```

even though the heatmaps look different.


## Planned Experiments

### 1. Graphon → Graph sampling

```
plot_graphon(W)

A, u = sample_GnW(W, 400)

plot_adj(A)
plot_adj(A, order=u.argsort())
```

Demonstrates how sampled graphs reflect the graphon structure.

Better yet, use the 4-panel sampling visualization to see the full pipeline:

```python
from graphlimpy.viz import plot_sampling_4panel
plot_sampling_4panel(W, n=400, k=25)
```


### 2. Relabelings

Permuting vertices changes adjacency matrix appearance but not structure.

Experiment:

```
A_perm = permute(A)

cut_distance_graphs(A, A_perm)
```

Distance $\approx 0$.


### 3. Graph sequence convergence

Fix a graphon and sample larger graphs.

```
for n in [50, 100, 200, 400]:
    A, _ = sample_GnW(W, n)
```

Observe:

- adjacency matrices stabilize  
- subgraph densities converge  
- cut distance to graphon decreases  


### 4. Step graphon approximations

```
B = empirical_step_graphon(A, k=20)
```

Increasing $k$ improves approximation.

Related to the **weak regularity lemma**.


## Dependencies

The core library intentionally has **minimal dependencies**:

```
numpy
matplotlib
```

This keeps the code lightweight and easy to read.

Optional integrations (not required):

```
networkx
ipywidgets
scipy
```

These may be useful for visualization, experimentation, or optimization.


## Future Extensions

Possible later additions:

- stochastic block model inference
- spectral graphon estimation
- cut-distance visualization
- graphon fitting algorithms
- sparse graphon models


## Example Session

```python
import matplotlib.pyplot as plt
from graphlimpy.graphons import half_graphon, constant
from graphlimpy.sample import sample_GnW
from graphlimpy.viz import plot_graphon, plot_adj
from graphlimpy.cut import cut_distance_graphons

# 1. Define and visualize a graphon
W = half_graphon()
plot_graphon(W, title="Half Graphon")

# 2. Sample a graph and visualize its adjacency matrix
A, u = sample_GnW(W, n=400)
plot_adj(A, title="Sampled Graph (Raw)")
plot_adj(A, order=u.argsort(), title="Sampled Graph (Sorted by latent u)")

# 3. Compute cut distance
# Distance to self should be 0
dist_self = cut_distance_graphons(W, W)
# Distance to a constant graphon (p=0.5) should be non-zero
dist_other = cut_distance_graphons(W, constant(0.5))

print(f"Distance to self: {dist_self:.4f}")
print(f"Distance to constant(0.5): {dist_other:.4f}")

plt.show()
```


## Purpose

This repository is meant as a **graphon playground for experimentation and learning**, particularly useful for reading groups working through Lovász’s *Large Networks and Graph Limits*.