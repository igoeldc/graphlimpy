from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from .utils import rng


Graphon = Callable[[np.ndarray, np.ndarray], np.ndarray]


def edge_density_graph(A: np.ndarray, *, simple: bool = True) -> float:
    """
    Edge density of a graph given by adjacency matrix A.

    For a simple undirected graph with no loops:
      density = (#edges) / (n choose 2)
              = sum_{i<j} A_ij / (n(n-1)/2)

    Args:
        A: (n,n) adjacency matrix.
        simple: if True, ignore diagonal and assume undirected.

    Returns:
        Edge density in [0,1] (for 0/1 A).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (n,n)")
    n = A.shape[0]
    if n <= 1:
        return 0.0

    if simple:
        # assume symmetric; count upper triangle
        m = np.triu(A, 1).sum()
        return float(m) / (n * (n - 1) / 2.0)
    else:
        # directed / weighted variant: average of off-diagonals
        return float((A.sum() - np.trace(A)) / (n * (n - 1)))


def triangle_density_graph(
    A: np.ndarray,
    *,
    simple: bool = True,
    exact_if_n_leq: int = 1200,
    rng_seed: Optional[Union[int, np.random.Generator]] = None,
    samples: int = 200_000,
) -> float:
    """
    Triangle density t(K3, G) for a graph adjacency matrix A.

    For a simple undirected graph:
      #triangles = trace(A^3) / 6
      density = #triangles / (n choose 3)

    For large n, exact matrix multiplication may be expensive; we provide
    an optional Monte Carlo estimator.

    Args:
        A: (n,n) adjacency.
        simple: if True, treat as simple undirected, ignore diagonal.
        exact_if_n_leq: compute exact if n <= this threshold.
        rng_seed: seed for Monte Carlo if used.
        samples: number of random triples for Monte Carlo.

    Returns:
        Triangle density in [0,1] for 0/1 A.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (n,n)")
    n = A.shape[0]
    if n <= 2:
        return 0.0

    if simple:
        # enforce no diagonal for triangle counting
        A = A.copy()
        np.fill_diagonal(A, 0.0)

    if n <= exact_if_n_leq:
        # trace(A^3) gives 6 * #triangles for undirected simple graphs
        A2 = A @ A
        tri6 = float(np.trace(A2 @ A))
        triangles = tri6 / 6.0
        denom = n * (n - 1) * (n - 2) / 6.0
        return float(triangles / denom)

    # Monte Carlo: sample triples (i,j,k) uniformly without replacement
    rngen = rng(rng_seed)
    hits = 0
    for _ in range(samples):
        i, j, k = rngen.choice(n, size=3, replace=False)
        hits += (A[i, j] * A[j, k] * A[k, i] > 0.5)
    return float(hits) / float(samples)


def edge_density_graphon(
    W: Graphon,
    *,
    M: int = 200_000,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> float:
    """
    Monte Carlo estimate of t(K2, W) = ∫∫ W(x,y) dx dy.
    """
    rngen = rng(seed)
    x = rngen.random(M)
    y = rngen.random(M)
    return float(np.mean(W(x, y)))


def triangle_density_graphon(
    W: Graphon,
    *,
    M: int = 200_000,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> float:
    """
    Monte Carlo estimate of t(K3, W) = ∫∫∫ W(x,y)W(y,z)W(z,x) dxdydz.
    """
    rngen = rng(seed)
    x = rngen.random(M)
    y = rngen.random(M)
    z = rngen.random(M)
    return float(np.mean(W(x, y) * W(y, z) * W(z, x)))


def C4_density_graphon(
    W: Graphon,
    *,
    M: int = 200_000,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> float:
    """
    Monte Carlo estimate of t(C4, W) = ∫ W(a,b)W(b,c)W(c,d)W(d,a) da db dc dd.
    """
    rngen = rng(seed)
    a = rngen.random(M)
    b = rngen.random(M)
    c = rngen.random(M)
    d = rngen.random(M)
    return float(np.mean(W(a, b) * W(b, c) * W(c, d) * W(d, a)))


def C4_density_graph(
    A: np.ndarray,
    *,
    simple: bool = True,
    exact_if_n_leq: int = 600,
) -> float:
    """
    Density of 4-cycles C4 in a graph. For undirected simple graphs:
      #C4 = (sum_{i<j} (A^2_{ij} choose 2)) / 2
    and density = #C4 / (#labeled 4-tuples forming C4), commonly normalized by (n choose 4)*3.

    Here we return:
      t(C4, G) = hom(C4, G) / n^4 approximately, but for simple graphs it’s
    more common to report:
      #unlabeled C4 / (n choose 4).

    To avoid confusion, we provide a reasonable "graphon-style" normalization:
      t(C4, G) ≈ trace(A^4) / n^4  (with diagonal set to 0)
    which matches the homomorphism density definition for dense graphs.

    For early reading groups, you may skip this.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    n = A.shape[0]
    if n == 0:
        return 0.0

    if simple:
        A = A.copy()
        np.fill_diagonal(A, 0.0)

    if n <= exact_if_n_leq:
        A2 = A @ A
        A4 = A2 @ A2
        return float(np.trace(A4) / (n**4))

    # For larger n, you can increase exact_if_n_leq or implement sampling.
    A2 = A @ A
    A4 = A2 @ A2
    return float(np.trace(A4) / (n**4))