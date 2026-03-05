from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


def rng(seed: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    """Return a numpy Generator from an int seed or an existing Generator."""
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def grid_points(m: int) -> np.ndarray:
    """Midpoint grid in [0,1] of length m."""
    if m <= 0:
        raise ValueError("m must be positive")
    return np.linspace(0, 1, m, endpoint=False) + 0.5 / m


def permutation(n: int, seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
    """Random permutation of {0,...,n-1}."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    g = rng(seed)
    return g.permutation(n)


def permute_matrix(A: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Conjugate-permute a square matrix: A[p,p].
    Useful for relabeling adjacency matrices.
    """
    A = np.asarray(A)
    p = np.asarray(p)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if p.ndim != 1 or p.shape[0] != A.shape[0]:
        raise ValueError("p must be a 1D permutation of length n")
    return A[np.ix_(p, p)]


def clip01(x: np.ndarray) -> np.ndarray:
    """Clip array to [0,1]."""
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def normalize_splits(splits: Sequence[float]) -> np.ndarray:
    """Normalize positive splits to sum to 1."""
    s = np.asarray(splits, dtype=float).ravel()
    if s.ndim != 1 or s.size == 0:
        raise ValueError("splits must be a non-empty 1D sequence")
    if np.any(s <= 0):
        raise ValueError("splits must be positive")
    return s / s.sum()