from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np


Array = np.ndarray
Graphon = Callable[[Array, Array], Array]


def _asarray(x) -> Array:
    return np.asarray(x)


def _clip01(z: Array) -> Array:
    z = _asarray(z).astype(float, copy=False)
    return np.clip(z, 0.0, 1.0)


def constant(p: float) -> Graphon:
    """
    Constant graphon: W(x,y)=p.
    """
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")

    def W(x: Array, y: Array) -> Array:
        x, y = np.broadcast_arrays(_asarray(x), _asarray(y))
        return p + 0.0 * x + 0.0 * y

    return W


def half_graphon() -> Graphon:
    """
    Half-graphon: W(x,y) = 1[x+y >= 1].
    Symmetric, {0,1}-valued.
    """
    def W(x: Array, y: Array) -> Array:
        x, y = np.broadcast_arrays(_asarray(x), _asarray(y))
        return (x + y >= 1.0).astype(float)

    return W


def ramp(a: float = 0.05, b: float = 0.9) -> Graphon:
    """
    Smooth-ish example: W(x,y) = clip(a + b*x*y, 0, 1).
    """
    a = float(a)
    b = float(b)

    def W(x: Array, y: Array) -> Array:
        x, y = np.broadcast_arrays(_asarray(x), _asarray(y))
        return _clip01(a + b * (x * y))

    return W


def bipartite(split: float = 0.5, p_in: float = 0.05, p_out: float = 0.4) -> Graphon:
    """
    2-block step graphon:
      - same side of split -> p_in
      - different sides     -> p_out
    """
    split = float(split)
    if not (0.0 < split < 1.0):
        raise ValueError("split must be in (0,1)")
    p_in = float(p_in)
    p_out = float(p_out)
    if not (0.0 <= p_in <= 1.0 and 0.0 <= p_out <= 1.0):
        raise ValueError("p_in and p_out must be in [0,1]")

    def W(x: Array, y: Array) -> Array:
        x, y = np.broadcast_arrays(_asarray(x), _asarray(y))
        A = x < split
        B = y < split
        same = (A & B) | (~A & ~B)
        return np.where(same, p_in, p_out).astype(float)

    return W


def sbm(P: Union[Array, Sequence[Sequence[float]]],
        splits: Union[Array, Sequence[float]]) -> Graphon:
    """
    General k-block (stochastic block model) step graphon.

    Args:
        P: (k,k) matrix of probabilities in [0,1]
        splits: length-k positive weights summing to 1 (block sizes)

    Returns:
        W(x,y) that maps x,y in [0,1] to corresponding block probability.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square (k,k) matrix")
    k = P.shape[0]
    if np.any(P < 0.0) or np.any(P > 1.0):
        raise ValueError("P entries must be in [0,1]")

    splits = np.asarray(splits, dtype=float).ravel()
    if splits.shape[0] != k:
        raise ValueError("splits must have length k matching P")
    if np.any(splits <= 0):
        raise ValueError("splits must be positive")
    splits = splits / splits.sum()

    edges = np.concatenate([[0.0], np.cumsum(splits)])  # length k+1, ends at 1

    def block_id(u: Array) -> Array:
        u = _asarray(u).astype(float, copy=False)
        # Map u=1 to last block; otherwise find interval in edges.
        idx = np.searchsorted(edges, u, side="right") - 1
        return np.clip(idx, 0, k - 1)

    def W(x: Array, y: Array) -> Array:
        x, y = np.broadcast_arrays(_asarray(x), _asarray(y))
        bx = block_id(x)
        by = block_id(y)
        return P[bx, by]

    return W


def rank1(f: Callable[[Array], Array], *, clip: bool = True) -> Graphon:
    """
    Rank-1 symmetric graphon: W(x,y) = f(x) f(y).
    Typically you'd choose f:[0,1]->[0,1] so product is in [0,1].

    Args:
        f: callable on arrays
        clip: whether to clip output to [0,1]
    """
    def W(x: Array, y: Array) -> Array:
        x, y = np.broadcast_arrays(_asarray(x), _asarray(y))
        z = _asarray(f(x)) * _asarray(f(y))
        return _clip01(z) if clip else z.astype(float, copy=False)

    return W


def step_from_matrix(B: Union[Array, Sequence[Sequence[float]]],
                     splits: Optional[Union[Array, Sequence[float]]] = None) -> Graphon:
    """
    Step graphon defined by a k×k matrix B and (optionally) non-uniform block sizes.

    Args:
        B: (k,k) probabilities in [0,1]
        splits: optional length-k positive weights summing to 1.
                If None, uses equal-sized blocks.

    Returns:
        Callable step-graphon W(x,y).
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("B must be a square (k,k) matrix")
    if np.any(B < 0.0) or np.any(B > 1.0):
        raise ValueError("B entries must be in [0,1]")
    k = B.shape[0]

    if splits is None:
        splits = np.ones(k, dtype=float) / k
    else:
        splits = np.asarray(splits, dtype=float).ravel()
        if splits.shape[0] != k:
            raise ValueError("splits must have length k matching B")
        if np.any(splits <= 0):
            raise ValueError("splits must be positive")
        splits = splits / splits.sum()

    edges = np.concatenate([[0.0], np.cumsum(splits)])

    def block_id(u: Array) -> Array:
        u = _asarray(u).astype(float, copy=False)
        idx = np.searchsorted(edges, u, side="right") - 1
        return np.clip(idx, 0, k - 1)

    def W(x: Array, y: Array) -> Array:
        x, y = np.broadcast_arrays(_asarray(x), _asarray(y))
        bx = block_id(x)
        by = block_id(y)
        return B[bx, by]

    return W