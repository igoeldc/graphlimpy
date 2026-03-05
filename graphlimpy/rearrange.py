from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np

from .utils import rng, normalize_splits

Graphon = Callable[[np.ndarray, np.ndarray], np.ndarray]
Phi = Callable[[np.ndarray], np.ndarray]


def rearrange_graphon(W: Graphon, phi: Phi) -> Graphon:
    """
    Rearranged graphon: W^phi(x,y) = W(phi(x), phi(y)).

    If phi is measure-preserving, W and W^phi are weakly isomorphic representatives.
    """
    def Wphi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return W(phi(x), phi(y))
    return Wphi


def shift(alpha: float) -> Phi:
    """
    Rotation of [0,1): phi(x) = (x + alpha) mod 1.
    Measure-preserving bijection.
    """
    a = float(alpha) % 1.0

    def phi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x + a) % 1.0

    return phi


def interval_reorder(
    order: Sequence[int],
    *,
    splits: Optional[Sequence[float]] = None,
) -> Phi:
    """
    Measure-preserving bijection that reorders intervals of [0,1].

    IMPORTANT: `order` specifies the NEW left-to-right ORDER of the ORIGINAL intervals.

    - If splits is None: use k equal intervals.
    - If splits is provided: interval lengths are given by splits (normalized to sum to 1),
      and reordering preserves lengths exactly (pure translations, no scaling).

    Example (k=4):
        order = [2,0,3,1]
    means:
        new tiling is [I2 | I0 | I3 | I1]
    where Ii is the i-th interval in the original tiling.

    This is genuinely measure-preserving for any splits (because it only translates pieces).
    """
    order = np.asarray(order, dtype=int).ravel()
    if order.ndim != 1 or order.size == 0:
        raise ValueError("order must be a non-empty 1D permutation")
    k = int(order.size)
    if set(order.tolist()) != set(range(k)):
        raise ValueError("order must be a permutation of 0..k-1")

    if splits is None:
        lengths = np.ones(k, dtype=float) / k
    else:
        lengths = normalize_splits(splits)
        if lengths.size != k:
            raise ValueError("splits must have length k matching order")

    # Original partition edges
    edges = np.concatenate([[0.0], np.cumsum(lengths)])  # length k+1, ends at 1

    # New partition edges after reordering (same lengths, permuted in order)
    lengths_new = lengths[order]
    edges_new = np.concatenate([[0.0], np.cumsum(lengths_new)])

    # Inverse map: for original interval index i, find its position r in the new order
    inv = np.empty(k, dtype=int)
    inv[order] = np.arange(k)

    def phi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # keep endpoints sane
        x_clipped = np.clip(x, 0.0, 1.0)

        # find original interval index i
        i = np.searchsorted(edges, x_clipped, side="right") - 1
        i = np.clip(i, 0, k - 1)

        # local coordinate within original interval
        offset = x_clipped - edges[i]

        # new position r for that original interval
        rpos = inv[i]

        # translate into its new location (no scaling)
        out = edges_new[rpos] + offset

        # ensure x==1 maps to 1 exactly
        out = np.where(x_clipped >= 1.0, 1.0, out)
        return out

    return phi


def random_interval_reorder(
    k: int,
    *,
    seed: Optional[Union[int, np.random.Generator]] = None,
    splits: Optional[Sequence[float]] = None,
) -> Phi:
    """Random interval reordering (measure-preserving)."""
    if k <= 0:
        raise ValueError("k must be positive")
    g = rng(seed)
    order = g.permutation(k)
    return interval_reorder(order, splits=splits)


def swap_intervals(a: int, b: int, *, k: int = 2) -> Phi:
    """
    Swap two intervals among k equal intervals via a reorder permutation.
    For k=2, swaps [0,1/2) and [1/2,1].
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not (0 <= a < k and 0 <= b < k):
        raise ValueError("a,b must be in {0,...,k-1}")
    order = np.arange(k)
    order[a], order[b] = order[b], order[a]
    return interval_reorder(order)