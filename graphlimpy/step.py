from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

Graphon = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _equal_cuts(n: int, k: int) -> np.ndarray:
    """Integer cut points for an equal partition of {0,...,n-1} into k bins."""
    if k <= 0:
        raise ValueError("k must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    return np.linspace(0, n, k + 1).astype(int)


def block_densities(
    A: np.ndarray,
    *,
    order: Optional[np.ndarray] = None,
    k: int = 20,
    include_diagonal: bool = False,
) -> np.ndarray:
    """
    Compute k×k block densities from an adjacency matrix A under a given ordering.

    Args:
        A: (n,n) adjacency matrix (0/1 or weighted in [0,1]).
        order: optional permutation of vertices.
        k: number of blocks (equal-sized by index).
        include_diagonal: if False, subtract diagonal contribution on diagonal blocks.

    Returns:
        B: (k,k) block density matrix in [0,1] (for simple graphs).
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square (n,n) array")
    n = A.shape[0]

    if order is None:
        A_ord = A
    else:
        order = np.asarray(order)
        if order.ndim != 1 or order.shape[0] != n:
            raise ValueError("order must be a 1D permutation of length n")
        A_ord = A[np.ix_(order, order)]

    cuts = _equal_cuts(n, k)
    B = np.zeros((k, k), dtype=float)

    for i in range(k):
        i0, i1 = cuts[i], cuts[i + 1]
        ni = i1 - i0
        if ni == 0:
            continue
        for j in range(k):
            j0, j1 = cuts[j], cuts[j + 1]
            nj = j1 - j0
            if nj == 0:
                continue

            block = A_ord[i0:i1, j0:j1].astype(float, copy=False)

            if i != j:
                B[i, j] = block.mean()
            else:
                if include_diagonal:
                    B[i, j] = block.mean()
                else:
                    # exclude diagonal entries for simple-graph density
                    # denom = ni^2 - ni (pairs with i!=j inside the block)
                    denom = ni * ni - ni
                    if denom <= 0:
                        B[i, j] = 0.0
                    else:
                        B[i, j] = (block.sum() - np.trace(block)) / denom

    # Clip for numerical safety
    return np.clip(B, 0.0, 1.0)


from .graphons import step_from_matrix


def empirical_step_graphon(
    A: np.ndarray,
    *,
    order: Optional[np.ndarray] = None,
    k: int = 20,
    return_callable: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, Graphon]]:
    """
    Convenience: graph -> block densities (and optionally a callable step graphon).
    """
    B = block_densities(A, order=order, k=k, include_diagonal=False)
    if not return_callable:
        return B
    W_hat = step_from_matrix(B)
    return B, W_hat