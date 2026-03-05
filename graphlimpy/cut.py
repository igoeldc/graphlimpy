from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from .utils import rng


Graphon = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CutNormResult:
    """
    Approximate cut norm result for a matrix A.

    We return:
      - value: approximate ||A||_□ (unnormalized, i.e. sum over entries in the cut)
      - S: boolean mask for rows (subset S ⊆ [n])
      - T: boolean mask for cols (subset T ⊆ [m])
    """
    value: float
    S: np.ndarray
    T: np.ndarray


def _to_pm1(mask: np.ndarray) -> np.ndarray:
    """Convert boolean mask to ±1 vector (True -> +1, False -> -1)."""
    return np.where(mask, 1.0, -1.0)


def _to_mask_pm1(v: np.ndarray) -> np.ndarray:
    """Convert real vector to boolean mask by sign (>=0 -> True)."""
    return (v >= 0)


def cut_value(A: np.ndarray, S: np.ndarray, T: np.ndarray) -> float:
    """
    Compute |sum_{i in S, j in T} A_ij| for boolean masks S,T.
    """
    A = np.asarray(A, dtype=float)
    S = np.asarray(S, dtype=bool)
    T = np.asarray(T, dtype=bool)
    return float(abs(A[np.ix_(S, T)].sum()))


def cut_norm(
    A: np.ndarray,
    *,
    trials: int = 64,
    iters: int = 20,
    seed: Optional[Union[int, np.random.Generator]] = None,
    return_sets: bool = False,
) -> Union[float, CutNormResult]:
    """
    Approximate cut norm ||A||_□ = max_{S,T} |sum_{i in S, j in T} A_ij|.

    Heuristic:
      - alternating maximization over sign vectors s,t in {±1}
      - convert to candidate sets S = {i: s_i=+1}, T = {j: t_j=+1}
      - report the *actual* cut value |sum_{S×T} A_ij| for those sets

    Returns:
      - float if return_sets=False
      - CutNormResult if return_sets=True

    NOTE: For adjacency matrices (dense graph setting), you often normalize by n^2
    outside this function.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    n, m = A.shape
    if n == 0 or m == 0:
        if return_sets:
            return CutNormResult(0.0, np.zeros(n, dtype=bool), np.zeros(m, dtype=bool))
        return 0.0

    rngen = rng(seed)

    # Quick exit for (near-)zero matrices
    if np.allclose(A, 0.0):
        if return_sets:
            return CutNormResult(0.0, np.zeros(n, dtype=bool), np.zeros(m, dtype=bool))
        return 0.0

    best_val = -np.inf
    best_S = None
    best_T = None

    for _ in range(trials):
        t = rngen.choice(np.array([-1.0, 1.0]), size=m)

        for _k in range(iters):
            s = np.sign(A @ t)
            s[s == 0] = 1.0
            t = np.sign(A.T @ s)
            t[t == 0] = 1.0

        S = (s >= 0)
        T = (t >= 0)

        # Evaluate the *actual* cut value for these sets
        val = cut_value(A, S, T)
        if val > best_val:
            best_val = val
            best_S = S.copy()
            best_T = T.copy()

    if not return_sets:
        return float(best_val)

    return CutNormResult(float(best_val), S=best_S, T=best_T)


def cut_distance_graphs(
    A: np.ndarray,
    B: np.ndarray,
    *,
    trials: int = 64,
    iters: int = 20,
    seed: Optional[Union[int, np.random.Generator]] = None,
    normalize: bool = True,
) -> float:
    """
    Approximate cut distance between two graphs represented by adjacency matrices.

    We compute ||A - B||_□ (approx), optionally normalized by n^2.

    Args:
        A, B: (n,n) adjacency matrices (assumed same shape).
        trials, iters, seed: passed to cut_norm
        normalize: if True, divide by n^2.

    Returns:
        Approximate cut distance.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square adjacency matrices")

    D = A.astype(float) - B.astype(float)
    val = cut_norm(D, trials=trials, iters=iters, seed=seed, return_sets=False)
    if normalize:
        n = A.shape[0]
        val = float(val) / (n * n)
    return float(val)


def cut_distance_graphons(
    W1: Graphon,
    W2: Graphon,
    *,
    n: int = 250,
    trials: int = 64,
    iters: int = 20,
    seed: Optional[Union[int, np.random.Generator]] = None,
    normalize: bool = False,
) -> float:
    """
    Approximate ||W1 - W2||_□ by discretizing on n sampled points.

    Procedure:
      1) sample u_1,...,u_n ~ Unif[0,1]
      2) form matrices M1_ij = W1(u_i,u_j), M2_ij = W2(u_i,u_j)
      3) compute approx cut norm of M1 - M2

    Args:
        W1, W2: graphon callables.
        n: discretization size (bigger is better but slower).
        trials, iters, seed: passed to cut_norm.
        normalize: if True, divide by n^2 (often convenient).

    Returns:
        Approximate cut distance between W1 and W2 (without rearrangements).
        NOTE: This is not delta_□ (which also infimizes over measure-preserving
        maps). This is the raw cut norm distance of the two representatives.
    """
    rngen = rng(seed)
    u = rngen.random(n)
    U = u[:, None]

    M1 = np.asarray(W1(U, U.T), dtype=float)
    M2 = np.asarray(W2(U, U.T), dtype=float)
    D = M1 - M2

    val = cut_norm(D, trials=trials, iters=iters, seed=rngen, return_sets=False)
    if normalize:
        val = float(val) / (n * n)
    return float(val)


def cut_reorder(
    A: np.ndarray,
    S: np.ndarray,
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Reorder rows/cols so that S rows come first and T cols come first.

    Returns:
        A_re: reordered matrix
        row_perm: permutation array for rows
        col_perm: permutation array for cols
        s_size: |S|
        t_size: |T|
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    n, m = A.shape

    S = np.asarray(S, dtype=bool)
    T = np.asarray(T, dtype=bool)
    if S.shape != (n,):
        raise ValueError(f"S must have shape ({n},)")
    if T.shape != (m,):
        raise ValueError(f"T must have shape ({m},)")

    row_perm = np.concatenate([np.flatnonzero(S), np.flatnonzero(~S)])
    col_perm = np.concatenate([np.flatnonzero(T), np.flatnonzero(~T)])

    A_re = A[np.ix_(row_perm, col_perm)]
    return A_re, row_perm, col_perm, int(S.sum()), int(T.sum())


def cut_best_reordered(
    A: np.ndarray,
    *,
    trials: int = 64,
    iters: int = 20,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> tuple[CutNormResult, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Convenience wrapper:
      - run cut_norm(A, return_sets=True)
      - return result plus the reordered matrix and split locations.

    Returns:
        (res, A_re, row_perm, col_perm, s_size, t_size)
    """
    res = cut_norm(A, trials=trials, iters=iters, seed=seed, return_sets=True)
    A_re, row_perm, col_perm, s_size, t_size = cut_reorder(A, res.S, res.T)
    return res, A_re, row_perm, col_perm, s_size, t_size