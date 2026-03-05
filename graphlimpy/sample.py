from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np

from .utils import rng


Graphon = Callable[[np.ndarray, np.ndarray], np.ndarray]


def sample_GnW(
    W: Graphon,
    n: int,
    *,
    seed: Optional[Union[int, np.random.Generator]] = None,
    return_latent: bool = True,
    return_probs: bool = False,
    simple: bool = True,
    dtype: np.dtype = np.uint8,
) -> Tuple[np.ndarray, ...]:
    """
    Sample an undirected graph A ~ G(n, W) from a graphon W.

    Model:
        u_i ~ Unif[0,1] i.i.d.
        For i<j: A_ij ~ Bernoulli(W(u_i, u_j)) independent given u
        A is symmetrized; diagonal is 0 if simple=True.

    Args:
        W: graphon callable, should support numpy broadcasting.
        n: number of vertices.
        seed: int or numpy Generator (for reproducibility).
        return_latent: whether to return latent u (length n).
        return_probs: whether to also return the probability matrix P.
        simple: if True, set diagonal to 0.
        dtype: dtype for adjacency (default uint8).

    Returns:
        If return_latent and return_probs: (A, u, P)
        If return_latent only: (A, u)
        If return_probs only: (A, P)
        Else: (A,)
    """
    if n <= 0:
        raise ValueError("n must be positive")

    g = rng(seed)
    u = g.random(n)
    U = u[:, None]

    P = W(U, U.T)
    P = np.asarray(P, dtype=float)

    # Safety: clip to [0,1] to avoid accidental slight numeric overshoot
    np.clip(P, 0.0, 1.0, out=P)

    # Sample edges
    R = g.random((n, n))
    A = (R < P).astype(dtype)

    # Symmetrize + remove diagonal
    A = np.triu(A, 1)
    A = A + A.T
    if simple:
        np.fill_diagonal(A, 0)

    out = (A,)
    if return_latent:
        out = out + (u,)
    if return_probs:
        out = out + (P,)
    return out