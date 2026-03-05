from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .sample import sample_GnW
from .utils import grid_points


Graphon = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _make_ax(ax: Optional[plt.Axes], figsize: Tuple[float, float]) -> tuple[plt.Axes, bool]:
    """Return (ax, created_fig). If ax is None, create a figure+axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        # Add a small padding between figure edge and content
        if fig.get_layout_engine() is not None:
            fig.get_layout_engine().set(h_pad=0.1)
        return ax, True
    return ax, False


def _format_axes(ax: plt.Axes):
    """
    Standard axis formatting for graphon/adjacency plots.

    - x ticks at top
    """
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")


def _set_title(ax: plt.Axes, title: str, *, pad: float):
    """Place the title above the top ticks/labels. pad is in points."""
    ax.set_title(title, pad=pad)


def _finalize(created_fig: bool, ax: plt.Axes):
    """Finalization hook (layout handled by 'constrained' in _make_ax)."""
    pass


def plot_graphon(
    W: Graphon,
    m: int = 400,
    *,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    partitions: Optional[Sequence[float]] = None,
    show_diagonal: bool = False,
    title_pad: float = 12.0,
    cmap: str = "gray_r",
    aspect: str = "equal",
) -> plt.Axes:
    u = grid_points(m)
    X, Y = np.meshgrid(u, u, indexing="ij")
    Z = np.asarray(W(X, Y))

    ax, created = _make_ax(ax, figsize=(5, 4))

    im = ax.imshow(
        Z,
        origin="upper",
        extent=[0, 1, 1, 0],  # y=0 at top, y=1 at bottom
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect=aspect,
        cmap=cmap,
    )

    _format_axes(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _set_title(ax, title if title is not None else "Graphon", pad=title_pad)

    if partitions is not None:
        for p in partitions:
            p = float(p)
            if 0.0 < p < 1.0:
                ax.axvline(p, linewidth=1.0)
                ax.axhline(p, linewidth=1.0)

    if show_diagonal:
        ax.plot([0, 1], [0, 1], linewidth=1.0)

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _finalize(created, ax)
    return ax


def plot_adj(
    A: np.ndarray,
    *,
    order: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title_pad: float = 12.0,
    cmap: str = "gray_r",
    aspect: str = "equal",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Axes:
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square (n,n) array")

    if order is not None:
        order = np.asarray(order)
        if order.ndim != 1 or order.shape[0] != A.shape[0]:
            raise ValueError("order must be a 1D permutation of length n")
        A = A[np.ix_(order, order)]

    ax, created = _make_ax(ax, figsize=(5, 5))

    ax.imshow(
        A,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect=aspect,
        cmap=cmap,
    )

    _format_axes(ax)

    ax.set_xlabel("vertex")
    ax.set_ylabel("vertex")
    _set_title(ax, title if title is not None else "Adjacency", pad=title_pad)

    _finalize(created, ax)
    return ax


def plot_step(
    B: np.ndarray,
    *,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    title_pad: float = 12.0,
    cmap: str = "gray_r",
    aspect: str = "equal",
) -> plt.Axes:
    B = np.asarray(B)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("B must be a square (k,k) array")

    ax, created = _make_ax(ax, figsize=(4, 4))

    im = ax.imshow(
        B,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect=aspect,
        cmap=cmap,
    )

    _format_axes(ax)

    ax.set_xlabel("block")
    ax.set_ylabel("block")
    _set_title(ax, title if title is not None else "Step graphon", pad=title_pad)

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _finalize(created, ax)
    return ax


def order_by_latent(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u)
    if u.ndim != 1:
        raise ValueError("u must be 1D")
    return np.argsort(u)


def order_by_degree(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square (n,n) array")
    return np.argsort(A.sum(axis=1))


def plot_sampling_4panel(
    W: Graphon,
    n: int = 300,
    *,
    k: int = 20,
    m: int = 400,
    seed: Optional[int] = None,
    graphon_title: str = "Original W",
    raw_title: str = "Sampled A (raw)",
    sorted_title: str = "Sampled A (sorted)",
    step_title: str = "Empirical Step Graphon",
    show_diagonal: bool = True,
    show_colorbar: bool = True,
    cmap: str = "gray_r",
    title_pad: float = 12.0,
    aspect_graphon: str = "equal",
    aspect_adj: str = "equal",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    4-panel figure:
      1) original graphon heatmap
      2) sampled adjacency (raw ordering)
      3) sampled adjacency (sorted by latent u)
      4) empirical step graphon (k x k approximation)
    """
    from .step import block_densities

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(w_pad=0.3, h_pad=0.2)

    plot_graphon(
        W,
        m=m,
        ax=axes[0],
        title=graphon_title,
        show_diagonal=show_diagonal,
        show_colorbar=show_colorbar,
        cmap=cmap,
        title_pad=title_pad,
        aspect=aspect_graphon,
        vmin=vmin,
        vmax=vmax,
    )

    A, u = sample_GnW(W, n=n, seed=seed)

    plot_adj(
        A,
        ax=axes[1],
        title=raw_title,
        cmap=cmap,
        title_pad=title_pad,
        aspect=aspect_adj,
    )

    u_order = order_by_latent(u)
    plot_adj(
        A,
        order=u_order,
        ax=axes[2],
        title=sorted_title,
        cmap=cmap,
        title_pad=title_pad,
        aspect=aspect_adj,
    )

    # Compute empirical step graphon from the sorted adjacency
    B = block_densities(A, order=u_order, k=k)
    plot_step(
        B,
        ax=axes[3],
        title=step_title,
        show_colorbar=show_colorbar,
        cmap=cmap,
        title_pad=title_pad,
        aspect=aspect_graphon,
        vmin=vmin,
        vmax=vmax,
    )

    return fig, axes