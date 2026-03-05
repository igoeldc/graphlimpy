from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import matplotlib.pyplot as plt

from graphlimpy.graphons import bipartite
from graphlimpy.sample import sample_GnW
from graphlimpy.step import empirical_step_graphon
from graphlimpy.viz import plot_graphon, plot_adj, plot_step, order_by_latent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    W = bipartite(split=0.55, p_in=0.05, p_out=0.45)
    A, u = sample_GnW(W, n=args.n, seed=args.seed)

    order = order_by_latent(u)
    B, W_hat = empirical_step_graphon(A, order=order, k=args.k, return_callable=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.3, h_pad=0.2)

    plot_graphon(W, ax=axes[0], title="W", cmap="gray_r", show_colorbar=False, show_diagonal=True)
    plot_adj(A, ax=axes[1], title="A (raw)", cmap="gray_r")
    plot_adj(A, ax=axes[2], order=order, title="A (sorted by u)", cmap="gray_r")
    plot_step(B, ax=axes[3], title=f"Empirical step (k={args.k})", cmap="gray_r", show_colorbar=False)

    plt.show()


if __name__ == "__main__":
    main()