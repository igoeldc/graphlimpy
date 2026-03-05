from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import matplotlib.pyplot as plt

from graphlimpy.graphons import bipartite
from graphlimpy.rearrange import rearrange_graphon, random_interval_reorder, shift
from graphlimpy.viz import plot_graphon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12, help="number of intervals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift", type=float, default=None, help="if set, use rotation by this amount")
    args = ap.parse_args()

    W = bipartite(split=0.55, p_in=0.05, p_out=0.45)

    if args.shift is not None:
        phi = shift(args.shift)
        title2 = f"W ∘ shift({args.shift})"
    else:
        phi = random_interval_reorder(args.k, seed=args.seed)
        title2 = f"W ∘ interval_reorder(k={args.k})"

    Wphi = rearrange_graphon(W, phi)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    plot_graphon(W, ax=axes[0], title="W", show_colorbar=False, cmap="gray_r", show_diagonal=True)
    plot_graphon(Wphi, ax=axes[1], title=title2, show_colorbar=False, cmap="gray_r", show_diagonal=True)
    plt.show()


if __name__ == "__main__":
    main()