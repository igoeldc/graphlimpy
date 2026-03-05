from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import matplotlib.pyplot as plt

from graphlimpy.graphons import bipartite, constant
from graphlimpy.sample import sample_GnW
from graphlimpy.cut import cut_best_reordered  # from the helpers we added
from graphlimpy.viz import plot_adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--center", action="store_true", help="center A by subtracting its mean")
    args = ap.parse_args()

    # Compare two graph models by looking at the cut structure of their difference.
    W1 = bipartite(split=0.55, p_in=0.05, p_out=0.45)
    W2 = constant(0.25)

    A, _ = sample_GnW(W1, n=args.n, seed=args.seed)
    B, _ = sample_GnW(W2, n=args.n, seed=args.seed + 1)

    D = A.astype(float) - B.astype(float)
    if args.center:
        D = D - D.mean()

    res, D_re, _, _, s_size, t_size = cut_best_reordered(
        D, trials=args.trials, iters=args.iters, seed=args.seed
    )

    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    plot_adj(D_re, ax=ax, title=f"Reordered diff by cut (value={res.value:.3e})", cmap="gray_r")

    # draw the cut boundaries
    ax.axhline(s_size - 0.5, linewidth=2.0)
    ax.axvline(t_size - 0.5, linewidth=2.0)

    plt.show()


if __name__ == "__main__":
    main()