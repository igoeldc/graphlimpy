from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import matplotlib.pyplot as plt

from graphlimpy.graphons import bipartite, half_graphon, ramp
from graphlimpy.viz import plot_sampling_4panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["bipartite", "half", "ramp"], default="bipartite")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=20, help="Number of blocks for empirical step graphon")
    ap.add_argument("--m", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.model == "bipartite":
        W = bipartite()
    elif args.model == "half":
        W = half_graphon()
    else:
        W = ramp()

    plot_sampling_4panel(
        W,
        n=args.n,
        k=args.k,
        m=args.m,
        seed=args.seed,
        graphon_title=f"{args.model.capitalize()} Graphon (W)",
        raw_title=f"Sampled A (n={args.n})",
        sorted_title="Sorted by latent $u$",
        step_title=f"Empirical Step ($k={args.k}$)",
        cmap="gray_r",  # 0=white, 1=black
        show_diagonal=True,
        show_colorbar=True,
    )
    plt.show()


if __name__ == "__main__":
    main()