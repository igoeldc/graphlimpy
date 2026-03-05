from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse

from graphlimpy.graphons import bipartite
from graphlimpy.sample import sample_GnW
from graphlimpy.stats import (
    edge_density_graph,
    triangle_density_graph,
    edge_density_graphon,
    triangle_density_graphon,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=500_000, help="MC samples for graphon densities")
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400, 800])
    args = ap.parse_args()

    W = bipartite(split=0.55, p_in=0.05, p_out=0.45)

    t2W = edge_density_graphon(W, M=args.M, seed=args.seed)
    t3W = triangle_density_graphon(W, M=args.M, seed=args.seed)

    print(f"t(K2,W) ≈ {t2W}")
    print(f"t(K3,W) ≈ {t3W}")
    print()

    for n in args.ns:
        A, _ = sample_GnW(W, n=n, seed=args.seed)
        t2G = edge_density_graph(A)
        t3G = triangle_density_graph(A)
        print(f"{n} t(K2,G) = {t2G}   t(K3,G) = {t3G}")


if __name__ == "__main__":
    main()