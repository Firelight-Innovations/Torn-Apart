"""tools/_tree_geom_probe.py — throwaway: render a tree variant mesh to PNG.

Headless matplotlib render of a TreeMesh (wood brown, leaves green) from a few
angles, so branch socketing and leaf placement/orientation/density can be
judged WITHOUT the game's scatter, lighting, or GPU.  Diagnostic only (leading
underscore), not part of the suite.

    python tools/_tree_geom_probe.py tree_gnarled_oak --variant 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("species", default="tree_gnarled_oak", nargs="?")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(args.seed)
    vs = get(args.species)
    m = vs.meshes[args.variant]
    tris = m.indices.reshape(-1, 3)
    leaf_vert = m.uvs[:, 0] >= 0.5
    is_leaf = leaf_vert[tris].all(axis=1)
    P = m.positions

    wood = P[tris[~is_leaf]]
    leaf = P[tris[is_leaf]]
    print(
        f"{args.species} v{args.variant}: {len(tris)} tris, "
        f"{(~is_leaf).sum()} wood, {is_leaf.sum()} leaf"
    )

    fig = plt.figure(figsize=(15, 6))
    for i, (elev, azim, title) in enumerate(
        [(8, -88, "front (-Y)"), (8, 0, "side (+X)"), (40, -60, "above")]
    ):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.add_collection3d(
            Poly3DCollection(wood, facecolor="#6b4f31", edgecolor="#3a2a18", linewidths=0.15)
        )
        if len(leaf):
            ax.add_collection3d(
                Poly3DCollection(leaf, facecolor="#5a8a36", edgecolor="none", alpha=0.85)
            )
        r = float(np.abs(P[:, :2]).max()) + 0.3
        top = float(P[:, 2].max()) + 0.3
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(0, top)
        ax.set_box_aspect((1, 1, top / (2 * r)))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_axis_off()

    out = os.path.join(_ROOT, "tools", "out", "trees", f"_geom_{args.species}_v{args.variant}.png")
    fig.savefig(out, dpi=90, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
