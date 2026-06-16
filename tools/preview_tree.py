"""
tools/preview_tree.py — Headless CLI preview for TreeSpeciesDef variant sets.

The authoring feedback loop for species scripts
(``fire_engine/procedural/flora/species/*.py``): dump every variant mesh as
an OBJ (open in any viewer) and the species atlas + impostor atlas as PNGs —
no panda3d, no GPU.

Usage
-----
::

    python tools/preview_tree.py <species_name> [--seed SEED] [--obj] [--png]
    python tools/preview_tree.py --all [--png]

Outputs (under ``tools/out/trees/``):
    <name>_v<k>.obj          one per variant (--obj)
    <name>_atlas.png         bark|leaf species atlas (--png)
    <name>_impostors.png     far-LOD sprite row (--png)

Examples
--------
::

    python tools/preview_tree.py tree_gnarled_oak --obj --png
    python tools/preview_tree.py --all --png --seed 99
"""

from __future__ import annotations

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_BUILTIN_SPECIES = ("tree_gnarled_oak", "tree_dead", "bush_scrub", "bush_berry")


def _write_obj(path: str, mesh) -> None:
    """Write one TreeMesh as a Wavefront OBJ (v/vt/vn + f, 1-based)."""
    import numpy as np

    pos, uv, nrm = mesh.positions, mesh.uvs, mesh.normals
    tri = mesh.indices.reshape(-1, 3) + 1  # OBJ is 1-based
    lines = ["# Torn Apart tree variant (preview_tree.py)"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in pos]
    lines += [f"vt {u:.5f} {v:.5f}" for u, v in uv]
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in nrm]
    lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in np.asarray(tri)]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview a TreeSpeciesDef variant pool (headless)."
    )
    parser.add_argument(
        "species", nargs="?", default=None, help="Registered species name, e.g. 'tree_gnarled_oak'."
    )
    parser.add_argument("--all", action="store_true", help="Preview every built-in species.")
    parser.add_argument(
        "--seed", type=int, default=None, help="World seed override (default: config.toml)."
    )
    parser.add_argument("--obj", action="store_true", help="Write one OBJ per variant.")
    parser.add_argument("--png", action="store_true", help="Write atlas + impostor-atlas PNGs.")
    args = parser.parse_args()

    if not args.species and not args.all:
        parser.error("give a species name or --all")
    if not args.obj and not args.png:
        args.obj = args.png = True  # default: everything

    from fire_engine.core.config import load_config
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg.world_seed
    set_world_seed(seed)

    out_dir = os.path.join(_script_dir, "out", "trees")
    os.makedirs(out_dir, exist_ok=True)

    names = _BUILTIN_SPECIES if args.all else (args.species,)
    for name in names:
        try:
            vs = get(name)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.obj:
            for k, mesh in enumerate(vs.meshes):
                path = os.path.join(out_dir, f"{name}_v{k}.obj")
                _write_obj(path, mesh)
            print(f"{name}: {vs.n_variants} OBJ variant(s) -> {out_dir}")

        if args.png:
            try:
                from PIL import Image
            except ImportError:
                print("Error: Pillow not installed (pip install pillow)", file=sys.stderr)
                sys.exit(1)
            Image.fromarray(vs.atlas, mode="RGBA").save(os.path.join(out_dir, f"{name}_atlas.png"))
            Image.fromarray(vs.impostors, mode="RGBA").save(
                os.path.join(out_dir, f"{name}_impostors.png")
            )
            print(
                f"{name}: atlas {vs.atlas.shape[1]}x{vs.atlas.shape[0]}, "
                f"impostors {vs.impostors.shape[1]}x{vs.impostors.shape[0]}"
                f" -> {out_dir}"
            )

        print(f"  h_max={vs.max_height_m:.2f} m  r_max={vs.max_radius_m:.2f} m  seed={seed}")


if __name__ == "__main__":
    main()
