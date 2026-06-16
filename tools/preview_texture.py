"""
tools/preview_texture.py — Headless CLI tool to render a ProceduralTextureDef to PNG.

Usage
-----
::

    python tools/preview_texture.py <def_name> [--seed SEED] [--width W] [--height H]

Output is written to ``tools/out/<def_name>.png``.

Requirements
------------
- Pillow (``pip install pillow``)
- No panda3d required — fully headless.

Examples
--------
::

    python tools/preview_texture.py wasteland_ground
    # → tools/out/wasteland_ground.png

    python tools/preview_texture.py wasteland_ground --seed 9999 --width 512
    # → tools/out/wasteland_ground.png  (512×256 preview with seed 9999)
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the project root is on sys.path (so ``fire_engine`` imports work when
# the script is run from any directory).
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a ProceduralTextureDef to a PNG file (headless)."
    )
    parser.add_argument(
        "def_name",
        help="Registered ProceduralDef name, e.g. 'wasteland_ground'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=("World seed override.  Defaults to the value in config.toml (or 0 if not set)."),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Texture width in pixels (passed as a param override).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Texture height in pixels (passed as a param override).",
    )
    args = parser.parse_args()

    # --- Load config (for default seed) ---
    from fire_engine.core.config import load_config
    from fire_engine.core.rng import set_world_seed

    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg.world_seed
    set_world_seed(seed)

    # --- Import procedural package (auto-registers all built-in defs) ---
    from fire_engine.procedural import get

    # --- Build optional param overrides ---
    params: dict = {}
    if args.width is not None:
        params["width"] = args.width
    if args.height is not None:
        params["height"] = args.height

    # --- Generate ---
    try:
        arr = get(args.def_name, **params)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    import numpy as np

    if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 4:
        print(
            f"Error: '{args.def_name}' did not return a (H,W,4) ndarray; "
            f"got type={type(arr)}, shape={getattr(arr, 'shape', 'N/A')}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Write PNG via Pillow (headless — no panda3d) ---
    try:
        from PIL import Image
    except ImportError:
        print(
            "Error: Pillow is not installed.  Run: pip install pillow",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = os.path.join(_script_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{args.def_name}.png")
    img = Image.fromarray(arr, mode="RGBA")
    img.save(out_path)

    print(f"Saved: {out_path}  ({arr.shape[1]}×{arr.shape[0]} px, seed={seed})")


if __name__ == "__main__":
    main()
