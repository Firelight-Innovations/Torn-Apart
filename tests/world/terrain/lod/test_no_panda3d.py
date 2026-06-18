"""
tests/world/terrain/lod/test_no_panda3d.py — Hard Rule 1 guard for lod/ core.

The whole ``world/terrain/lod/`` package must stay headless: importing it (and
the pure P2 coarse-core modules in particular) must never pull in panda3d.  We
assert this in a *fresh* subprocess so the result is independent of whatever
other tests imported into the shared pytest process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]

_PROBE = """
import sys
import fire_engine.world.terrain.lod
import fire_engine.world.terrain.lod.node
import fire_engine.world.terrain.lod.downsample
import fire_engine.world.terrain.lod.coarse_chunk
import fire_engine.world.terrain.lod.desired
leaked = [m for m in sys.modules if m == "panda3d" or m.startswith("panda3d.")]
if leaked:
    print("PANDA3D LEAKED:", leaked)
    sys.exit(1)
print("clean")
"""


def test_lod_core_imports_no_panda3d() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "clean" in proc.stdout
