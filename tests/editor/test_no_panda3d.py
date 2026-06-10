"""Import-graph guard: ``fire_editor`` must never pull in panda3d.

EDITOR_PRD hard rule 1: the editor's whole premise is running with the game
closed; panda3d in the daemon is a regression. We assert this in a *fresh*
subprocess so the result is independent of whatever other tests imported in the
shared pytest process.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_EDITOR = _ROOT / "editor"

_PROBE = """
import sys
# Import the whole daemon surface, including the entry module.
import fire_editor
import fire_editor.daemon
import fire_editor.server
import fire_editor.rpc
import fire_editor.binary
import fire_editor.services
leaked = [m for m in sys.modules if m == "panda3d" or m.startswith("panda3d.")]
if leaked:
    print("PANDA3D LEAKED:", leaked)
    sys.exit(1)
print("clean")
"""


def test_fire_editor_imports_no_panda3d():
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(_ROOT),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "clean" in proc.stdout


def _subprocess_env() -> dict:
    import os

    env = dict(os.environ)
    extra = os.pathsep.join([str(_ROOT), str(_EDITOR)])
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env
