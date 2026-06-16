"""
tests/render/test_offscreen.py — window-marked test for the offscreen renderer.

This needs a real GL context (a GPU), so it is excluded from the default headless
run via ``addopts = -m "not window"`` and must be invoked explicitly::

    .venv/Scripts/python.exe -m pytest tests/render/test_offscreen.py -m window -q

It drives the SAME path the editor's ``world.screenshot`` RPC uses: build a world
save, then spawn ``python -m fire_engine.render._impl.offscreen`` and assert it
writes a non-empty PNG of the requested size. The orchestration around it (temp
save, argv, error handling) is covered headlessly in
tests/editor/test_screenshot_rpc.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EDITOR_DIR = _REPO_ROOT / "editor"
_SEED = 1337


def _save_world(path: Path) -> int:
    """Write a small edited world save; return its seed."""
    sys.path.insert(0, str(_EDITOR_DIR))
    from fire_editor import EditorSession

    session = EditorSession.from_seed(_SEED)
    session.save(str(path))
    return session.seed


@pytest.mark.window
def test_offscreen_render_writes_png_of_requested_size(tmp_path):
    pytest.importorskip("panda3d.core")

    ta_path = tmp_path / "world.ta"
    seed = _save_world(ta_path)
    out_png = tmp_path / "shot.png"

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), str(_EDITOR_DIR), env.get("PYTHONPATH")) if p
    )
    width, height = 320, 240
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "fire_engine.render._impl.offscreen",
            "--save",
            str(ta_path),
            "--seed",
            str(seed),
            "--px",
            "0",
            "--py",
            "-20",
            "--pz",
            "12",
            "--pitch",
            "-20",
            "--width",
            str(width),
            "--height",
            str(height),
            "--frames",
            "30",
            "--out",
            str(out_png),
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 0, f"render failed:\n{proc.stderr}"
    assert "SCREENSHOT_RESULT" in proc.stdout
    assert out_png.exists() and out_png.stat().st_size > 0

    from panda3d.core import PNMImage

    img = PNMImage()
    assert img.read(str(out_png)), "output is not a readable image"
    assert img.get_x_size() == width
    assert img.get_y_size() == height
