"""tools/probe_grass_crater.py — verify craters cull grass blades.

Boots the demo, carves a SphereBrush crater in the middle of the demo grass
volume, moves the camera close, steps frames (height-field re-bake happens on
TerrainEditedEvent) and saves a screenshot.  Dev diagnostics only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as demo  # noqa: E402
from fire_engine.core.math3d import Vec3, Quat  # noqa: E402
from fire_engine.world.terrain import SphereBrush, BrushMode, apply_brush  # noqa: E402

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)

# Carve a crater dead-centre in the demo grass volume (surface at z=8).
apply_brush(
    SphereBrush(3.0),
    Vec3(0.0, 10.0, 8.0),
    BrushMode.REMOVE,
    material=1,
    chunk_provider=app.chunk_manager.get_or_create,
    bus=app._event_bus,
)

# Camera just outside the volume looking at the crater.
app.camera_go.transform.position = Vec3(0.0, -6.0, 12.0)
app.camera_go.transform.local_rotation = Quat.from_axis_angle(
    Vec3.RIGHT, math.radians(-14.0)
).normalized()

for _ in range(200):
    app.taskMgr.step()

app.win.save_screenshot("tools/out/grass_crater.png")
print("wrote tools/out/grass_crater.png")
