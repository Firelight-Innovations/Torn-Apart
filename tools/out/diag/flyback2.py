import sys, math
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import numpy as np
import main as demo
from fire_engine.core.math3d import Vec3, Quat
from panda3d.core import PNMImage, Filename

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = 12.0 * 3600.0
hold = app._clock.game_time_of_day
pipe = app.lighting_pipeline
# DISABLE FOG to isolate lighting
pipe.fog_enabled = False
pipe.bind_surface_inputs(app.terrain_root)

cam = app.camera_go.transform
y = float(cam.position.y)
for i in range(600):
    y += 1.0
    cam.position = Vec3(0.0, y, 10.0)
    app.taskMgr.step()
    app._clock.game_time_of_day = hold

# look back and steeply down to see ground we left
cam.local_rotation = (
    Quat.from_axis_angle(Vec3.UP, math.radians(180.0))
    * Quat.from_axis_angle(Vec3.RIGHT, math.radians(-25.0))
).normalized()
for _ in range(8):
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
out = _R / "tools" / "out" / "diag" / "flyback_nofog.png"
img = PNMImage()
if app.win.get_screenshot(img):
    img.write(Filename.from_os_specific(str(out)))
print("WROTE", out)

# Also: sample what the shader fallback would give. Sample c2 at a point ~300m behind (y~280), z=8
# and a point at y~600 (within c2? c2 box computed)
for casc in pipe.cascades:
    om = casc.window.world_origin_m
    sz = casc.window.size_m
    print(f"c{casc.index} box_y=[{om[1]:.0f}..{om[1] + sz:.0f}]")
import os, sys

sys.stdout.flush()
os._exit(0)
