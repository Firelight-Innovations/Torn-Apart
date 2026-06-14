import sys, math
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import math, main as demo
from fire_engine.core.math3d import Vec3, Quat
from panda3d.core import PNMImage, Filename
import os

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = 12.0 * 3600.0
hold = app._clock.game_time_of_day
pipe = app.lighting_pipeline
pipe.fog_enabled = False
pipe.bind_surface_inputs(app.terrain_root)
cam = app.camera_go.transform
y = float(cam.position.y)
for i in range(600):
    y += 1.0
    cam.position = Vec3(0.0, y, 10.0)
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
cam.position = Vec3(0.0, y, 10.0)
cam.local_rotation = Quat.from_axis_angle(Vec3.RIGHT, math.radians(-45.0)).normalized()
for _ in range(12):
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
out = _R / "tools" / "out" / "diag" / "fly600_down45_nofog.png"
img = PNMImage()
app.win.get_screenshot(img)
img.write(Filename.from_os_specific(str(out)))
w, h = img.get_x_size(), img.get_y_size()
s = 0
n = 0
for yy in range(int(h * 0.6), h, 5):
    for xx in range(0, w, 10):
        s += img.get_bright(xx, yy)
        n += 1
print("fly600 lower-third mean bright=%.3f cam_y=%.0f" % (s / n, y))
sys.stdout.flush()
os._exit(0)
