import sys, math
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import numpy as np, main as demo
from fire_engine.core.math3d import Vec3, Quat
from panda3d.core import PNMImage, Filename
import os

FLY = int(sys.argv[1])
NAME = sys.argv[2]
TOD = float(sys.argv[3])
app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = TOD * 3600.0
hold = app._clock.game_time_of_day
cam = app.camera_go.transform
cam.local_rotation = Quat.from_axis_angle(Vec3.RIGHT, math.radians(-35.0)).normalized()
y = float(cam.position.y)
for i in range(FLY):
    y += 1.2
    cam.position = Vec3(0.0, y, 10.0)
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
cam.position = Vec3(0.0, y, 10.0)
for _ in range(20):
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
img = PNMImage()
app.win.get_screenshot(img)
out = _R / "tools" / "out" / "diag" / NAME
img.write(Filename.from_os_specific(str(out)))
w, h = img.get_x_size(), img.get_y_size()
s = 0
n = 0
for yy in range(int(h * 0.55), h, 4):
    for xx in range(0, w, 8):
        s += img.get_bright(xx, yy)
        n += 1
print(
    NAME,
    "ground-region mean bright=%.3f" % (s / n),
    "exposure=%.2f" % app.lighting_pipeline.exposure,
)
sys.stdout.flush()
os._exit(0)
