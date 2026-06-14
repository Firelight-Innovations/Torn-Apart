import sys, math
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import numpy as np, main as demo
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
cam.local_rotation = Quat.from_axis_angle(
    Vec3.RIGHT, math.radians(-12.0)
).normalized()  # shallow, see horizon/edge
y = float(cam.position.y)
# stand at y=470 (just inside +500 edge), look toward +Y edge
for i in range(int((470 - y) / 1.2)):
    y += 1.2
    cam.position = Vec3(0.0, y, 12.0)
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
cam.position = Vec3(0.0, 470.0, 12.0)
for _ in range(30):
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
img = PNMImage()
app.win.get_screenshot(img)
img.write(Filename.from_os_specific(str(_R / "tools" / "out" / "diag" / "world_edge_nofog.png")))
print("wrote edge shot at cam_y", y)
sys.stdout.flush()
os._exit(0)
