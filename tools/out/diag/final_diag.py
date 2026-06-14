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
cam.local_rotation = Quat.from_axis_angle(Vec3.RIGHT, math.radians(-35.0)).normalized()
y = float(cam.position.y)
for i in range(700):
    y += 1.2
    cam.position = Vec3(0.0, y, 10.0)
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
cam.position = Vec3(0.0, y, 10.0)
for _ in range(25):
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
img = PNMImage()
app.win.get_screenshot(img)
img.write(Filename.from_os_specific(str(_R / "tools" / "out" / "diag" / "moved_NOFOG_fwd.png")))

# Are the chunks in front meshed with real geometry?
cm = app.chunk_manager
front_meshed = 0
front_total = 0
tr = app.terrain_root
geom_world_ys = []
for gpath in tr.find_all_matches("**/+GeomNode"):
    b = gpath.get_tight_bounds()
    if b:
        cy = (b[0][1] + b[1][1]) * 0.5
        geom_world_ys.append(cy)
geom_world_ys.sort()
print("cam_y=%.0f" % y)
print("num geom nodes with bounds:", len(geom_world_ys))
if geom_world_ys:
    print("geom world-y min=%.0f max=%.0f" % (geom_world_ys[0], geom_world_ys[-1]))
    infront = [v for v in geom_world_ys if v > y]
    print("geoms in front of camera (y>cam_y):", len(infront), "max_y=%.0f" % (max(geom_world_ys)))
    print("gap from cam to nearest front geom:", (min(infront) - y) if infront else "NONE IN FRONT")
sys.stdout.flush()
os._exit(0)
