import math
import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import os

import main as demo
from fire_engine.core.math3d import Quat, Vec3

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = 12.0 * 3600.0
hold = app._clock.game_time_of_day
cam = app.camera_go.transform
cam.local_rotation = Quat.from_axis_angle(Vec3.RIGHT, math.radians(-35.0)).normalized()
y = float(cam.position.y)
for i in range(700):
    y += 1.2
    cam.position = Vec3(0.0, y, 10.0)
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
for _ in range(20):
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
tr = app.terrain_root
# Find a chunk whose name maps to a y near the camera (cy ~ 51 => world 816..832)
target = None
for k in tr.get_children():
    nm = k.get_name()  # chunk_cx_cy_cz
    parts = nm.split("_")
    if len(parts) == 4 and parts[2].lstrip("-").isdigit():
        cy = int(parts[2])
        if 50 <= cy <= 52 and parts[1] == "0":
            target = k
            break
print("target node:", target.get_name() if target else None)
if target:
    print("  pos:", [round(v, 1) for v in target.get_pos()])
    print("  net transform pos:", [round(v, 1) for v in target.get_pos(app.render)])
    b = target.node().get_bounds()
    print("  node.get_bounds():", b)
    tb = target.get_tight_bounds()
    print(
        "  tight_bounds:",
        ([round(v, 0) for v in tb[0]], [round(v, 0) for v in tb[1]]) if tb else None,
    )
    # geom count + first vertex
    gn = target.node()
    print("  is GeomNode:", gn.is_geom_node())
    if gn.is_geom_node():
        print("  num geoms:", gn.get_num_geoms())
        for gi in range(min(1, gn.get_num_geoms())):
            g = gn.get_geom(gi)
            vd = g.get_vertex_data()
            from panda3d.core import GeomVertexReader

            r = GeomVertexReader(vd, "vertex")
            v0 = r.get_data3()
            print("  first vertex (object space):", [round(x, 1) for x in v0])
            print("  geom bounds:", g.get_bounds())
    # bounded by which volume?
    print("  bounds_type:", target.node().get_bounds_type())
    # Is it being culled? check final state
    print("  is_hidden:", target.is_hidden())
sys.stdout.flush()
os._exit(0)
