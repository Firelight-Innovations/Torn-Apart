import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import os

import main as demo
from fire_engine.core.math3d import Vec3

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = 12.0 * 3600.0
hold = app._clock.game_time_of_day
pipe = app.lighting_pipeline
cam = app.camera_go.transform


# Fly continuously and, each frame, check: is any LOADED terrain chunk's surface
# OUTSIDE all three cascade boxes (-> shader fallback)? And track c2 lag.
def chunk_surface_outside_all(camy):
    # loaded chunk coords
    cm_m = pipe._config.chunk_meters
    worst = None
    boxes = []
    for c in pipe.cascades:
        if c.window.origin_cell is None:
            boxes.append(None)
            continue
        om = c.window.world_origin_m
        sz = c.window.size_m
        boxes.append((om, sz))
    n_out = 0
    n_tot = 0
    for cx, cyc, cz in app.chunk_manager.chunks.keys():
        if cz != 0:
            continue
        # chunk center world xy, surface z~8
        wx = (cx + 0.5) * cm_m
        wy = (cyc + 0.5) * cm_m
        wz = 8.0
        n_tot += 1
        inside = False
        for b in boxes:
            if b is None:
                continue
            om, sz = b
            if all(om[i] <= (wx, wy, wz)[i] < om[i] + sz for i in range(3)):
                inside = True
                break
        if not inside:
            n_out += 1
    return n_out, n_tot


y = float(cam.position.y)
maxout = 0
for i in range(700):
    y += 1.2
    cam.position = Vec3(0.0, y, 10.0)
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
    if i % 40 == 0:
        no, nt = chunk_surface_outside_all(y)
        c2 = pipe.cascades[2]
        print(
            f"frame {i} cam_y={y:.0f} chunks={nt} surf_outside_all_cascades={no} c2_inflight={c2._assembly_inflight} c2_box_y=[{c2.window.world_origin_m[1]:.0f}..{c2.window.world_origin_m[1] + c2.window.size_m:.0f}]"
        )
        maxout = max(maxout, no)
print("MAX chunks with surface outside all cascades during flight:", maxout)
sys.stdout.flush()
os._exit(0)
