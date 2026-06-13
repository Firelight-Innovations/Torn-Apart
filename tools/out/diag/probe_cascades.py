import sys, math
from pathlib import Path
_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path: sys.path.insert(0, str(_R))
import numpy as np
import main as demo
from fire_engine.core.math3d import Vec3, Quat

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = 12.0*3600.0

pipe = app.lighting_pipeline
print("pipeline:", type(pipe).__name__)

def dump(tag):
    print("===", tag, "cam=", tuple(round(float(x),1) for x in app.camera_go.transform.position))
    for casc in pipe.cascades:
        w = casc.window
        oc = w.origin_cell
        if oc is None:
            print(f"  c{casc.index}: origin_cell=None inflight={casc._assembly_inflight}")
            continue
        om = w.world_origin_m
        size = w.size_m
        print(f"  c{casc.index}: cells={casc.cells} cell_m={casc.cell_m} origin_m={tuple(round(v,1) for v in om)} "
              f"box=[{om[0]:.0f}..{om[0]+size:.0f}] inflight={casc._assembly_inflight} needs_inject={casc.needs_inject}")

# warm up at origin
hold = app._clock.game_time_of_day
for _ in range(120):
    app.taskMgr.step(); app._clock.game_time_of_day = hold
dump("after spawn warmup")

# Now teleport camera far away (well beyond streaming + cascade 0/1) and keep stepping
for step_target in [(0,40,10),(0,120,10),(0,300,10),(0,600,10)]:
    app.camera_go.transform.position = Vec3(*[float(x) for x in step_target])
    for _ in range(60):
        app.taskMgr.step(); app._clock.game_time_of_day = hold
    dump(f"moved to {step_target}")

# Check which cascade covers camera feet & a far point
def covers(casc, wp):
    oc = casc.window.origin_cell
    if oc is None: return None
    om = casc.window.world_origin_m
    cm, n = casc.cell_m, casc.cells
    uv = [(wp[i]-om[i])/(cm*n) for i in range(3)]
    inb = all(0.02 < u < 0.98 for u in uv[:1]) and all(0<u<1 for u in uv)
    return inb, tuple(round(u,3) for u in uv)

cam = app.camera_go.transform.position
feet = (float(cam.x), float(cam.y), 8.0)
print("\nFEET COVERAGE at", feet)
for casc in pipe.cascades:
    print(f"  c{casc.index}:", covers(casc, feet))

# how many chunks loaded?
print("\nloaded chunks:", len(app.chunk_manager.chunks))
# bounds of loaded chunks
keys = list(app.chunk_manager.chunks.keys())
xs=[k[0] for k in keys]; ys=[k[1] for k in keys]
print("chunk x range", min(xs), max(xs), "y range", min(ys), max(ys))
import os,sys; sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
