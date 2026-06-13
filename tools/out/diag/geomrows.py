import sys,math
from pathlib import Path
_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path: sys.path.insert(0, str(_R))
import numpy as np, main as demo
from fire_engine.core.math3d import Vec3, Quat
import os
app=demo.build_demo()
app.input_state.mouse_captured=False; app._set_mouse_capture(False)
app._clock.game_time_of_day=12.0*3600.0; hold=app._clock.game_time_of_day
cam=app.camera_go.transform
cam.local_rotation=Quat.from_axis_angle(Vec3.RIGHT,math.radians(-35.0)).normalized()
y=float(cam.position.y)
for i in range(700):
    y+=1.2; cam.position=Vec3(0.0,y,10.0); app.taskMgr.step(); app._clock.game_time_of_day=hold
for _ in range(20): app.taskMgr.step(); app._clock.game_time_of_day=hold
tr=app.terrain_root
def info(node):
    gn=node.node()
    out=[]
    for gi in range(gn.get_num_geoms()):
        g=gn.get_geom(gi)
        vd=g.get_vertex_data()
        nb=g.get_bounds()
        out.append((vd.get_num_rows(), g.get_num_primitives(), str(nb)[:40]))
    return out
for k in tr.get_children():
    nm=k.get_name(); parts=nm.split("_")
    if len(parts)==4 and parts[1]=="0" and parts[2].lstrip("-").isdigit():
        cy=int(parts[2])
        if cy in (50,51,52,0,-1) :
            # also dump first few positions
            gn=k.node(); g=gn.get_geom(0); vd=g.get_vertex_data()
            from panda3d.core import GeomVertexReader
            try:
                rdr=GeomVertexReader(vd,"vertex"); p0=rdr.get_data3(); p0=[round(x,0) for x in p0]
            except Exception as e: p0=f"ERR {e}"
            print(nm, "geoms:",info(k), "firstpos:",p0)
sys.stdout.flush(); os._exit(0)
