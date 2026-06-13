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
def report(tag):
    y=float(cam.position.y)
    tr=app.terrain_root
    print(f"\n[{tag}] cam_y={y:.0f}")
    # inspect a few chunk nodes far out
    kids=tr.get_children()
    shown=0
    for i in range(kids.get_num_paths()):
        np_=kids[i]
        p=np_.get_pos()
        b=np_.get_tight_bounds()
        nm=np_.get_name()
        if abs(p[1]-y)<40 and shown<6:  # near camera y
            verts=0
            for g in np_.find_all_matches("**/+GeomNode"):
                gn=g.node()
                for gi in range(gn.get_num_geoms()):
                    verts+=gn.get_geom(gi).get_vertex_data().get_num_rows()
            bb = ([round(v,0) for v in b[0]],[round(v,0) for v in b[1]]) if b else None
            print(f"  {nm} pos={[round(v,0) for v in p]} verts={verts} bounds={bb} culled={np_.is_hidden()}")
            shown+=1

for _ in range(120): app.taskMgr.step(); app._clock.game_time_of_day=hold
report("spawn")
y=float(cam.position.y)
for i in range(700):
    y+=1.2; cam.position=Vec3(0.0,y,10.0); app.taskMgr.step(); app._clock.game_time_of_day=hold
for _ in range(20): app.taskMgr.step(); app._clock.game_time_of_day=hold
report("moved 840m")
sys.stdout.flush(); os._exit(0)
