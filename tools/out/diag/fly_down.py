import sys, math
from pathlib import Path
_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path: sys.path.insert(0, str(_R))
import numpy as np, main as demo
from fire_engine.core.math3d import Vec3, Quat
from panda3d.core import PNMImage, Filename
import os

def shot(name, fly_frames, fog, pitch, height=10.0):
    app=demo.build_demo()
    app.input_state.mouse_captured=False; app._set_mouse_capture(False)
    app._clock.game_time_of_day=12.0*3600.0; hold=app._clock.game_time_of_day
    pipe=app.lighting_pipeline
    if not fog:
        pipe.fog_enabled=False; pipe.bind_surface_inputs(app.terrain_root)
    cam=app.camera_go.transform
    y=float(cam.position.y)
    for i in range(fly_frames):
        y+=1.0; cam.position=Vec3(0.0,y,height)
        app.taskMgr.step(); app._clock.game_time_of_day=hold
    cam.position=Vec3(0.0,y,height)
    cam.local_rotation=Quat.from_axis_angle(Vec3.RIGHT,math.radians(pitch)).normalized()
    for _ in range(10):
        app.taskMgr.step(); app._clock.game_time_of_day=hold
    out=_R/"tools"/"out"/"diag"/name
    img=PNMImage()
    if app.win.get_screenshot(img): img.write(Filename.from_os_specific(str(out)))
    # mean pixel brightness in lower third
    w,h=img.get_x_size(),img.get_y_size()
    s=0;n=0
    for yy in range(int(h*0.6),h,5):
        for xx in range(0,w,10):
            s+=img.get_bright(xx,yy);n+=1
    print(name,"lower-third mean bright=%.3f"%(s/n),"cam_y=%.0f"%y)
    sys.stdout.flush()

shot("fly0_down45_nofog.png", 0, False, -45.0)      # spawn, look down, no fog
shot("fly600_down45_nofog.png", 600, False, -45.0)  # moved 600m, look down, no fog
os._exit(0)
