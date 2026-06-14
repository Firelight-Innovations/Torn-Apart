import sys, math
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import numpy as np, main as demo
from fire_engine.core.math3d import Vec3, Quat
from panda3d.core import Texture
import os

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = 12.0 * 3600.0
hold = app._clock.game_time_of_day
pipe = app.lighting_pipeline


def rb(tex):
    app.graphicsEngine.extract_texture_data(tex, app.win.get_gsg())
    pta = tex.get_ram_image()
    if not pta:
        return None
    comp = tex.get_num_components()
    if tex.get_component_type() == Texture.T_float:
        arr = np.frombuffer(memoryview(pta), dtype=np.float16).astype(np.float32)
    else:
        arr = np.frombuffer(memoryview(pta), dtype=np.uint8).astype(np.float32)
    z, y, x = tex.get_z_size(), tex.get_y_size(), tex.get_x_size()
    comp = arr.size // (z * y * x)
    return arr.reshape(z, y, x, comp)


def sample(casc, wp, tex_arr):
    # nearest sample, page-major (z,y,x), BGRA
    om = casc.window.world_origin_m
    cm = casc.cell_m
    n = casc.cells
    uv = [(wp[i] - om[i]) / (cm * n) for i in range(3)]
    if not all(0 <= u < 1 for u in uv):
        return None
    ix = [min(n - 1, max(0, int(uv[i] * n))) for i in range(3)]
    px = tex_arr[ix[2], ix[1], ix[0]]
    return px  # BGRA


def report(tag):
    print(f"\n### {tag}: exposure={pipe.exposure:.3f} exposure_sky={pipe.exposure_sky:.3f}")
    sun = pipe._last_sun
    sun_dir = np.array(sun[0])
    sun_rad = np.array(sun[1])
    amb = np.array(sun[4])
    print("  sun_rad", sun_rad.round(2), "amb", amb.round(3))
    vis_t = [rb(c.vis) for c in pipe.cascades]
    rad_t = [rb(c.radiance_current) for c in pipe.cascades]
    cam = app.camera_go.transform.position
    # Evaluate the surface composite for a ground point at the camera's xy, z=8
    for dist in [0.0, 30.0, 120.0, 250.0, 400.0]:
        wp = (float(cam.x), float(cam.y) - dist, 8.4)  # behind camera, just above ground
        # mimic sampleCascades priority c0->c1->c2->fallback
        chosen = None
        for ci, c in enumerate(pipe.cascades):
            om = c.window.world_origin_m
            cm = c.cell_m
            n = c.cells
            uv = [(wp[i] - om[i]) / (cm * n) for i in range(3)]
            pad = [0.02, 0.01, 0.005][ci]
            if all(pad < u < 1 - pad for u in uv):
                v = sample(c, wp, vis_t[ci])
                r = sample(c, wp, rad_t[ci])
                chosen = (ci, v, r)
                break
        if chosen is None:
            # fallback
            rad = amb * 0.6
            vis = np.array([1, 1, 1.0])
            src = "FALLBACK"
            sunvis = 1.0
        else:
            ci, v, r = chosen
            # BGRA: vis sun=R=idx2, radiance rgb = B,G,R idx0,1,2 -> reorder
            sunvis = v[2]
            rad = np.array([r[2], r[1], r[0]])
            src = f"c{ci}"
        NdotL = max(sun_dir[2], 0.0)  # flat ground normal up
        direct = sun_rad * (sunvis * NdotL)
        base = np.array([0.12, 0.20, 0.06])  # approx grass albedo linear
        hdr = base * (direct + rad)  # ao~1
        ldr = np.clip(hdr * pipe.exposure, 0, 4)
        print(
            f"  dist={dist:4.0f} src={src:8s} sunvis={float(sunvis):.2f} rad={rad.round(3)} hdr={hdr.round(3)} ldr~={ldr.round(2)}"
        )


# spawn
for _ in range(150):
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
report("SPAWN")
# fly forward 500m
cam = app.camera_go.transform
y = float(cam.position.y)
for i in range(500):
    y += 1.0
    cam.position = Vec3(0.0, y, 10.0)
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
report("MOVED +500m")
sys.stdout.flush()
os._exit(0)
