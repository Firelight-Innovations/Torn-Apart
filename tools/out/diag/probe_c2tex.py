import sys, math
from pathlib import Path
_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path: sys.path.insert(0, str(_R))
import numpy as np
import main as demo
from fire_engine.core.math3d import Vec3, Quat
from panda3d.core import Texture

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = 12.0*3600.0
pipe = app.lighting_pipeline
hold = app._clock.game_time_of_day

# Warm up at origin so cascades fully assemble & propagate
for _ in range(200):
    app.taskMgr.step(); app._clock.game_time_of_day = hold

def readback(tex):
    eng = app.graphicsEngine
    eng.extract_texture_data(tex, app.win.get_gsg())
    pta = tex.get_ram_image()
    if not pta: return None
    comp = tex.get_num_components()
    ct = tex.get_component_type()
    if ct == Texture.T_float:
        arr = np.frombuffer(memoryview(pta), dtype=np.float16 if tex.get_component_width()==2 else np.float32)
    else:
        arr = np.frombuffer(memoryview(pta), dtype=np.uint8)
    z=tex.get_z_size(); y=tex.get_y_size(); x=tex.get_x_size()
    return arr.reshape(z,y,x,comp)  # BGRA, page-major

for casc in pipe.cascades:
    geom = readback(casc.geom)
    vis  = readback(casc.vis)
    rad  = readback(casc.radiance_current)
    print(f"\n=== cascade {casc.index} cell_m={casc.cell_m} cells={casc.cells} ===")
    if geom is None:
        print("  geom readback FAILED"); continue
    occ = geom[...,3].astype(np.float32)/255.0 if geom.dtype==np.uint8 else geom[...,3]
    solidfrac = float((occ>0.5).mean())
    print(f"  geom occ solid_fraction={solidfrac:.4f}")
    # vis is rgba16f BGRA: channel order after frombuffer is B,G,R,A => sun=R is index2
    if vis is not None:
        v = vis.astype(np.float32)
        # BGRA layout: index0=B(skyVis), 1=G(moonVis),2=R(sunVis),3=A
        sun = v[...,2]; sky=v[...,0]
        print(f"  vis sun: min={sun.min():.3f} max={sun.max():.3f} mean={sun.mean():.3f}")
        print(f"  vis sky: min={sky.min():.3f} max={sky.max():.3f} mean={sky.mean():.3f}")
    if rad is not None:
        r = rad.astype(np.float32)
        mag = np.linalg.norm(r[...,:3],axis=-1)
        print(f"  radiance mag: min={mag.min():.4f} max={mag.max():.4f} mean={mag.mean():.4f}")
        # near-ground slice: cells around z where ground is (z~8m). cell index = (8 - origin_z)/cell_m
        oz = casc.window.world_origin_m[2]
        kz = int((8.0 - oz)/casc.cell_m)
        if 0<=kz<casc.cells:
            # page-major => first index is z
            sl_sun = v[kz][...,2]
            sl_rad = mag[kz]
            print(f"  z=8m slice (k={kz}): sun vis mean={sl_sun.mean():.3f} rad mean={sl_rad.mean():.4f}")

print("\nsun_dir/radiance/ambient:", pipe._last_sun)
import os,sys; sys.stdout.flush(); os._exit(0)
