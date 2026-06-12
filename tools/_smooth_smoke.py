"""
tools/_smooth_smoke.py — scratch: minimal compute-dispatch test of smooth.comp.

No demo boot: offscreen GSG, tiny 16^3 volume, all-air geometry, zero
source/lit, random-noise radiance.  After one SMOOTH dispatch the readback
must be the air-masked 3^3 local mean of the input (variance way down,
interior cell exactly equal to the 27-cell mean).  Verifies the shader
compiles and the image bindings are live — independent of the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from panda3d.core import loadPrcFileData  # noqa: E402

loadPrcFileData("", "window-type offscreen\naudio-library-name null\n"
                    "notify-level-glgsg error\ngl-debug true")

from direct.showbase.ShowBase import ShowBase  # noqa: E402
from panda3d.core import (  # noqa: E402
    NodePath, SamplerState, Shader, ShaderAttrib, Texture)

from fire_engine.lighting import glsl  # noqa: E402

N = 16


def make_tex(name: str, hdr: bool) -> Texture:
    t = Texture(name)
    if hdr:
        t.setup_3d_texture(N, N, N, Texture.T_float, Texture.F_rgba16)
    else:
        t.setup_3d_texture(N, N, N, Texture.T_unsigned_byte, Texture.F_rgba8)
    t.set_clear_color((0.0, 0.0, 0.0, 0.0))
    t.set_minfilter(SamplerState.FT_nearest)
    t.set_magfilter(SamplerState.FT_nearest)
    return t


def upload_f32(t: Texture, arr: np.ndarray) -> None:
    """arr: float32 [z, y, x, rgba] -> BGRA ram image."""
    bgra = arr[..., [2, 1, 0, 3]].astype(np.float32)
    t.set_ram_image(np.ascontiguousarray(bgra).tobytes())


def readback(base, t: Texture) -> np.ndarray:
    gsg = base.win.get_gsg()
    assert base.graphicsEngine.extract_texture_data(t, gsg)
    arr = np.frombuffer(t.get_ram_image(), dtype=np.float32).copy()
    return arr.reshape(N, N, N, 4)[..., [2, 1, 0, 3]]


def main() -> None:
    base = ShowBase()
    geom = make_tex("geom", hdr=False)          # all zeros = all air
    source = make_tex("source", hdr=True)
    lit = make_tex("lit", hdr=True)
    src = make_tex("src", hdr=True)
    dst = make_tex("dst", hdr=True)

    rng = np.random.default_rng(7)
    noise = np.zeros((N, N, N, 4), np.float32)
    noise[..., :3] = rng.random((N, N, N, 3)).astype(np.float32)
    noise[..., 3] = 1.0
    upload_f32(src, noise)
    # geom/source/lit keep their clear color (0) — but force ram images so
    # the upload path is identical for all.
    zeros8 = np.zeros((N, N, N, 4), np.uint8)
    geom.set_ram_image(zeros8.tobytes())
    zsrc = np.zeros((N, N, N, 4), np.float32)
    upload_f32(source, zsrc)
    upload_f32(lit, zsrc)
    upload_f32(dst, zsrc)

    np_node = NodePath("smooth_test")
    shader = Shader.make_compute(Shader.SL_GLSL, glsl.SMOOTH_COMPUTE)
    np_node.set_shader(shader)
    np_node.set_shader_input("u_geom", geom)
    np_node.set_shader_input("u_source", source)
    np_node.set_shader_input("u_lit", lit)
    np_node.set_shader_input("u_src", src)
    np_node.set_shader_input("u_dst", dst)
    np_node.set_shader_input("u_cells", N)

    gsg = base.win.get_gsg()
    base.graphicsEngine.dispatch_compute(
        (N // 4, N // 4, N // 4), np_node.get_attrib(ShaderAttrib), gsg)

    out = readback(base, dst)
    # Expected: air-masked (here: full) 3^3 local mean with edge clamping.
    inp = noise[..., 0]
    got = out[..., 0]
    print(f"input  r: mean {inp.mean():.4f} std {inp.std():.4f}")
    print(f"output r: mean {got.mean():.4f} std {got.std():.4f}")
    # Interior reference value at (8,8,8): plain 27-cell mean.
    ref = inp[7:10, 7:10, 7:10].mean()
    print(f"interior cell got {got[8, 8, 8]:.4f} expected {ref:.4f}")
    ok = abs(got[8, 8, 8] - ref) < 2e-2 and got.std() < 0.6 * inp.std()
    print("SMOKE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
