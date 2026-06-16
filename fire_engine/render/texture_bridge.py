"""
world/texture_bridge.py — Convert numpy RGBA arrays to Panda3D Texture objects.

This module is the only place that couples the procedural-texture pipeline to
Panda3D.  It lives in ``world/`` (the only package permitted to import panda3d
for non-GPU work — see ARCHITECTURE.md §3 and the hard rules in CLAUDE.md).

``procedural/`` and all its callers remain headless-testable; only the final
upload step (this file) touches Panda3D.

Public API
----------
``to_panda_texture(rgba) -> panda3d.core.Texture``
    Convert a ``(H, W, 4) uint8`` RGBA numpy array to a Panda3D ``Texture``
    configured for the engine's **retro nearest-neighbour** look.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get
    from fire_engine.render.texture_bridge import to_panda_texture

    set_world_seed(42)
    arr = get("wasteland_ground")               # (256, 256, 4) uint8
    tex = to_panda_texture(arr)                 # panda3d.core.Texture
    # Attach to geometry:
    #   node_path.set_texture(tex)
"""

from __future__ import annotations

import numpy as np
from panda3d.core import SamplerState, Texture

__all__ = [
    "to_data_texture_f32",
    "to_field_texture",
    "to_panda_cubemap",
    "to_panda_texture",
    "to_panda_texture_3d",
]


def to_panda_texture(rgba: np.ndarray) -> Texture:
    """
    Convert a numpy RGBA array to a Panda3D ``Texture`` with nearest-neighbour
    filtering (retro hard-pixel look).

    The conversion uses a single bulk ``set_ram_image`` call via a
    ``memoryview`` — no per-pixel Python loops.

    Panda3D stores 2-D textures with row 0 at the *bottom* (OpenGL convention),
    so the array is vertically flipped before upload to match the expected UV
    mapping (UV origin at bottom-left).

    Parameters
    ----------
    rgba : numpy.ndarray
        Shape ``(H, W, 4)``, dtype ``uint8``, RGBA channel order.
        - H and W must be positive integers.
        - Channel 3 is alpha; 255 = fully opaque.
        - Typically produced by ``procedural.get("texture_name")``.

    Returns
    -------
    panda3d.core.Texture
        A Panda3D ``Texture`` object configured as:
        - Format: ``Texture.F_rgba``
        - Type:   ``Texture.T_unsigned_byte``
        - Minification filter:  ``SamplerState.FT_nearest``
        - Magnification filter: ``SamplerState.FT_nearest``
        The texture is ready to attach to a ``NodePath`` via
        ``node_path.set_texture(tex)``.

    Raises
    ------
    ValueError
        If *rgba* does not have shape ``(H, W, 4)`` or dtype ``uint8``.

    Example
    -------
    ::

        import numpy as np
        from fire_engine.render.texture_bridge import to_panda_texture

        # Solid red 4×4 texture
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[..., 0] = 255   # R
        arr[..., 3] = 255   # A
        tex = to_panda_texture(arr)
        assert tex.get_x_size() == 4
        assert tex.get_y_size() == 4
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"to_panda_texture expects shape (H, W, 4), got {rgba.shape}")
    if rgba.dtype != np.uint8:
        raise ValueError(f"to_panda_texture expects dtype uint8, got {rgba.dtype}")

    H, W = rgba.shape[:2]

    tex = Texture()
    tex.setup_2d_texture(W, H, Texture.T_unsigned_byte, Texture.F_rgba)

    # Panda3D UV origin is bottom-left (OpenGL convention); flip vertically
    # so the image appears right-side-up.  Panda3D RAM images for F_rgba are
    # stored **BGRA** byte order (its native component order), so reorder the
    # channels R<->B on the way in or every texture renders blue-for-brown.
    flipped = rgba[::-1]  # vertical flip (view)
    bgra = np.ascontiguousarray(flipped[..., [2, 1, 0, 3]])  # RGBA -> BGRA

    # Bulk RAM upload — one memoryview write, no per-pixel loop.
    tex.set_ram_image(bytes(bgra))

    # Nearest-neighbour filters for the retro hard-pixel look.
    tex.set_minfilter(SamplerState.FT_nearest)
    tex.set_magfilter(SamplerState.FT_nearest)

    return tex


def to_panda_cubemap(faces: np.ndarray) -> Texture:
    """
    Upload a 6-face RGBA cube map (e.g. ``procedural.get("night_sky_cube")``).

    The input must follow the OpenGL cube-map convention produced by
    ``procedural.textures.night_sky.cube_face_directions``: face order
    +X, −X, +Y, −Y, +Z, −Z and array row 0 = the ``tc = −1`` texel row.
    Pages upload in that exact order with NO vertical flip (GL cube faces
    are t-down, unlike regular GL textures — flipping here would mirror
    every face), only the engine-wide RGBA→BGRA channel swap.  Sampling
    ``texture(samplerCube, dir)`` in GLSL then returns the texel that was
    generated for ``dir``.

    Parameters
    ----------
    faces : numpy.ndarray
        Shape ``(6, S, S, 4)``, dtype ``uint8``, S a power of two.

    Returns
    -------
    panda3d.core.Texture
        Nearest-filtered cube map ready for ``set_shader_input``.  Nearest
        (not linear) keeps the engine-wide **retro hard-pixel** look — the
        star/galaxy sky reads as crisp pixels instead of a smeared bilinear
        blur.
    """
    if (
        faces.ndim != 4
        or faces.shape[0] != 6
        or faces.shape[3] != 4
        or faces.shape[1] != faces.shape[2]
    ):
        raise ValueError(f"to_panda_cubemap expects shape (6, S, S, 4), got {faces.shape}")
    if faces.dtype != np.uint8:
        raise ValueError(f"to_panda_cubemap expects dtype uint8, got {faces.dtype}")

    size = faces.shape[1]
    tex = Texture("cubemap")
    tex.setup_cube_map(size, Texture.T_unsigned_byte, Texture.F_rgba)
    bgra = np.ascontiguousarray(faces[..., [2, 1, 0, 3]])  # RGBA -> BGRA
    tex.set_ram_image(bytes(bgra))
    tex.set_minfilter(SamplerState.FT_nearest)
    tex.set_magfilter(SamplerState.FT_nearest)
    return tex


def to_panda_texture_3d(volume: np.ndarray, *, linear: bool = True, repeat: bool = True) -> Texture:
    """
    Upload a ``(N, N, N, 4) uint8`` volume as a Panda3D 3-D ``Texture``.

    Used for the baked volumetric-cloud noise (``sky.cloud_noise``).  The array
    is page-major — index ``[z, y, x, c]`` (page ``z``, row ``y``, column
    ``x``) — which is Panda3D's native 3-D texture page order, so no transpose
    is needed; only the engine-wide RGBA→BGRA channel swap.  One bulk
    ``set_ram_image`` write, no per-voxel loop.

    Parameters
    ----------
    volume : numpy.ndarray
        Shape ``(N, N, N, 4)``, dtype ``uint8``, RGBA channel order, indexed
        ``[z, y, x, channel]``.
    linear : bool, default True
        Linear filtering (smooth cloud density) vs nearest.
    repeat : bool, default True
        Wrap mode: ``WM_repeat`` (tileable noise — the default for clouds) vs
        ``WM_clamp``.  Tileable bakes wrap seamlessly, so repeat never seams.

    Returns
    -------
    panda3d.core.Texture
        A 3-D ``F_rgba`` texture ready for ``set_shader_input`` (sampler3D).
    """
    if volume.ndim != 4 or volume.shape[3] != 4:
        raise ValueError(f"to_panda_texture_3d expects shape (N, N, N, 4), got {volume.shape}")
    if volume.dtype != np.uint8:
        raise ValueError(f"to_panda_texture_3d expects dtype uint8, got {volume.dtype}")

    d, h, w = volume.shape[:3]
    tex = Texture("volume3d")
    tex.setup_3d_texture(w, h, d, Texture.T_unsigned_byte, Texture.F_rgba)
    bgra = np.ascontiguousarray(volume[..., [2, 1, 0, 3]])  # RGBA -> BGRA
    tex.set_ram_image(bytes(bgra))
    filt = SamplerState.FT_linear if linear else SamplerState.FT_nearest
    tex.set_minfilter(filt)
    tex.set_magfilter(filt)
    wrap = SamplerState.WM_repeat if repeat else SamplerState.WM_clamp
    tex.set_wrap_u(wrap)
    tex.set_wrap_v(wrap)
    tex.set_wrap_w(wrap)
    return tex


def to_data_texture_f32(block: np.ndarray) -> Texture:
    """
    Upload a float32 data block as a 2-D **RGBA32F** texture (texelFetch).

    Used for per-instance transform data (``zones/tree_placement.py::
    instances_data_block``): the array's first axis becomes texture ROWS
    (one row per instance, fetched with ``ivec2(col, gl_InstanceID)``), the
    second axis becomes columns/texels, the last axis the RGBA channels.
    NO vertical flip (row 0 = texel row 0 — instance ids map directly) and
    no filtering/wrap concerns: shaders must read it with ``texelFetch``,
    never ``texture()``.

    Like every Panda3D 4-component RAM image the byte order is **BGRA**
    even at float type, so channels are reordered on the way in — a shader
    ``texelFetch(...).r`` returns ``block[i, col, 0]`` exactly.

    Parameters
    ----------
    block : numpy.ndarray
        Shape ``(rows, cols, 4)``, dtype ``float32``.  For tree instances:
        ``(N, 2, 4)`` — texel 0 ``(x, y, z, yaw)``, texel 1
        ``(scale, phase, tint, variant)``.

    Returns
    -------
    panda3d.core.Texture
        ``F_rgba32`` texture, ``cols × rows`` texels, nearest/clamped.
    """
    if block.ndim != 3 or block.shape[2] != 4:
        raise ValueError(f"to_data_texture_f32 expects shape (rows, cols, 4), got {block.shape}")
    if block.dtype != np.float32:
        raise ValueError(f"to_data_texture_f32 expects dtype float32, got {block.dtype}")

    rows, cols = block.shape[:2]
    tex = Texture("instance_data")
    tex.setup_2d_texture(cols, rows, Texture.T_float, Texture.F_rgba32)
    bgra = np.ascontiguousarray(block[..., [2, 1, 0, 3]])  # RGBA -> BGRA
    tex.set_ram_image(bgra.tobytes())
    tex.set_minfilter(SamplerState.FT_nearest)
    tex.set_magfilter(SamplerState.FT_nearest)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex


def to_field_texture(rgba: np.ndarray) -> Texture:
    """
    Upload a **data field** (not an image) as a 2-D RGBA8 texture.

    Unlike :func:`to_panda_texture` there is NO vertical flip: array row 0
    lands at texture V=0, so shaders can sample with
    ``uv = (world_xy - field_min) / field_size`` and array index
    ``[iy, ix]`` maps directly to world ``(+y, +x)``.  Filtering is nearest
    and wrap is clamped — field texels are exact samples (e.g. the grass
    height field's R channel encodes a height byte; interpolating between a
    height and the 255 no-ground sentinel would invent garbage heights).

    Parameters
    ----------
    rgba : numpy.ndarray
        Shape ``(H, W, 4)``, dtype ``uint8``.  Channel semantics are the
        producer's contract (e.g. ``zones.bake_grass_height_field``).

    Returns
    -------
    panda3d.core.Texture
        Nearest-filtered, edge-clamped 2-D texture.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"to_field_texture expects shape (H, W, 4), got {rgba.shape}")
    if rgba.dtype != np.uint8:
        raise ValueError(f"to_field_texture expects dtype uint8, got {rgba.dtype}")

    H, W = rgba.shape[:2]
    tex = Texture("field")
    tex.setup_2d_texture(W, H, Texture.T_unsigned_byte, Texture.F_rgba)
    bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])  # RGBA -> BGRA
    tex.set_ram_image(bytes(bgra))
    tex.set_minfilter(SamplerState.FT_nearest)
    tex.set_magfilter(SamplerState.FT_nearest)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex
