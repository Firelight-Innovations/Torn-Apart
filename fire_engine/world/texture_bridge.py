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
    from fire_engine.world.texture_bridge import to_panda_texture

    set_world_seed(42)
    arr = get("wasteland_ground")               # (256, 256, 4) uint8
    tex = to_panda_texture(arr)                 # panda3d.core.Texture
    # Attach to geometry:
    #   node_path.set_texture(tex)
"""

from __future__ import annotations

import numpy as np
from panda3d.core import Texture, SamplerState  # type: ignore[import]

__all__ = ["to_panda_texture", "to_field_texture", "to_panda_cubemap"]


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
        from fire_engine.world.texture_bridge import to_panda_texture

        # Solid red 4×4 texture
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[..., 0] = 255   # R
        arr[..., 3] = 255   # A
        tex = to_panda_texture(arr)
        assert tex.get_x_size() == 4
        assert tex.get_y_size() == 4
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(
            f"to_panda_texture expects shape (H, W, 4), got {rgba.shape}"
        )
    if rgba.dtype != np.uint8:
        raise ValueError(
            f"to_panda_texture expects dtype uint8, got {rgba.dtype}"
        )

    H, W = rgba.shape[:2]

    tex = Texture()
    tex.setup_2d_texture(W, H, Texture.T_unsigned_byte, Texture.F_rgba)

    # Panda3D UV origin is bottom-left (OpenGL convention); flip vertically
    # so the image appears right-side-up.  Panda3D RAM images for F_rgba are
    # stored **BGRA** byte order (its native component order), so reorder the
    # channels R<->B on the way in or every texture renders blue-for-brown.
    flipped = rgba[::-1]                       # vertical flip (view)
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
        Linear-filtered cube map ready for ``set_shader_input``.
    """
    if faces.ndim != 4 or faces.shape[0] != 6 or faces.shape[3] != 4 \
            or faces.shape[1] != faces.shape[2]:
        raise ValueError(
            f"to_panda_cubemap expects shape (6, S, S, 4), got {faces.shape}")
    if faces.dtype != np.uint8:
        raise ValueError(
            f"to_panda_cubemap expects dtype uint8, got {faces.dtype}")

    size = faces.shape[1]
    tex = Texture("cubemap")
    tex.setup_cube_map(size, Texture.T_unsigned_byte, Texture.F_rgba)
    bgra = np.ascontiguousarray(faces[..., [2, 1, 0, 3]])   # RGBA -> BGRA
    tex.set_ram_image(bytes(bgra))
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
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
        raise ValueError(
            f"to_field_texture expects shape (H, W, 4), got {rgba.shape}"
        )
    if rgba.dtype != np.uint8:
        raise ValueError(
            f"to_field_texture expects dtype uint8, got {rgba.dtype}"
        )

    H, W = rgba.shape[:2]
    tex = Texture("field")
    tex.setup_2d_texture(W, H, Texture.T_unsigned_byte, Texture.F_rgba)
    bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])   # RGBA -> BGRA
    tex.set_ram_image(bytes(bgra))
    tex.set_minfilter(SamplerState.FT_nearest)
    tex.set_magfilter(SamplerState.FT_nearest)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex
