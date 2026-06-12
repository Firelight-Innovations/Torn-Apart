"""
world/post_process.py — HDR offscreen render target + post-processing chain.

The scene (terrain, sky dome, clouds, grass) renders into a linear **RGBA16F
float** buffer instead of straight to the window, then a chain of fullscreen
passes turns that HDR signal into the final image.  This is what lets bright
things (the sun disc, the grazing-sunrise horizon, emissive surfaces) keep
values far above 1.0 so they can bloom, flare, and tonemap correctly — the old
pipeline tonemapped + clamped inside every surface shader, destroying that range
before anything could use it.

Panda3D wiring
--------------
Built on ``direct.filter.FilterManager``: ``renderSceneInto`` redirects the main
camera's scene into our textures and hands back a screen-spanning card; we set
the composite shader on that card.  FilterManager installs **no frame task**, so
it composes cleanly with :class:`world.app.App`'s custom frame loop.  It also
auto-resizes the buffers when the window resizes.

The object shaders are told to emit linear HDR via a single ``u_hdr_output``
shader-input set on ``render`` (every surface shader inherits it).  When
post-processing is disabled (``gfx_post_process = false`` / preset ``"off"``, or
the GPU can't allocate the buffer) that flag stays 0.0 and the shaders tonemap
internally exactly as before — the legacy path is the safety net.

Phase status: this is the scaffold — scene buffer + a passthrough composite
(ACES tonemap + sRGB gamma, matching the old per-shader output).  Bloom, lens
flare, god rays and FXAA insert into the chain in later phases via
:meth:`insert_pass_before_composite`.

panda3d imports are allowed here (``world/`` per ARCHITECTURE §3).

Example
-------
    post = PostProcessPipeline(app, cfg)      # after lighting + sky are wired
    # ... each frame, after the lighting pipeline updates:
    post.update(app.lighting_pipeline)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panda3d.core import (  # type: ignore[import]
    FrameBufferProperties,
    Shader,
    Texture,
)

from fire_engine.core.log import get_logger
from fire_engine.world import post_shaders

if TYPE_CHECKING:
    from fire_engine.core.config import Config

_log = get_logger("post_process")


class PostProcessPipeline:
    """
    Owns the HDR scene buffer and the fullscreen post-processing passes.

    Parameters
    ----------
    base : world.app.App (ShowBase)
        The running application; provides ``win``, ``cam``, ``render``.
    config : Config
        Engine config; reads the ``gfx_*`` graphics-quality knobs.

    Attributes
    ----------
    enabled : bool
        False when post-processing is configured off, or when the GPU could not
        allocate the offscreen buffer (the engine falls back to in-shader
        tonemapping — see module docstring).
    hdr_color_tex : Texture | None
        The linear-HDR scene color buffer (RGBA16F).  Read by downstream passes
        (bloom bright-pass, lens flare).
    depth_tex : Texture | None
        The scene depth buffer.  Used by the lens-flare pass to test whether the
        sun is occluded by terrain.
    """

    def __init__(self, base: Any, config: "Config") -> None:
        self.base = base
        self.config = config
        self.enabled: bool = bool(getattr(config, "gfx_post_process", True))

        self._manager = None
        self.hdr_color_tex: Texture | None = None
        self.depth_tex: Texture | None = None
        self._final_quad = None
        # Passes that should run between the scene render and the composite
        # (bloom/flare/god-rays insert here).  Each entry is a NodePath card.
        self._mid_passes: list = []
        self._bloom_dummy: Texture | None = None
        # Bloom: keep refs so the render-target textures aren't GC'd.
        self._bloom_textures: list = []
        self.bloom_tex: Texture | None = None
        self.flare_tex: Texture | None = None

        if not self.enabled:
            _log.info("Post-processing disabled (gfx_post_process=false) — "
                      "surface shaders tonemap internally (legacy path).")
            return

        try:
            self._build()
        except Exception as exc:  # noqa: BLE001 — never fatal; fall back to legacy
            _log.warning("Post-processing setup failed (%s); falling back to "
                         "in-shader tonemapping.", exc)
            self.enabled = False
            self._set_hdr_output(False)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        from direct.filter.FilterManager import FilterManager  # type: ignore[import]

        manager = FilterManager(self.base.win, self.base.cam)

        color_tex = Texture("hdr_scene")
        depth_tex = Texture("scene_depth")

        fbp = FrameBufferProperties()
        if str(getattr(self.config, "gfx_hdr_format", "rgba16f")) == "rgba16f":
            fbp.set_float_color(True)
            fbp.set_rgba_bits(16, 16, 16, 16)
        else:
            fbp.set_rgba_bits(8, 8, 8, 8)
        fbp.set_depth_bits(24)
        # Preserve geometry MSAA inside the offscreen buffer (FXAA, added later,
        # covers post-resolve edges; until then this keeps edges as crisp as the
        # legacy window-MSAA path).
        msaa = int(getattr(self.config, "msaa_samples", 0))
        if msaa > 0:
            fbp.set_multisamples(msaa)

        quad = manager.renderSceneInto(colortex=color_tex, depthtex=depth_tex,
                                       fbprops=fbp)
        if quad is None:
            raise RuntimeError("FilterManager.renderSceneInto returned no quad")

        # A 1x1 black texture stands in for the (not-yet-built) bloom buffer so
        # the composite shader always has a valid sampler bound.
        dummy = Texture("bloom_dummy")
        dummy.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_rgba)
        dummy.set_clear_color((0.0, 0.0, 0.0, 1.0))
        self._bloom_dummy = dummy

        shader = Shader.make(Shader.SL_GLSL,
                             vertex=post_shaders.POST_FULLSCREEN_VERTEX,
                             fragment=post_shaders.COMPOSITE_FRAGMENT)
        quad.set_shader(shader)
        quad.set_shader_input("u_scene", color_tex)
        quad.set_shader_input("u_bloom", dummy)
        quad.set_shader_input("u_bloom_strength", 0.0)
        quad.set_shader_input("u_flare", dummy)
        quad.set_shader_input("u_flare_strength", 0.0)

        self._manager = manager
        self.hdr_color_tex = color_tex
        self.depth_tex = depth_tex
        self._final_quad = quad

        # Bloom pyramid (reads the HDR scene buffer, feeds the composite).
        self._build_bloom(color_tex)

        # Lens flare (image-based ghosts + halo from the bright scene).
        self._build_flare(color_tex)

        # Flip every surface shader to linear-HDR output.
        self._set_hdr_output(True)

        self._log_buffer_props(manager)

    def _build_bloom(self, hdr_tex: Texture) -> None:
        """
        Build the downsample/upsample bloom pyramid feeding the composite.

        Call-of-Duty-style: a soft-knee bright-pass + Karis-averaged 13-tap
        downsample chain (``gfx_bloom_mips`` halvings), then a 3x3-tent upsample
        chain that progressively adds each level back — a smooth, wide,
        firefly-free glow.  All buffers are half-res-and-down RGBA16F, so the
        cost is a small fraction of a full-res pass (iGPU-friendly).  No-op when
        ``gfx_bloom`` is off (composite keeps the black dummy, strength 0).
        """
        cfg = self.config
        if not bool(getattr(cfg, "gfx_bloom", True)):
            return

        mips = max(1, int(getattr(cfg, "gfx_bloom_mips", 5)))
        fbp = FrameBufferProperties()
        fbp.set_float_color(True)
        fbp.set_rgba_bits(16, 16, 16, 16)
        down_sh = Shader.make(Shader.SL_GLSL,
                              vertex=post_shaders.POST_FULLSCREEN_VERTEX,
                              fragment=post_shaders.BLOOM_DOWN_FRAGMENT)
        up_sh = Shader.make(Shader.SL_GLSL,
                            vertex=post_shaders.POST_FULLSCREEN_VERTEX,
                            fragment=post_shaders.BLOOM_UP_FRAGMENT)
        threshold = float(getattr(cfg, "gfx_bloom_threshold", 1.0))
        knee = float(getattr(cfg, "gfx_bloom_knee", 0.5))

        # Downsample chain: div 2, 4, 8, … (each level halves resolution).
        down_tex: list[Texture] = []
        src = hdr_tex
        for i in range(mips):
            tex = Texture(f"bloom_down_{i}")
            quad = self._manager.renderQuadInto(f"bloom_down_{i}",
                                                div=2 ** (i + 1),
                                                colortex=tex, fbprops=fbp)
            if quad is None:
                break
            quad.set_shader(down_sh)
            quad.set_shader_input("u_tex", src)
            quad.set_shader_input("u_prefilter", 1.0 if i == 0 else 0.0)
            quad.set_shader_input("u_threshold", threshold)
            quad.set_shader_input("u_knee", knee)
            down_tex.append(tex)
            src = tex

        if not down_tex:
            return
        self._bloom_textures.extend(down_tex)

        # Upsample chain: from the smallest mip back up to div 2, adding each
        # same-resolution downsample level on the way.
        up_src = down_tex[-1]
        for i in range(len(down_tex) - 2, -1, -1):
            tex = Texture(f"bloom_up_{i}")
            quad = self._manager.renderQuadInto(f"bloom_up_{i}",
                                                div=2 ** (i + 1),
                                                colortex=tex, fbprops=fbp)
            if quad is None:
                break
            quad.set_shader(up_sh)
            quad.set_shader_input("u_src", up_src)
            quad.set_shader_input("u_add", down_tex[i])
            self._bloom_textures.append(tex)
            up_src = tex

        self.bloom_tex = up_src
        self._final_quad.set_shader_input("u_bloom", up_src)
        self._final_quad.set_shader_input(
            "u_bloom_strength", float(getattr(cfg, "gfx_bloom_strength", 0.06)))

    # Lens-flare tuning (held here, not in config — aesthetic constants).
    _FLARE_THRESHOLD = 4.0     # HDR luminance: isolate the sun, not bright sky
    _FLARE_GHOSTS = 5          # ghost reflections along the centre axis
    _FLARE_DISPERSAL = 0.32    # ghost spacing
    _FLARE_HALO_WIDTH = 0.45   # halo ring radius (UV)
    _FLARE_CHROMA = 0.012      # chromatic-aberration spread (UV)
    _FLARE_STRENGTH = 0.22     # contribution added at composite

    def _build_flare(self, hdr_tex: Texture) -> None:
        """
        Build the image-based lens-flare pass (ghosts + halo) feeding composite.

        Reads the HDR scene at quarter-res, isolates the sun (a high HDR
        threshold), and rebuilds ghost reflections (mirrored through the screen
        centre, with chromatic fringing) + a halo ring.  Occlusion is automatic
        — an occluded sun isn't bright in the buffer, so the flare disappears.
        No-op when ``gfx_lens_flare`` is off.
        """
        cfg = self.config
        if not bool(getattr(cfg, "gfx_lens_flare", True)):
            return
        tex = Texture("lens_flare")
        quad = self._manager.renderQuadInto("lens_flare", div=4, colortex=tex)
        if quad is None:
            return
        shader = Shader.make(Shader.SL_GLSL,
                             vertex=post_shaders.POST_FULLSCREEN_VERTEX,
                             fragment=post_shaders.LENS_FLARE_FRAGMENT)
        quad.set_shader(shader)
        quad.set_shader_input("u_tex", hdr_tex)
        quad.set_shader_input("u_threshold", self._FLARE_THRESHOLD)
        quad.set_shader_input("u_ghosts", self._FLARE_GHOSTS)
        quad.set_shader_input("u_dispersal", self._FLARE_DISPERSAL)
        quad.set_shader_input("u_halo_width", self._FLARE_HALO_WIDTH)
        quad.set_shader_input("u_chroma", self._FLARE_CHROMA)
        self._bloom_textures.append(tex)   # keep ref alive
        self.flare_tex = tex
        self._final_quad.set_shader_input("u_flare", tex)
        self._final_quad.set_shader_input("u_flare_strength", self._FLARE_STRENGTH)

    def _log_buffer_props(self, manager: Any) -> None:
        """Log what the GPU actually granted (esp. whether float survived)."""
        try:
            buf = manager.buffers[-1] if manager.buffers else None
            props = buf.get_fb_properties() if buf is not None else None
            if props is not None:
                _log.info("HDR scene buffer: %s (float_color=%s, samples=%d)",
                          props, props.get_float_color(),
                          props.get_multisamples())
                if (str(getattr(self.config, "gfx_hdr_format", "rgba16f"))
                        == "rgba16f" and not props.get_float_color()):
                    _log.warning("Requested an RGBA16F float buffer but the GPU "
                                 "gave a fixed-point one — HDR range will clip. "
                                 "Set [graphics] gfx_hdr_format = \"rgba8\" to "
                                 "silence this, or use a different GPU.")
        except Exception as exc:  # noqa: BLE001 — diagnostics only
            _log.debug("Buffer property query failed: %s", exc)

    def _set_hdr_output(self, on: bool) -> None:
        """Set the ``u_hdr_output`` flag inherited by every surface shader."""
        self.base.render.set_shader_input("u_hdr_output", 1.0 if on else 0.0)

    # ------------------------------------------------------------------
    # Extension seam (bloom / lens flare / god rays plug in here)
    # ------------------------------------------------------------------

    def insert_pass_before_composite(self, card: Any) -> None:
        """
        Register a fullscreen pass card to run before the final composite.

        Reserved for the bloom / lens-flare / god-ray phases.  The card is a
        NodePath produced by ``FilterManager.renderQuadInto`` (or set on the
        final quad); ordering is insertion order.  Kept as a simple list so the
        later phases own their own wiring without reshaping this scaffold.
        """
        self._mid_passes.append(card)

    @property
    def final_quad(self) -> Any:
        """The screen-spanning composite card (set shader inputs on it)."""
        return self._final_quad

    @property
    def manager(self) -> Any:
        """The underlying FilterManager (for renderQuadInto in later phases)."""
        return self._manager

    # ------------------------------------------------------------------
    # Per-frame
    # ------------------------------------------------------------------

    def update(self, lighting_pipeline: Any = None) -> None:
        """
        Per-frame refresh of post-process inputs.

        No-op for the scaffold: auto-exposure is applied inside the surface
        shaders (so it also scales bloom), and the composite needs nothing
        per-frame yet.  Bloom strength and the lens-flare sun position are
        pushed here once those phases land.
        """
        if not self.enabled:
            return
