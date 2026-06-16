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
    LPoint2f,
    LPoint3f,
    LVecBase2f,
    Shader,
    Texture,
)

from fire_engine.core.log import get_logger
from fire_engine.render import post_shaders

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

    def __init__(self, base: Any, config: Config) -> None:
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
        # Keep refs so the render-target textures aren't GC'd.
        self._bloom_textures: list = []
        self.bloom_tex: Texture | None = None
        self.flare_tex: Texture | None = None
        self.godray_tex: Texture | None = None
        self._composite_quad = None
        self._godray_quad = None

        if not self.enabled:
            _log.info(
                "Post-processing disabled (gfx_post_process=false) — "
                "surface shaders tonemap internally (legacy path)."
            )
            return

        try:
            self._build()
        except Exception as exc:
            _log.warning(
                "Post-processing setup failed (%s); falling back to in-shader tonemapping.", exc
            )
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

        quad = manager.renderSceneInto(colortex=color_tex, depthtex=depth_tex, fbprops=fbp)
        if quad is None:
            raise RuntimeError("FilterManager.renderSceneInto returned no quad")

        # A 1x1 black texture stands in for any disabled effect buffer so the
        # composite's samplers are always validly bound.
        dummy = Texture("fx_dummy")
        dummy.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_rgba)
        dummy.set_clear_color((0.0, 0.0, 0.0, 1.0))
        self._bloom_dummy = dummy

        self._manager = manager
        self.hdr_color_tex = color_tex
        self.depth_tex = depth_tex
        self._final_quad = quad

        # Effect passes read the HDR scene and store result textures; the
        # composite is built LAST so it runs after them (correct sort order).
        self._build_bloom(color_tex)
        self._build_flare(color_tex)
        self._build_godrays(color_tex)
        self._build_composite(color_tex, quad)

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
        down_sh = Shader.make(
            Shader.SL_GLSL,
            vertex=post_shaders.POST_FULLSCREEN_VERTEX,
            fragment=post_shaders.BLOOM_DOWN_FRAGMENT,
        )
        up_sh = Shader.make(
            Shader.SL_GLSL,
            vertex=post_shaders.POST_FULLSCREEN_VERTEX,
            fragment=post_shaders.BLOOM_UP_FRAGMENT,
        )
        threshold = float(getattr(cfg, "gfx_bloom_threshold", 1.0))
        knee = float(getattr(cfg, "gfx_bloom_knee", 0.5))

        # Downsample chain: div 2, 4, 8, … (each level halves resolution).
        down_tex: list[Texture] = []
        src = hdr_tex
        for i in range(mips):
            tex = Texture(f"bloom_down_{i}")
            quad = self._manager.renderQuadInto(
                f"bloom_down_{i}", div=2 ** (i + 1), colortex=tex, fbprops=fbp
            )
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
            quad = self._manager.renderQuadInto(
                f"bloom_up_{i}", div=2 ** (i + 1), colortex=tex, fbprops=fbp
            )
            if quad is None:
                break
            quad.set_shader(up_sh)
            quad.set_shader_input("u_src", up_src)
            quad.set_shader_input("u_add", down_tex[i])
            self._bloom_textures.append(tex)
            up_src = tex

        self.bloom_tex = up_src  # composite binds this in _build_composite

    # Lens-flare geometry tuning (aesthetic constants; strength + threshold are
    # config-exposed via gfx_lens_flare_strength / gfx_lens_flare_threshold).
    _FLARE_GHOSTS = 5  # ghost reflections along the centre axis
    _FLARE_DISPERSAL = 0.32  # ghost spacing
    _FLARE_HALO_WIDTH = 0.45  # halo ring radius (UV)
    _FLARE_CHROMA = 0.012  # chromatic-aberration spread (UV)

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
        shader = Shader.make(
            Shader.SL_GLSL,
            vertex=post_shaders.POST_FULLSCREEN_VERTEX,
            fragment=post_shaders.LENS_FLARE_FRAGMENT,
        )
        quad.set_shader(shader)
        quad.set_shader_input("u_tex", hdr_tex)
        quad.set_shader_input("u_threshold", float(getattr(cfg, "gfx_lens_flare_threshold", 4.0)))
        quad.set_shader_input("u_ghosts", self._FLARE_GHOSTS)
        quad.set_shader_input("u_dispersal", self._FLARE_DISPERSAL)
        quad.set_shader_input("u_halo_width", self._FLARE_HALO_WIDTH)
        quad.set_shader_input("u_chroma", self._FLARE_CHROMA)
        self._bloom_textures.append(tex)  # keep ref alive
        self.flare_tex = tex  # composite binds this

    # God-ray geometry tuning (aesthetic constants; strength is config-exposed
    # via gfx_god_ray_strength, sample count via gfx_god_ray_samples).
    _GODRAY_DENSITY = 0.9  # ray length toward the sun (fraction of screen)
    _GODRAY_DECAY = 0.95  # per-step attenuation
    _GODRAY_THRESHOLD = 3.0  # isolate the sun from bright sky

    def _build_godrays(self, hdr_tex: Texture) -> None:
        """
        Build the screen-space god-ray (crepuscular shaft) pass.

        Half-res radial light-scatter from the sun's screen position (set per
        frame in :meth:`update`); occlusion is automatic (clouds/terrain that
        are dark in the scene block the shafts).  No-op when ``gfx_god_rays``
        is off.
        """
        cfg = self.config
        if not bool(getattr(cfg, "gfx_god_rays", True)):
            return
        tex = Texture("god_rays")
        quad = self._manager.renderQuadInto("god_rays", div=2, colortex=tex)
        if quad is None:
            return
        shader = Shader.make(
            Shader.SL_GLSL,
            vertex=post_shaders.POST_FULLSCREEN_VERTEX,
            fragment=post_shaders.GOD_RAYS_FRAGMENT,
        )
        quad.set_shader(shader)
        quad.set_shader_input("u_tex", hdr_tex)
        quad.set_shader_input("u_samples", int(getattr(cfg, "gfx_god_ray_samples", 32)))
        quad.set_shader_input("u_density", self._GODRAY_DENSITY)
        quad.set_shader_input("u_decay", self._GODRAY_DECAY)
        quad.set_shader_input("u_threshold", self._GODRAY_THRESHOLD)
        # Per-frame sun screen position + activation (see update()).
        quad.set_shader_input("u_sun_screen", LVecBase2f(0.5, 0.5))
        quad.set_shader_input("u_active", 0.0)
        self._bloom_textures.append(tex)
        self.godray_tex = tex
        self._godray_quad = quad

    def _build_composite(self, color_tex: Texture, screen_quad: Any) -> None:
        """
        Build the final composite (scene + bloom + flare + god rays → tonemap),
        with optional FXAA as the very last pass.

        The composite is created AFTER the effect passes so it renders after
        them.  With FXAA on, the composite renders into an LDR buffer and the
        screen quad runs FXAA reading it; otherwise the composite IS the screen
        quad.
        """
        cfg = self.config
        comp_sh = Shader.make(
            Shader.SL_GLSL,
            vertex=post_shaders.POST_FULLSCREEN_VERTEX,
            fragment=post_shaders.COMPOSITE_FRAGMENT,
        )
        comp_quad = screen_quad
        if bool(getattr(cfg, "gfx_fxaa", True)):
            ldr = Texture("composite_ldr")
            cq = self._manager.renderQuadInto("composite", colortex=ldr)
            if cq is not None:
                comp_quad = cq
                fxaa_sh = Shader.make(
                    Shader.SL_GLSL,
                    vertex=post_shaders.POST_FULLSCREEN_VERTEX,
                    fragment=post_shaders.FXAA_FRAGMENT,
                )
                screen_quad.set_shader(fxaa_sh)
                screen_quad.set_shader_input("u_tex", ldr)
                self._bloom_textures.append(ldr)

        comp_quad.set_shader(comp_sh)
        comp_quad.set_shader_input("u_scene", color_tex)
        comp_quad.set_shader_input("u_bloom", self.bloom_tex or self._bloom_dummy)
        comp_quad.set_shader_input(
            "u_bloom_strength",
            float(getattr(cfg, "gfx_bloom_strength", 0.06)) if self.bloom_tex is not None else 0.0,
        )
        comp_quad.set_shader_input("u_flare", self.flare_tex or self._bloom_dummy)
        comp_quad.set_shader_input(
            "u_flare_strength",
            float(getattr(cfg, "gfx_lens_flare_strength", 0.055))
            if self.flare_tex is not None
            else 0.0,
        )
        comp_quad.set_shader_input("u_godray", self.godray_tex or self._bloom_dummy)
        comp_quad.set_shader_input(
            "u_godray_strength",
            float(getattr(cfg, "gfx_god_ray_strength", 0.4))
            if self.godray_tex is not None
            else 0.0,
        )
        comp_quad.set_shader_input(
            "u_hue_preserve", float(getattr(cfg, "gfx_tonemap_hue_preserve", 0.8))
        )
        self._composite_quad = comp_quad

    def _log_buffer_props(self, manager: Any) -> None:
        """Log what the GPU actually granted (esp. whether float survived)."""
        try:
            buf = manager.buffers[-1] if manager.buffers else None
            props = buf.get_fb_properties() if buf is not None else None
            if props is not None:
                _log.info(
                    "HDR scene buffer: %s (float_color=%s, samples=%d)",
                    props,
                    props.get_float_color(),
                    props.get_multisamples(),
                )
                if (
                    str(getattr(self.config, "gfx_hdr_format", "rgba16f")) == "rgba16f"
                    and not props.get_float_color()
                ):
                    _log.warning(
                        "Requested an RGBA16F float buffer but the GPU "
                        "gave a fixed-point one — HDR range will clip. "
                        'Set [graphics] gfx_hdr_format = "rgba8" to '
                        "silence this, or use a different GPU."
                    )
        except Exception as exc:
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

        Auto-exposure is applied inside the surface shaders (so it scales bloom
        too) and bloom/flare are screen-space self-contained, so the only
        per-frame work is feeding the god-ray pass the sun's screen position.
        """
        if not self.enabled or self._godray_quad is None:
            return
        self._update_godray_sun()

    def _update_godray_sun(self) -> None:
        """Project the sun to screen space for the god-ray radial scatter."""
        base = self.base
        sky = getattr(base, "sky_system", None)
        st = getattr(sky, "state", None) if sky is not None else None
        quad = self._godray_quad
        if st is None or float(st.sun_dir.z) <= 0.02:  # below horizon → off
            quad.set_shader_input("u_active", 0.0)
            return
        sun = st.sun_dir
        cam_pos = base.camera.get_pos(base.render)
        far_pt = LPoint3f(
            cam_pos.x + float(sun.x) * 1.0e6,
            cam_pos.y + float(sun.y) * 1.0e6,
            cam_pos.z + float(sun.z) * 1.0e6,
        )
        rel = base.cam.get_relative_point(base.render, far_pt)
        ndc = LPoint2f()
        if not base.camLens.project(rel, ndc):
            quad.set_shader_input("u_active", 0.0)
            return
        u = ndc.x * 0.5 + 0.5
        v = ndc.y * 0.5 + 0.5
        # Allow a margin so shafts from a just-off-screen sun still stream in.
        on = (-0.35 <= u <= 1.35) and (-0.35 <= v <= 1.35)
        quad.set_shader_input("u_sun_screen", LVecBase2f(float(u), float(v)))
        quad.set_shader_input("u_active", 1.0 if on else 0.0)
