"""
The _Cascade private class for GpuLightingPipeline.

Extracted from ``fire_engine.lighting.gpu`` to keep that module under the
500-line limit.  Re-exported from ``gpu.py`` so internal references remain
valid.

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

from panda3d.core import NodePath, Shader, Texture

from fire_engine.lighting.volume import EMISSION_SCALE, VolumeWindow

__all__ = ["_Cascade", "make_volume_texture"]


def make_volume_texture(name: str, cells: int, *, hdr: bool, linear: bool) -> Texture:
    """
    Allocate one cascade 3-D texture.

    Parameters
    ----------
    name : str
        Debug name.
    cells : int
        Texels per axis.
    hdr : bool
        True → ``rgba16f`` (GPU-written radiance), False → ``rgba8``
        (CPU-uploaded geometry/emission or GPU-written visibility).
    linear : bool
        Trilinear filtering (radiance/visibility sampling) vs nearest.

    Docs: docs/systems/lighting.md
    """
    from panda3d.core import SamplerState

    tex = Texture(name)
    if hdr:
        tex.setup_3d_texture(cells, cells, cells, Texture.T_float, Texture.F_rgba16)
        tex.set_keep_ram_image(False)
    else:
        tex.setup_3d_texture(cells, cells, cells, Texture.T_unsigned_byte, Texture.F_rgba8)
    tex.set_clear_color((0.0, 0.0, 0.0, 0.0))
    filt = SamplerState.FT_linear if linear else SamplerState.FT_nearest
    tex.set_minfilter(filt)
    tex.set_magfilter(filt)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    tex.set_wrap_w(SamplerState.WM_clamp)
    return tex


class _Cascade:
    """One radiance cascade: window + textures + compute node paths."""

    def __init__(
        self,
        index: int,
        cells: int,
        cell_m: float,
        inject_shader: Shader,
        gather_shader: Shader,
        smooth_shader: Shader,
        shift_shader: Shader,
        bounce: float,
        gi_rays: int,
        gi_steps: int,
        *,
        margin_cells: int = 8,
    ) -> None:
        self.index = index
        self.window = VolumeWindow(cells=cells, cell_m=cell_m, margin_cells=margin_cells)
        self.cells = cells
        self.cell_m = cell_m

        self.geom = make_volume_texture(f"lit_geom_{index}", cells, hdr=False, linear=True)
        self.emis = make_volume_texture(f"lit_emis_{index}", cells, hdr=False, linear=True)
        self.vis = make_volume_texture(f"lit_vis_{index}", cells, hdr=True, linear=True)
        # Surface-radiosity proxies (celestial first bounce + emissive leak —
        # no skylight, no dynamic lights), written by INJECT, gathered off
        # surfaces by GATHER.
        self.source = make_volume_texture(f"lit_source_{index}", cells, hdr=True, linear=True)
        # Dynamic-light direct radiance in air; added once per cell by GATHER
        # (own-cell term), never re-gathered off surfaces.
        self.lit = make_volume_texture(f"lit_dyn_{index}", cells, hdr=True, linear=True)
        self.radiance = [
            make_volume_texture(f"lit_rad_{index}_a", cells, hdr=True, linear=True),
            make_volume_texture(f"lit_rad_{index}_b", cells, hdr=True, linear=True),
        ]
        self.ping = 0  # index of the radiance texture holding current light
        self.needs_inject = True  # re-run the injection pass next update
        # Async assembly bookkeeping (main thread only): one job in flight per
        # cascade at a time; ``window.origin_cell`` is the COMMITTED origin (the
        # one the uploaded geom + shader uniforms use) and only advances when a
        # result lands.  ``_pending_seq`` matches the in-flight job.
        self._assembly_inflight = False
        self._pending_seq = -1

        # Injection node (inputs refreshed before each dirty dispatch).
        self.inject_np = NodePath(f"lit_inject_{index}")
        self.inject_np.set_shader(inject_shader)
        self.inject_np.set_shader_input("u_geom", self.geom)
        self.inject_np.set_shader_input("u_emis", self.emis)
        self.inject_np.set_shader_input("u_vis", self.vis)
        self.inject_np.set_shader_input("u_source", self.source)
        self.inject_np.set_shader_input("u_lit", self.lit)
        self.inject_np.set_shader_input("u_cells", cells)
        self.inject_np.set_shader_input("u_emission_scale", float(EMISSION_SCALE))

        # Two pre-bound gather nodes: a→b and b→a (the previous gather feeds
        # the multi-bounce feedback term).  Per-dispatch inputs (sky ambient,
        # window origin) are refreshed by the pipeline before each run.
        self.gather_np: list[NodePath] = []
        for src, dst in ((0, 1), (1, 0)):
            gn = NodePath(f"lit_gather_{index}_{src}{dst}")
            gn.set_shader(gather_shader)
            gn.set_shader_input("u_prev", self.radiance[src])
            gn.set_shader_input("u_next", self.radiance[dst])
            gn.set_shader_input("u_source", self.source)
            gn.set_shader_input("u_lit", self.lit)
            gn.set_shader_input("u_geom", self.geom)
            gn.set_shader_input("u_cells", cells)
            gn.set_shader_input("u_rays", int(gi_rays))
            gn.set_shader_input("u_steps", int(gi_steps))
            gn.set_shader_input("u_bounce", float(bounce))
            gn.set_shader_input("u_cell_m", float(cell_m))
            self.gather_np.append(gn)

        # Two pre-bound smooth nodes (a→b and b→a): air-masked 3³ box filter
        # of the ray-gathered GI component, run after the gather iterations
        # (light_gi_smooth_passes times).  The own-cell contact term
        # (u_source + u_lit) is recomposed crisp; solids are never crossed.
        self.smooth_np: list[NodePath] = []
        for src, dst in ((0, 1), (1, 0)):
            sn = NodePath(f"lit_smooth_{index}_{src}{dst}")
            sn.set_shader(smooth_shader)
            sn.set_shader_input("u_src", self.radiance[src])
            sn.set_shader_input("u_dst", self.radiance[dst])
            sn.set_shader_input("u_geom", self.geom)
            sn.set_shader_input("u_source", self.source)
            sn.set_shader_input("u_lit", self.lit)
            sn.set_shader_input("u_cells", cells)
            self.smooth_np.append(sn)

        # Two pre-bound shift nodes (a→b and b→a): copy the CURRENT radiance
        # (read side = ``ping``) into the other texture offset by the recenter
        # cell delta, then the caller swaps ``ping`` so the same-frame re-gather
        # feedback reads the spatially-aligned field.  Only ``u_shift`` is
        # refreshed per use.
        self.shift_np: list[NodePath] = []
        for src, dst in ((0, 1), (1, 0)):
            sn = NodePath(f"lit_shift_{index}_{src}{dst}")
            sn.set_shader(shift_shader)
            sn.set_shader_input("u_src", self.radiance[src])
            sn.set_shader_input("u_dst", self.radiance[dst])
            sn.set_shader_input("u_cells", cells)
            self.shift_np.append(sn)

    @property
    def radiance_current(self) -> Texture:
        """The radiance texture holding the latest gathered light."""
        return self.radiance[self.ping]

    def origin_m(self) -> tuple[float, float, float]:
        """World min-corner of the window (meters)."""
        return self.window.world_origin_m
