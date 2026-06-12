"""
tests/test_lit_surface.py — the shared lit-surface lighting contract.

Headless (no panda3d): shaders are composed through ``load_glsl`` directly
(the thin ``world/*_shaders.py`` re-export modules may import panda3d, so we
anchor on the world package path instead).  Pins:

1. Every lit fragment shader gets exactly ONE canonical ``sampleCascades``
   (the 4-arg cross-fade version) — i.e. it includes lit_surface.glsl and
   does not hand-copy/drift its own.
2. Every lit fragment carries the ``u_hdr_output`` gate (the missing gate on
   trees/flora was the double-tonemap "washed-out foliage" bug).
3. Raw sidecar files define none of the library functions locally.
4. Library tuning constants (cascade fade bands, march step counts) are
   pinned so silent edits fail loudly.
5. The refinement march + occluder-box arrays only compile under
   ``#define LIT_REFINE`` (cheap shaders must not pay its uniform budget).
6. Sampler budget: GL 3.3 guarantees only 16 fragment samplers; composed
   terrain sits at exactly 15 — adding a sampler to the library breaks iGPUs.
"""

from __future__ import annotations

import pathlib
import re

from fire_engine.core.shader_source import load_glsl

_REPO = pathlib.Path(__file__).resolve().parents[1]
_WORLD_ANCHOR = str(_REPO / "fire_engine" / "world" / "fake.py")
_SHADER_DIR = _REPO / "fire_engine" / "world" / "shaders"

# Every fragment shader that draws a lit surface.  Phase-3 foliage rewrites
# extend this list (grass.frag, flora.frag, tree.frag, mote_leaf.frag).
_LIT_FRAGMENTS = [
    "terrain.frag",
]

# Functions owned by lit_surface.glsl — consumers must never define locally.
_LIBRARY_FUNCTIONS = [
    "void sampleCascades(",
    "void sampleCascadeAt(",
    "vec3 c_uv(",
    "bool inBox(",
    "float boxWeight(",
    "float occCell(",
    "float boxVis(",
    "float refineVis(",
    "float refineVisSoft(",
    "vec3 acesTonemap(",
    "float litQuantSize(",
    "vec3 litQuantPos(",
    "float litAo(",
    "vec3 litFog(",
    "vec3 litFinish(",
]

_CANONICAL_SAMPLE = (
    "void sampleCascades(vec3 wp, out vec3 radiance, out vec3 vis, "
    "out float occ)"
)


def _composed(name: str) -> str:
    return load_glsl(_WORLD_ANCHOR, name)


def _raw(name: str) -> str:
    return (_SHADER_DIR / name).read_text(encoding="utf-8")


class TestLitFragments:
    def test_every_lit_fragment_has_canonical_sample_cascades(self):
        for name in _LIT_FRAGMENTS:
            text = _composed(name)
            assert text.count(_CANONICAL_SAMPLE) == 1, name

    def test_every_lit_fragment_has_hdr_gate(self):
        for name in _LIT_FRAGMENTS:
            assert "u_hdr_output > 0.5" in _composed(name), name

    def test_no_local_redefinitions(self):
        for name in _LIT_FRAGMENTS:
            raw = _raw(name)
            for fn in _LIBRARY_FUNCTIONS:
                assert fn not in raw, f"{name} locally defines {fn}"

    def test_every_lit_fragment_includes_the_library(self):
        for name in _LIT_FRAGMENTS:
            assert '//#include "lit_surface.glsl"' in _raw(name), name


class TestLibraryPins:
    def test_library_constants_pinned(self):
        lib = _raw("lit_surface.glsl")
        # Cascade cross-fade bands (fraction of box half-extent).
        for band in ("0.14", "0.12", "0.10"):
            assert f", {band});" in lib, f"fade band {band} missing"
        # Refinement march step counts per cascade.
        for steps in ("i < 28", "i < 24", "i < 12"):
            assert steps in lib, f"march loop '{steps}' missing"

    def test_library_has_no_version_line(self):
        # The library is include-only; a #version line would break consumers.
        # (The header comment shows one in its usage example — skip comments.)
        for line in _raw("lit_surface.glsl").splitlines():
            assert not line.lstrip().startswith("#version")

    def test_refine_block_guarded(self):
        lib = _raw("lit_surface.glsl")
        guard = lib.index("#ifdef LIT_REFINE")
        end = lib.index("#endif", guard)
        for symbol in ("uniform int   u_num_boxes",
                       "uniform vec4  u_box_min",
                       "uniform vec4  u_box_max",
                       "uniform float u_penumbra_tan",
                       "uniform float u_refine",
                       "float occCell(", "float boxVis(",
                       "float refineVis(", "float refineVisSoft("):
            pos = lib.index(symbol)
            assert guard < pos < end, f"{symbol} outside LIT_REFINE guard"

    def test_penumbra_band_pinned_in_terrain(self):
        # The refinement gate band (only refine inside the trilinear
        # penumbra) lives at the call sites, not the library.
        raw = _raw("terrain.frag")
        assert "vis.r > 0.02 && vis.r < 0.98" in raw
        assert "u_refine > 0.5" in raw


class TestSamplerBudget:
    @staticmethod
    def _sampler_count(text: str) -> int:
        return len(re.findall(r"^\s*uniform\s+sampler\dD\s", text, re.M))

    def test_terrain_sampler_budget(self):
        # 3 p3d textures + ground LUT + c0 emission + 9 cascade + 1 fog = 15.
        # GL 3.3 guarantees exactly 16 — one slot of headroom, no more.
        assert self._sampler_count(_composed("terrain.frag")) == 15

    def test_all_lit_fragments_within_gl33_minimum(self):
        for name in _LIT_FRAGMENTS:
            assert self._sampler_count(_composed(name)) <= 16, name
