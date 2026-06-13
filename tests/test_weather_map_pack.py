"""
tests/test_weather_map_pack.py — Headless tests for the M4 GPU weather-map pack.

Covers the ``sky.weather_map_pack.pack_weather_map`` contract:
- byte length (fp16 RGBA, cells²·4·2);
- round-trip: decode the fp16 buffer back to ``[y, x, BGRA]`` and assert every
  logical channel (coverage/density/precip/fog) survives within float16
  tolerance;
- BGRA swizzle: matches ``pack_wind_field``'s convention (B/G/R order, A last),
  and — unlike the wind packer — does NOT transpose (the raster is already
  row-major ``[y, x]``);
- a panda3d-import guard so the packer stays headless (Hard Rule 1).

Headless: no window, no GPU.  Uses a synthetic raster (distinct constant per
channel) so a mis-mapped channel is unambiguous.
"""

from __future__ import annotations

import numpy as np

from fire_engine.sky.weather_map_pack import pack_weather_map


def _synthetic_raster(cells: int = 8) -> np.ndarray:
    """A ``(cells, cells, 4)`` raster with a unique gradient per channel.

    Channel c gets ``c*0.2 + 0.1*(row/cells) + 0.01*(col/cells)`` so every
    (row, col, channel) value is distinct — any transpose/swizzle error shows
    up as a mismatch rather than aliasing to another channel.
    """
    row = (np.arange(cells, dtype=np.float32) / cells)[:, None, None]
    col = (np.arange(cells, dtype=np.float32) / cells)[None, :, None]
    chan = (np.arange(4, dtype=np.float32) * 0.2)[None, None, :]
    return (chan + 0.1 * row + 0.01 * col).astype(np.float32)  # (Y, X, 4) RGBA


def test_byte_length():
    raster = _synthetic_raster(16)
    data = pack_weather_map(raster)
    assert len(data) == 16 * 16 * 4 * 2          # fp16 RGBA


def test_channel_order_and_layout():
    # Decode the fp16 buffer and assert row-major (y, x) + BGRA mapping:
    # decoded[y, x] = (B=precip, G=density, R=coverage, A=fog) for the raster's
    # logical RGBA = (coverage, density, precip, fog) at the SAME [y, x] (no
    # transpose — the raster is already row-major).
    cells = 8
    raster = _synthetic_raster(cells)
    buf = np.frombuffer(pack_weather_map(raster), dtype=np.float16)
    dec = buf.reshape(cells, cells, 4).astype(np.float32)   # [y, x, BGRA]
    for (y, x) in [(0, 0), (3, 5), (7, 7), (1, 6)]:
        cov, den, precip, fog = raster[y, x]                # logical RGBA
        b, g, r, a = dec[y, x]                              # packed BGRA
        np.testing.assert_allclose(b, precip, atol=1e-3)    # B = precip
        np.testing.assert_allclose(g, den, atol=1e-3)       # G = density
        np.testing.assert_allclose(r, cov, atol=1e-3)       # R = coverage
        np.testing.assert_allclose(a, fog, atol=1e-3)       # A = fog


def test_swizzle_matches_pack_wind_field_convention():
    # pack_wind_field swaps RGBA -> BGRA via [..., [2, 1, 0, 3]]; assert the
    # weather packer uses the identical channel permutation (the pinned quirk).
    cells = 4
    raster = _synthetic_raster(cells)
    dec = np.frombuffer(pack_weather_map(raster), dtype=np.float16) \
        .reshape(cells, cells, 4).astype(np.float32)
    expected = raster[..., [2, 1, 0, 3]].astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(dec, expected, atol=1e-6)


def test_rejects_bad_shape():
    import pytest
    with pytest.raises(ValueError):
        pack_weather_map(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        pack_weather_map(np.zeros((8, 4, 4), dtype=np.float32))


def test_round_trips_real_raster():
    # End-to-end against a real WeatherMap raster (the production input type).
    from fire_engine.core import EventBus, load_config, set_world_seed
    from fire_engine.weather import WeatherMap, WeatherSystem

    set_world_seed(1337)
    cfg = load_config()
    ws = WeatherSystem(cfg, EventBus())
    wm = WeatherMap(cfg)
    raster = wm.rasterize(ws, center_xy=(0.0, 0.0), t_abs=12 * 3600.0)
    cells = raster.shape[0]
    dec = np.frombuffer(pack_weather_map(raster), dtype=np.float16) \
        .reshape(cells, cells, 4).astype(np.float32)
    # Logical channels survive (BGRA decode → RGBA).  Weather values are in
    # [0, 1] so float16 abs error is ~1e-3.
    np.testing.assert_allclose(dec[..., 2], raster[..., 0], atol=2e-3)  # coverage
    np.testing.assert_allclose(dec[..., 1], raster[..., 1], atol=2e-3)  # density
    np.testing.assert_allclose(dec[..., 0], raster[..., 2], atol=2e-3)  # precip
    np.testing.assert_allclose(dec[..., 3], raster[..., 3], atol=2e-3)  # fog


def test_no_panda3d_import():
    # The packer must stay headless (Hard Rule 1): AST-scan its source for any
    # panda3d/direct import (mirrors test_wind's wind-package guard — robust to
    # panda3d already being in sys.modules from a sibling test).
    import ast
    from pathlib import Path

    src = (Path(__file__).parent.parent / "fire_engine" / "sky"
           / "weather_map_pack.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name.split(".")[0] in ("panda3d", "direct"):
                offenders.append(f"import {name}")
    assert not offenders, f"panda3d leaked into weather_map_pack: {offenders}"
