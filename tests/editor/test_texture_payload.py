"""TEXTURE payload + ground-LUT tests (EDITOR_PRD textured terrain).

Covers the codec round-trip, deterministic per-seed ``ground_seed`` (so the
editor viewport matches the game), and that ``world.ground_lut`` ships exactly
one TEXTURE frame whose bytes decode back to ``EditorSession.ground_lut()``.
"""
from __future__ import annotations

import asyncio

import numpy as np

from fire_engine.core import load_config

from fire_editor import Daemon, EditorSession
from fire_editor._generated import SchemaId
from fire_editor.binary import decode_frame
from fire_editor.texturecodec import decode_texture_payload, encode_texture_payload


def _run(coro):
    return asyncio.run(coro)


class TestTextureCodec:
    def test_roundtrip(self):
        rgba = np.arange(3 * 256 * 4, dtype=np.uint8).reshape(3, 256, 4)
        out = decode_texture_payload(encode_texture_payload(rgba))
        assert out["width"] == 256 and out["height"] == 3
        np.testing.assert_array_equal(out["rgba"], rgba)

    def test_single_row(self):
        rgba = np.full((1, 16, 4), 200, dtype=np.uint8)
        out = decode_texture_payload(encode_texture_payload(rgba))
        assert out["width"] == 16 and out["height"] == 1
        np.testing.assert_array_equal(out["rgba"], rgba)


class TestGroundSeedDeterminism:
    def test_same_seed_same_ground_seed(self):
        cfg = load_config()
        a = EditorSession.from_seed(4242, cfg)
        b = EditorSession.from_seed(4242, cfg)
        assert a.ground_seed == b.ground_seed

    def test_different_seed_different_ground_seed(self):
        cfg = load_config()
        a = EditorSession.from_seed(1, cfg)
        b = EditorSession.from_seed(2, cfg)
        assert a.ground_seed != b.ground_seed


class TestGroundLutFrame:
    def test_lut_shape_and_palette(self):
        cfg = load_config()
        s = EditorSession.from_seed(1337, cfg)
        lut = s.ground_lut()
        assert lut.dtype == np.uint8
        assert lut.ndim == 3 and lut.shape[1] == 256 and lut.shape[2] == 4
        assert s.ground_lut() is lut  # cached

    def test_ground_lut_broadcasts_one_texture_frame(self):
        async def scenario():
            daemon = Daemon()
            daemon.session = EditorSession.from_seed(1337, load_config())
            sent: list[bytes] = []

            async def capture(frame: bytes) -> None:
                sent.append(frame)

            daemon.server.broadcast_binary = capture  # type: ignore[assignment]

            res = await daemon.chunks.ground_lut({})
            assert res["ok"] and res["width"] == 256
            assert len(sent) == 1, "exactly one TEXTURE frame per ground_lut call"
            schema_id, payload_id, payload = decode_frame(sent[0])
            assert schema_id == SchemaId.TEXTURE and payload_id == res["payload_id"]
            decoded = decode_texture_payload(payload)
            np.testing.assert_array_equal(
                decoded["rgba"], daemon.session.ground_lut()
            )

        _run(scenario())
