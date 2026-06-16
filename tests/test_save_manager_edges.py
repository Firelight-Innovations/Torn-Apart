"""
tests/test_save_manager_edges.py — Edge and error coverage for SaveManager.

Headless only: no panda3d / fire_engine.world / lighting.gpu.
Pins current behavior as golden-master; does NOT fix bugs.
Complements tests/test_save.py without duplicating its exact cases.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fire_engine.core import Clock, EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.save import Saveable, SaveIncompatibleError, SaveManager
from fire_engine.save.save_manager import (
    _compute_config_digest,
    _decode_delta,
    _decode_value,
    _encode_delta,
    _encode_value,
)

# ---------------------------------------------------------------------------
# Helpers (mirror _make_world pattern from test_save.py)
# ---------------------------------------------------------------------------


def _make_world(seed: int = 1337):
    """Return (cfg, clock, sm) for a lightweight headless world at ``seed``.

    Unlike test_save.py's helper, we skip ChunkManager (panda3d path) and
    register only minimal fake Saveables.  That keeps this module fully headless.
    """
    cfg = load_config()
    cfg = dataclasses.replace(cfg, world_seed=seed)
    set_world_seed(seed)
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    sm = SaveManager(cfg, clock)
    return cfg, clock, sm


class _FakeSaveable:
    """Minimal Saveable that records every call made to it."""

    def __init__(self, key: str, delta: dict | None = None):
        self.save_key = key
        self._delta = delta or {}
        self.applied: list[dict] = []  # every apply_delta call recorded here

    def get_delta(self) -> dict:
        return dict(self._delta)

    def apply_delta(self, delta: dict) -> None:
        self.applied.append(delta)


class _ExplodingApply:
    """Saveable whose apply_delta always raises RuntimeError."""

    save_key: str = "exploding"

    def get_delta(self) -> dict:
        return {"x": 1}

    def apply_delta(self, delta: dict) -> None:
        raise RuntimeError("intentional apply_delta failure")


# ---------------------------------------------------------------------------
# register() — TypeError on non-Saveable
# ---------------------------------------------------------------------------


class TestRegisterTypeError:
    def test_plain_object_raises_type_error(self):
        """Registering a plain object (no protocol) raises TypeError."""
        _, _, sm = _make_world()

        class Alien:
            pass

        with pytest.raises(TypeError):
            sm.register(Alien())

    def test_partial_protocol_missing_apply_delta_raises(self):
        """An object with save_key + get_delta but no apply_delta raises TypeError."""
        _, _, sm = _make_world()

        class Partial:
            save_key = "p"

            def get_delta(self):
                return {}

        with pytest.raises(TypeError):
            sm.register(Partial())

    def test_partial_protocol_missing_get_delta_raises(self):
        """An object with save_key + apply_delta but no get_delta raises TypeError."""
        _, _, sm = _make_world()

        class Partial:
            save_key = "p"

            def apply_delta(self, d):
                pass

        with pytest.raises(TypeError):
            sm.register(Partial())

    def test_partial_protocol_missing_save_key_raises(self):
        """An object with get_delta + apply_delta but no save_key raises TypeError."""
        _, _, sm = _make_world()

        class Partial:
            def get_delta(self):
                return {}

            def apply_delta(self, d):
                pass

        with pytest.raises(TypeError):
            sm.register(Partial())


# ---------------------------------------------------------------------------
# register() — valid Saveable is accepted
# ---------------------------------------------------------------------------


class TestRegisterValid:
    def test_fake_saveable_accepted(self):
        """A minimal _FakeSaveable satisfies isinstance(x, Saveable)."""
        _, _, sm = _make_world()
        fs = _FakeSaveable("foo")
        assert isinstance(fs, Saveable)
        sm.register(fs)  # must not raise

    def test_two_systems_registered_and_apply_in_order(self, tmp_path):
        """
        Pin registration order: apply_delta is called in the order register()
        was called (terrain before others per ARCHITECTURE §4a.4).
        """
        save_file = tmp_path / "order.ta"
        _, _, sm = _make_world()

        first = _FakeSaveable("alpha", {"v": 1})
        second = _FakeSaveable("beta", {"v": 2})
        sm.register(first)
        sm.register(second)
        sm.save(save_file)

        # Fresh manager, same config/clock — reload
        _, _, sm2 = _make_world()
        first2 = _FakeSaveable("alpha")
        second2 = _FakeSaveable("beta")
        sm2.register(first2)
        sm2.register(second2)
        sm2.load(save_file)

        # Both apply_delta called exactly once
        assert len(first2.applied) == 1
        assert len(second2.applied) == 1

        # Pin apply order — alpha was registered first so it must be applied first.
        # We record a global call sequence by wrapping both into a shared list.
        call_order: list[str] = []

        class _OrderedFake:
            def __init__(self, key):
                self.save_key = key
                self.applied: list[dict] = []

            def get_delta(self):
                return {"v": 0}

            def apply_delta(self, d):
                self.applied.append(d)
                call_order.append(self.save_key)

        _, _, sm3 = _make_world()
        oa = _OrderedFake("alpha")
        ob = _OrderedFake("beta")
        sm3.register(oa)
        sm3.register(ob)
        sm3.load(save_file)

        assert call_order == ["alpha", "beta"], (
            f"Expected apply_delta order ['alpha','beta'], got {call_order}"
        )


# ---------------------------------------------------------------------------
# Atomic write — no leftover .tmp file
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_no_tmp_file_after_successful_save(self, tmp_path):
        """After save(), no *.tmp file remains in tmp_path."""
        save_file = tmp_path / "world.ta"
        _, _, sm = _make_world()
        sm.register(_FakeSaveable("s", {"k": 1}))
        sm.save(save_file)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Leftover .tmp files after save: {tmp_files}"

    def test_save_file_exists_and_nonempty(self, tmp_path):
        """After save(), the destination file exists and has non-zero length."""
        save_file = tmp_path / "world.ta"
        _, _, sm = _make_world()
        sm.register(_FakeSaveable("s", {"k": 1}))
        sm.save(save_file)

        assert save_file.exists()
        assert save_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# Backward compat — absent save_key keeps baseline (apply_delta NOT called)
# ---------------------------------------------------------------------------


class TestAbsentSaveKey:
    def test_newly_registered_system_not_called_when_key_absent(self, tmp_path):
        """
        A system registered after a save was created (so its key is absent from
        the file) must NOT have apply_delta called on load.
        Pin: apply_delta call count == 0.
        """
        save_file = tmp_path / "old.ta"

        # Save with only system "alpha"
        _, _, sm = _make_world()
        sm.register(_FakeSaveable("alpha", {"v": 1}))
        sm.save(save_file)

        # Load with "alpha" + newly-added "gamma" (absent in save)
        _, _, sm2 = _make_world()
        alpha2 = _FakeSaveable("alpha")
        gamma = _FakeSaveable("gamma")
        sm2.register(alpha2)
        sm2.register(gamma)
        sm2.load(save_file)

        # alpha got applied; gamma was absent so it must NOT have been called
        assert len(alpha2.applied) == 1
        assert len(gamma.applied) == 0, (
            "apply_delta must NOT be called for a system whose key is absent from the save"
        )

    def test_absent_key_system_gets_empty_applied_list(self, tmp_path):
        """Complement: the absent system's applied list is truly empty (not {})."""
        save_file = tmp_path / "only_alpha.ta"
        _, _, sm = _make_world()
        sm.register(_FakeSaveable("alpha", {"x": 99}))
        sm.save(save_file)

        _, _, sm2 = _make_world()
        new_sys = _FakeSaveable("newbie")
        sm2.register(_FakeSaveable("alpha"))
        sm2.register(new_sys)
        sm2.load(save_file)

        assert new_sys.applied == []


# ---------------------------------------------------------------------------
# apply_delta error propagation — pin current behavior
# ---------------------------------------------------------------------------


class TestApplyDeltaError:
    def test_exploding_apply_delta_propagates_runtime_error(self, tmp_path):
        """
        If a Saveable.apply_delta raises, pin that load() lets the exception
        propagate (RuntimeError bubbles out of load()).

        SUSPECTED BUG: no try/except around apply_delta in save_manager.py
        at line ~586, so any exception will propagate. Pinning current behavior.
        """
        save_file = tmp_path / "boom.ta"

        # Save with the exploding system
        _, _, sm = _make_world()
        sm.register(_ExplodingApply())
        sm.save(save_file)

        # Load — expect apply_delta's RuntimeError to propagate
        _, _, sm2 = _make_world()
        sm2.register(_ExplodingApply())
        with pytest.raises(RuntimeError, match="intentional apply_delta failure"):
            sm2.load(save_file)

    def test_exploding_apply_delta_leaves_clock_already_restored(self, tmp_path):
        """
        Pin: clock.set_state is called BEFORE apply_delta (per §4a.4), so when
        apply_delta raises, the clock IS already modified (partial state change).

        This is a suspected bug: per the docstring 'no partial load when raised',
        but SaveIncompatibleError is the only exception that enforces this guarantee.
        RuntimeError from apply_delta is NOT caught, so the clock IS partially applied.
        """
        save_file = tmp_path / "boom_clock.ta"

        _, clock, sm = _make_world()
        clock.update(99.0)  # advance clock
        sm.register(_ExplodingApply())
        sm.save(save_file)

        _, clock2, sm2 = _make_world()
        original_state = clock2.get_state()
        sm2.register(_ExplodingApply())

        import contextlib

        with contextlib.suppress(RuntimeError):
            sm2.load(save_file)

        # Pin: clock state WAS changed even though apply_delta raised
        # (i.e. load() is NOT atomic for apply_delta errors)
        changed_state = clock2.get_state()
        assert changed_state != original_state, (
            "Pinning current behavior: clock is modified before apply_delta runs, "
            "so an apply_delta exception leaves the clock in a changed state."
        )


# ---------------------------------------------------------------------------
# config_digest mismatch → SaveIncompatibleError
# ---------------------------------------------------------------------------


class TestConfigDigestMismatch:
    def test_changed_chunk_size_raises(self, tmp_path):
        """Changing chunk_size (affects digest) → SaveIncompatibleError on load."""
        save_file = tmp_path / "digest.ta"

        cfg, _, sm = _make_world(seed=42)
        sm.register(_FakeSaveable("s", {}))
        sm.save(save_file)

        # Create a manager with a mutated chunk_size
        cfg2 = dataclasses.replace(cfg, chunk_size=cfg.chunk_size + 1)
        set_world_seed(42)
        bus2 = EventBus()
        clock2 = Clock(fixed_dt=cfg2.fixed_dt, bus=bus2)
        sm2 = SaveManager(cfg2, clock2)
        sm2.register(_FakeSaveable("s"))

        with pytest.raises(SaveIncompatibleError, match="config_digest"):
            sm2.load(save_file)

    def test_changed_voxel_size_raises(self, tmp_path):
        """Changing voxel_size (affects digest) → SaveIncompatibleError on load."""
        save_file = tmp_path / "vox.ta"

        cfg, _, sm = _make_world(seed=77)
        sm.register(_FakeSaveable("s", {}))
        sm.save(save_file)

        cfg2 = dataclasses.replace(cfg, voxel_size=cfg.voxel_size * 2.0)
        set_world_seed(77)
        bus2 = EventBus()
        clock2 = Clock(fixed_dt=cfg2.fixed_dt, bus=bus2)
        sm2 = SaveManager(cfg2, clock2)
        sm2.register(_FakeSaveable("s"))

        with pytest.raises(SaveIncompatibleError):
            sm2.load(save_file)

    def test_changed_light_grid_scale_raises(self, tmp_path):
        """Changing light_grid_scale (affects digest) → SaveIncompatibleError on load."""
        save_file = tmp_path / "lgs.ta"

        cfg, _, sm = _make_world(seed=55)
        sm.register(_FakeSaveable("s", {}))
        sm.save(save_file)

        cfg2 = dataclasses.replace(cfg, light_grid_scale=cfg.light_grid_scale + 1)
        set_world_seed(55)
        bus2 = EventBus()
        clock2 = Clock(fixed_dt=cfg2.fixed_dt, bus=bus2)
        sm2 = SaveManager(cfg2, clock2)
        sm2.register(_FakeSaveable("s"))

        with pytest.raises(SaveIncompatibleError):
            sm2.load(save_file)

    def test_world_seed_mismatch_raises_with_message(self, tmp_path):
        """world_seed mismatch → SaveIncompatibleError mentioning 'world_seed'."""
        save_file = tmp_path / "seed.ta"

        _, _, sm = _make_world(seed=111)
        sm.register(_FakeSaveable("s", {}))
        sm.save(save_file)

        _, _, sm2 = _make_world(seed=222)
        sm2.register(_FakeSaveable("s"))

        with pytest.raises(SaveIncompatibleError, match="world_seed"):
            sm2.load(save_file)

    def test_format_version_too_new_raises(self, tmp_path):
        """
        A save file with format_version > _FORMAT_VERSION raises
        SaveIncompatibleError (engine too old).
        """

        import msgpack

        save_file = tmp_path / "future.ta"

        _, _, sm = _make_world(seed=1)
        sm.register(_FakeSaveable("s", {}))
        sm.save(save_file)

        # Tamper: bump format_version way beyond current
        raw = save_file.read_bytes()
        envelope = msgpack.unpackb(raw, raw=False)
        envelope["header"]["format_version"] = 9999
        save_file.write_bytes(msgpack.packb(envelope, use_bin_type=True))

        _, _, sm2 = _make_world(seed=1)
        sm2.register(_FakeSaveable("s"))

        with pytest.raises(SaveIncompatibleError):
            sm2.load(save_file)


# ---------------------------------------------------------------------------
# Encoder / decoder edge cases
# ---------------------------------------------------------------------------


class TestEncoderEdges:
    def test_nan_array_round_trips_bit_identical(self):
        """
        float32 array containing NaN round-trips bit-identical via encode/decode.
        Uses tobytes() comparison since NaN != NaN by IEEE 754.

        SUSPECTED BUG: np.frombuffer result may be read-only (no copy), so
        comparing .tobytes() is the safe bit-level check.
        """
        arr = np.array([1.0, float("nan"), float("inf"), -float("inf")], dtype=np.float32)
        encoded = _encode_value(arr)
        decoded = _decode_value(encoded)

        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        assert arr.tobytes() == decoded.tobytes(), (
            "NaN/Inf float32 array must round-trip bit-identically"
        )

    def test_inf_float64_array_round_trips(self):
        """float64 array with +inf/-inf/nan survives encode→decode."""
        arr = np.array([float("nan"), float("inf"), -float("inf"), 0.0], dtype=np.float64)
        encoded = _encode_value(arr)
        decoded = _decode_value(encoded)

        assert decoded.dtype == np.float64
        assert arr.tobytes() == decoded.tobytes()

    def test_empty_array_round_trips(self):
        """Zero-element array encodes and decodes to an array of the same dtype and shape."""
        arr = np.array([], dtype=np.uint8)
        encoded = _encode_value(arr)
        decoded = _decode_value(encoded)

        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.uint8
        assert decoded.shape == (0,)
        assert len(decoded) == 0

    def test_noncontiguous_array_round_trips(self):
        """
        A non-contiguous (transposed) array is pinned through encode/decode.

        SUSPECTED BUG: _encode_value calls obj.tobytes() which uses the actual
        memory layout of the non-contiguous array, and reshape() on decode uses
        the stored shape.  For a transposed view the stored shape is the
        transposed shape, so the decoded array will have the transposed shape —
        values will be in the transposed byte order, NOT the original logical order.

        Pin current behavior: the decoded array has the transposed shape, and its
        bytes match the transposed (non-contiguous) view's bytes.
        """
        original = np.arange(6, dtype=np.float32).reshape(2, 3)
        transposed = original.T  # shape (3, 2), non-contiguous

        encoded = _encode_value(transposed)
        decoded = _decode_value(encoded)

        # Pin: decoded shape matches the transposed shape (3, 2)
        assert decoded.shape == transposed.shape, (
            f"Expected decoded shape {transposed.shape}, got {decoded.shape}"
        )
        # Pin: decoded bytes match the transposed view's bytes
        assert decoded.tobytes() == transposed.tobytes(), (
            "Non-contiguous array bytes must match transposed view bytes after round-trip"
        )

    def test_nested_dict_string_keys(self):
        """Nested dict with only string keys encodes without kv_pairs wrapper."""
        delta = {
            "outer": {
                "inner_a": 42,
                "inner_b": [1, 2, 3],
            },
            "count": 7,
        }
        raw = _encode_delta(delta)
        recovered = _decode_delta(raw)

        assert recovered["outer"]["inner_a"] == 42
        assert recovered["outer"]["inner_b"] == [1, 2, 3]
        assert recovered["count"] == 7

    def test_mixed_string_and_tuple_keys_in_top_level(self):
        """
        A delta dict whose top-level keys include at least one non-string key
        triggers kv_pairs encoding at the top level.
        """
        arr = np.zeros((4,), dtype=np.int32)
        delta = {
            (0, 0, 0): arr,
            (1, 2, 3): arr + 1,
        }
        raw = _encode_delta(delta)
        recovered = _decode_delta(raw)

        assert set(recovered.keys()) == {(0, 0, 0), (1, 2, 3)}
        assert np.array_equal(recovered[(0, 0, 0)], arr)
        assert np.array_equal(recovered[(1, 2, 3)], arr + 1)

    def test_2d_uint8_array_round_trips(self):
        """2D uint8 array round-trips with correct shape and values."""
        arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
        raw = _encode_delta({"arr": arr})
        recovered = _decode_delta(raw)

        assert np.array_equal(recovered["arr"], arr)
        assert recovered["arr"].dtype == np.uint8


# ---------------------------------------------------------------------------
# SaveIncompatibleError construction and typing
# ---------------------------------------------------------------------------


class TestSaveIncompatibleError:
    def test_is_exception_subclass(self):
        """SaveIncompatibleError is a subclass of Exception."""
        exc = SaveIncompatibleError("something went wrong")
        assert isinstance(exc, Exception)

    def test_message_preserved(self):
        """The message passed to SaveIncompatibleError is preserved in str()."""
        msg = "world_seed mismatch: saved=1 current=2"
        exc = SaveIncompatibleError(msg)
        assert msg in str(exc)

    def test_can_be_raised_and_caught(self):
        """SaveIncompatibleError can be raised and caught as Exception."""
        with pytest.raises(SaveIncompatibleError):
            raise SaveIncompatibleError("test")

    def test_can_be_caught_specifically(self):
        """SaveIncompatibleError can be caught by its specific type."""
        with pytest.raises(SaveIncompatibleError):
            raise SaveIncompatibleError("caught specifically")


# ---------------------------------------------------------------------------
# Saveable protocol structural checks
# ---------------------------------------------------------------------------


class TestSaveableProtocolStructural:
    def test_full_impl_is_saveable(self):
        """A class implementing all three members passes isinstance(x, Saveable)."""

        class FullImpl:
            save_key = "full"

            def get_delta(self):
                return {}

            def apply_delta(self, d):
                pass

        assert isinstance(FullImpl(), Saveable)

    def test_missing_apply_delta_fails_isinstance(self):
        """A class missing apply_delta fails isinstance(x, Saveable)."""

        class Incomplete:
            save_key = "x"

            def get_delta(self):
                return {}

        assert not isinstance(Incomplete(), Saveable)

    def test_missing_get_delta_fails_isinstance(self):
        """A class missing get_delta fails isinstance(x, Saveable)."""

        class Incomplete:
            save_key = "x"

            def apply_delta(self, d):
                pass

        assert not isinstance(Incomplete(), Saveable)

    def test_missing_save_key_fails_isinstance(self):
        """A class missing save_key fails isinstance(x, Saveable)."""

        class Incomplete:
            def get_delta(self):
                return {}

            def apply_delta(self, d):
                pass

        assert not isinstance(Incomplete(), Saveable)


# ---------------------------------------------------------------------------
# _compute_config_digest — stability and sensitivity
# ---------------------------------------------------------------------------


class TestConfigDigest:
    def test_same_config_produces_same_digest(self):
        """Same config always produces the same digest (determinism)."""
        cfg = load_config()
        d1 = _compute_config_digest(cfg)
        d2 = _compute_config_digest(cfg)
        assert d1 == d2

    def test_digest_is_32_hex_chars(self):
        """Digest is exactly 32 lowercase hex characters (blake2b digest_size=16)."""
        cfg = load_config()
        digest = _compute_config_digest(cfg)
        assert len(digest) == 32
        assert all(c in "0123456789abcdef" for c in digest)

    def test_different_chunk_size_different_digest(self):
        """Changing chunk_size changes the digest."""
        cfg = load_config()
        cfg2 = dataclasses.replace(cfg, chunk_size=cfg.chunk_size + 1)
        assert _compute_config_digest(cfg) != _compute_config_digest(cfg2)

    def test_debug_flag_does_not_change_digest(self):
        """Changing show_fps (debug flag) does NOT change the digest."""
        cfg = load_config()
        cfg2 = dataclasses.replace(cfg, show_fps=not cfg.show_fps)
        assert _compute_config_digest(cfg) == _compute_config_digest(cfg2)
