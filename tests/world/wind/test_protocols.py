"""
tests/world/wind/test_protocols.py — Tests for fire_engine/world/wind/protocols.py.

Covers the WindModifier Protocol: runtime-checkability, structural matching,
and that the exported symbol is the canonical Protocol class.

Headless only. No panda3d. No per-element Python loops.
"""

from __future__ import annotations

import numpy as np

from fire_engine.world.wind.protocols import WindModifier

# ---------------------------------------------------------------------------
# CORRECTNESS — WindModifier is a runtime-checkable Protocol
# ---------------------------------------------------------------------------


class TestWindModifierProtocol:
    def test_module_exports_wind_modifier(self):
        """WindModifier is importable directly from protocols module."""
        assert WindModifier is not None

    def test_is_a_protocol(self):
        """WindModifier is decorated with @runtime_checkable Protocol."""
        # CPython marks Protocol subclasses with _is_protocol=True or
        # __protocol_attrs__; either is acceptable structural evidence.
        is_proto = getattr(WindModifier, "_is_protocol", False) or hasattr(
            WindModifier, "__protocol_attrs__"
        )
        assert is_proto, "WindModifier does not appear to be a Protocol"

    def test_duck_type_satisfies_protocol(self):
        """A class with apply(X, Y, t, vx, vy, turb) satisfies WindModifier."""

        class Stub:
            def apply(
                self,
                X: np.ndarray,
                Y: np.ndarray,
                t: float,
                vx: np.ndarray,
                vy: np.ndarray,
                turb: np.ndarray,
            ) -> None:
                pass

        assert isinstance(Stub(), WindModifier)

    def test_class_without_apply_does_not_satisfy(self):
        """A class that lacks apply() is NOT a WindModifier."""

        class NoApply:
            def run(self, X, Y, t, vx, vy, turb):
                pass

        assert not isinstance(NoApply(), WindModifier)

    def test_empty_class_does_not_satisfy(self):
        """An empty class is not a WindModifier."""

        class Empty:
            pass

        assert not isinstance(Empty(), WindModifier)

    def test_apply_must_match_by_name(self):
        """A class with __call__ instead of apply() is NOT a WindModifier."""

        class CallOnly:
            def __call__(self, X, Y, t, vx, vy, turb):
                pass

        assert not isinstance(CallOnly(), WindModifier)


# ---------------------------------------------------------------------------
# CORRECTNESS — real apply() behaviour via structural check
# ---------------------------------------------------------------------------


class TestWindModifierApply:
    """Verify apply() semantics via a concrete implementation."""

    def _make_arrays(self, cells: int = 8):
        xs = (np.arange(cells) + 0.5) * 4.0
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        vx = np.zeros_like(X)
        vy = np.zeros_like(X)
        turb = np.zeros_like(X)
        return X, Y, vx, vy, turb

    def test_conforming_impl_mutates_in_place(self):
        """A conforming implementation that mutates in place works as expected."""

        class ConstantWind:
            """Adds a constant +X gust to the whole field."""

            def apply(self, X, Y, t, vx, vy, turb):
                vx += 3.0

        impl = ConstantWind()
        assert isinstance(impl, WindModifier)

        X, Y, vx, vy, turb = self._make_arrays()
        impl.apply(X, Y, 0.0, vx, vy, turb)
        assert np.all(vx == 3.0), "ConstantWind did not add 3 m/s to vx"
        assert np.all(vy == 0.0), "ConstantWind should not affect vy"

    def test_apply_return_value_ignored(self):
        """Protocol contract: apply() return value is always None (ignored)."""

        class ReturnsNone:
            def apply(self, X, Y, t, vx, vy, turb):
                return None

        impl = ReturnsNone()
        X, Y, vx, vy, turb = self._make_arrays()
        result = impl.apply(X, Y, 0.0, vx, vy, turb)
        assert result is None

    def test_protocol_allows_stateless_implementation(self):
        """A pure-function implementation (no __init__ state) satisfies the Protocol."""

        class Stateless:
            def apply(self, X, Y, t, vx, vy, turb):
                # Add sin wave in X
                vx += np.sin(X).astype(np.float32)

        impl = Stateless()
        assert isinstance(impl, WindModifier)
        X, Y, vx, vy, turb = self._make_arrays()
        impl.apply(X, Y, 1.0, vx, vy, turb)
        # vx now contains sin(X) values — just assert finite and changed
        assert np.isfinite(vx).all()
        assert not np.all(vx == 0.0)
