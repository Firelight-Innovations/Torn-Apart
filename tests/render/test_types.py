"""
tests/render/test_types.py — Headless tests for render/types.py (InputState dataclass).

No panda3d imports.
"""

from __future__ import annotations

from dataclasses import fields

from fire_engine.render.types import InputState


class TestInputState:
    """Tests for the InputState plain-data snapshot."""

    def test_default_construction(self) -> None:
        state = InputState()
        assert state.move_forward is False
        assert state.move_backward is False
        assert state.move_left is False
        assert state.move_right is False
        assert state.move_up is False
        assert state.move_down is False
        assert state.sprint is False
        assert state.mouse_dx == 0.0
        assert state.mouse_dy == 0.0
        assert state.mouse_captured is False
        assert state.escape_pressed is False

    def test_field_count(self) -> None:
        """InputState must have exactly 11 fields — catches accidental additions/removals."""
        assert len(fields(InputState)) == 11

    def test_keyword_construction(self) -> None:
        state = InputState(
            move_forward=True,
            sprint=True,
            mouse_dx=3.5,
            mouse_dy=-2.0,
            mouse_captured=True,
        )
        assert state.move_forward is True
        assert state.sprint is True
        assert state.mouse_dx == 3.5
        assert state.mouse_dy == -2.0
        assert state.mouse_captured is True
        # Untouched fields stay at defaults
        assert state.move_backward is False

    def test_mutation(self) -> None:
        """InputState is a plain dataclass (not frozen); fields are mutable."""
        state = InputState()
        state.move_left = True
        state.mouse_dx = 10.0
        assert state.move_left is True
        assert state.mouse_dx == 10.0

    def test_bool_fields_accept_bool(self) -> None:
        state = InputState(move_right=True, escape_pressed=True)
        assert state.move_right is True
        assert state.escape_pressed is True

    def test_float_fields(self) -> None:
        state = InputState(mouse_dx=1.23, mouse_dy=-4.56)
        assert abs(state.mouse_dx - 1.23) < 1e-9
        assert abs(state.mouse_dy - (-4.56)) < 1e-9

    def test_independence_between_instances(self) -> None:
        a = InputState(move_forward=True)
        b = InputState()
        assert a.move_forward is True
        assert b.move_forward is False
        a.move_forward = False
        assert b.move_forward is False
