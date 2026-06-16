"""
tests/resources/test_types.py — Unit tests for fire_engine.resources.types.

Covers: Handle construction, slots, initial state, repr, and mutation
of path/resource/refcount fields.

Headless only — no panda3d imports.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Handle — construction and initial state
# ---------------------------------------------------------------------------


class TestHandleConstruction:
    """Handle.__init__ sets fields correctly; refcount always starts at 0."""

    def test_resource_stored(self):
        from fire_engine.resources.types import Handle

        sentinel = object()
        h = Handle(resource=sentinel, path="assets/foo.fake")
        assert h.resource is sentinel

    def test_path_stored(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="assets/bar.fake")
        assert h.path == "assets/bar.fake"

    def test_initial_refcount_is_zero(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=42, path="x.fake")
        assert h.refcount == 0

    def test_refcount_type_is_int(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="p.fake")
        assert isinstance(h.refcount, int)

    def test_resource_can_be_none(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="null.fake")
        assert h.resource is None

    def test_resource_can_be_dict(self):
        from fire_engine.resources.types import Handle

        data = {"key": "value"}
        h = Handle(resource=data, path="data.fake")
        assert h.resource == {"key": "value"}

    def test_resource_can_be_bytes(self):
        from fire_engine.resources.types import Handle

        raw = b"\x00\x01\x02"
        h = Handle(resource=raw, path="raw.fake")
        assert h.resource == raw


# ---------------------------------------------------------------------------
# Handle — __slots__ contract
# ---------------------------------------------------------------------------


class TestHandleSlots:
    """Handle uses __slots__ — no __dict__, only the declared attributes."""

    def test_has_slots_attribute(self):
        from fire_engine.resources.types import Handle

        assert hasattr(Handle, "__slots__")

    def test_slots_contains_resource(self):
        from fire_engine.resources.types import Handle

        assert "resource" in Handle.__slots__

    def test_slots_contains_path(self):
        from fire_engine.resources.types import Handle

        assert "path" in Handle.__slots__

    def test_slots_contains_refcount(self):
        from fire_engine.resources.types import Handle

        assert "refcount" in Handle.__slots__

    def test_no_arbitrary_attributes(self):
        """Assigning an undeclared attribute must raise AttributeError."""
        import pytest

        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="p.fake")
        with pytest.raises(AttributeError):
            h.undeclared_field = "should fail"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Handle — field mutability
# ---------------------------------------------------------------------------


class TestHandleMutability:
    """All three fields are mutable after construction (no frozen semantics)."""

    def test_refcount_can_be_incremented(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="p.fake")
        h.refcount += 1
        assert h.refcount == 1

    def test_refcount_can_be_decremented(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="p.fake")
        h.refcount = 3
        h.refcount -= 1
        assert h.refcount == 2

    def test_resource_can_be_replaced(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource="old", path="p.fake")
        h.resource = "new"
        assert h.resource == "new"

    def test_path_can_be_replaced(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="original.fake")
        h.path = "updated.fake"
        assert h.path == "updated.fake"


# ---------------------------------------------------------------------------
# Handle — __repr__
# ---------------------------------------------------------------------------


class TestHandleRepr:
    """__repr__ includes path, refcount, and the resource type name."""

    def test_repr_contains_path(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=42, path="assets/model.egg")
        assert "assets/model.egg" in repr(h)

    def test_repr_contains_refcount(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="p.fake")
        h.refcount = 7
        assert "7" in repr(h)

    def test_repr_contains_resource_type(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource={"data": 1}, path="p.fake")
        r = repr(h)
        # The type name "dict" must appear somewhere in the repr
        assert "dict" in r

    def test_repr_is_string(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="p.fake")
        assert isinstance(repr(h), str)

    def test_repr_none_resource_type_name(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="p.fake")
        r = repr(h)
        assert "NoneType" in r


# ---------------------------------------------------------------------------
# Handle — identity and equality
# ---------------------------------------------------------------------------


class TestHandleIdentity:
    """Two Handle instances are distinct objects even with identical arguments."""

    def test_two_handles_are_not_same_object(self):
        from fire_engine.resources.types import Handle

        h1 = Handle(resource=None, path="same.fake")
        h2 = Handle(resource=None, path="same.fake")
        assert h1 is not h2

    def test_same_handle_is_same_object(self):
        from fire_engine.resources.types import Handle

        h = Handle(resource=None, path="p.fake")
        alias = h
        assert alias is h
