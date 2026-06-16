"""
tests/devtools/test_fields.py — tests for fire_engine/devtools/fields.py.

fields.py is a thin re-export shim (Field, Section, Button, Panel come from
types.py; FieldKind comes from enums.py).  Tests confirm the re-exports are
importable from this path AND exercise one real behaviour of each re-exported
symbol. Fully headless; no panda3d imports.
"""

from __future__ import annotations

from fire_engine.devtools.fields import Button, Field, FieldKind, Panel, Section

# ---------------------------------------------------------------------------
# Re-export presence
# ---------------------------------------------------------------------------


def test_field_importable_from_fields():
    assert Field is not None


def test_section_importable_from_fields():
    assert Section is not None


def test_button_importable_from_fields():
    assert Button is not None


def test_panel_importable_from_fields():
    assert Panel is not None


def test_fieldkind_importable_from_fields():
    assert FieldKind is not None


def test_dunder_all_covers_public_symbols():
    import fire_engine.devtools.fields as mod

    assert hasattr(mod, "__all__")
    for name in ("Button", "Field", "FieldKind", "Panel", "Section"):
        assert name in mod.__all__, f"{name!r} missing from __all__"


# ---------------------------------------------------------------------------
# Field behaviour
# ---------------------------------------------------------------------------


def test_field_get_returns_live_value():
    state = {"v": 42}
    f = Field("x", FieldKind.INT, lambda: state["v"])
    assert f.get() == 42
    state["v"] = 99
    assert f.get() == 99


def test_field_set_applies_value():
    store = {}
    f = Field("x", FieldKind.FLOAT, lambda: store.get("v"), lambda v: store.update(v=v))
    f.set(3.14)
    assert store["v"] == 3.14


def test_field_read_only_when_no_setter():
    f = Field("label", FieldKind.LABEL, lambda: "hello")
    assert f.read_only is True
    assert f.set is None


def test_field_not_read_only_with_setter():
    f = Field("x", FieldKind.INT, lambda: 0, lambda v: None)
    assert f.read_only is False


def test_field_defaults():
    f = Field("a", FieldKind.BOOL, lambda: True)
    assert f.step == 0.1
    assert f.units == ""
    assert f.choices is None


# ---------------------------------------------------------------------------
# Section behaviour
# ---------------------------------------------------------------------------


def test_section_holds_title_and_fields():
    f = Field("n", FieldKind.INT, lambda: 7)
    s = Section("Props", [f])
    assert s.title == "Props"
    assert s.fields == [f]


def test_section_empty_fields():
    s = Section("Empty", [])
    assert s.fields == []


# ---------------------------------------------------------------------------
# Button behaviour
# ---------------------------------------------------------------------------


def test_button_on_click_invoked():
    fired = []
    b = Button("Go", lambda: fired.append(1))
    b.on_click()
    assert fired == [1]


def test_button_label():
    b = Button("Launch", lambda: None)
    assert b.label == "Launch"


# ---------------------------------------------------------------------------
# Panel behaviour
# ---------------------------------------------------------------------------


def test_panel_defaults():
    p = Panel("id", "Title", [])
    assert p.tool_id == "id"
    assert p.title == "Title"
    assert p.sections == []
    assert p.buttons == []
    assert p.revision == 0


def test_panel_with_sections_and_buttons():
    sec = Section("S", [])
    btn = Button("B", lambda: None)
    p = Panel("t", "T", [sec], buttons=[btn], revision=5)
    assert p.sections == [sec]
    assert p.buttons == [btn]
    assert p.revision == 5
