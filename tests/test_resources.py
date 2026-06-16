"""
tests/test_resources.py — Headless and window-marked tests for the Resource Manager.

Headless tests (default run)
-----------------------------
Use a fake loader registered via ``register_loader(".fake", ...)`` to verify
all cache, refcount, eviction, dispatch, and error behaviour without any
panda3d dependency.

Window test (``@pytest.mark.window``)
--------------------------------------
Loads ``tests/fixtures/triangle.egg`` through
``world.resource_adapter.register_panda_loaders`` + the real Panda3D loader.
Excluded from the default headless run via ``addopts = -m "not window"`` in
``pytest.ini``.  Run explicitly with::

    .venv/Scripts/python.exe -m pytest tests/test_resources.py -m window -q
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Helpers: fake loaders module for headless isolation
# ---------------------------------------------------------------------------


class _FakeLoadersModule:
    """
    A minimal fake loaders module used to isolate ResourceManager tests from
    the real (panda3d-backed) loaders.

    Registered loaders are stored in a plain dict keyed by suffix.
    """

    def __init__(self) -> None:
        self._loaders: dict = {}

    def register_loader(self, suffix: str, fn) -> None:
        self._loaders[suffix.lower()] = fn

    def dispatch(self, path: str) -> object:
        from fire_engine.resources.loaders import UnknownResourceFormatError

        dot_idx = path.rfind(".")
        suffix = path[dot_idx:].lower() if dot_idx != -1 else ""
        if suffix not in self._loaders or self._loaders[suffix] is None:
            raise UnknownResourceFormatError(path, suffix)
        return self._loaders[suffix](path)


def _fresh_manager(fake_loaders=None):
    """
    Return a new isolated ResourceManager backed by *fake_loaders*.

    By default creates a new _FakeLoadersModule with a ``.fake`` loader
    pre-registered.
    """
    from fire_engine.resources.manager import ResourceManager

    if fake_loaders is None:
        fake_loaders = _FakeLoadersModule()
        fake_loaders.register_loader(".fake", lambda p: {"path": p, "data": b"fake"})

    return ResourceManager(loaders_module=fake_loaders), fake_loaders


# ---------------------------------------------------------------------------
# Headless: cache identity
# ---------------------------------------------------------------------------


class TestCacheIdentity:
    """load() of the same path must return the identical Handle object."""

    def test_same_path_same_handle(self):
        manager, _ = _fresh_manager()
        h1 = manager.load("assets/test.fake")
        h2 = manager.load("assets/test.fake")
        assert h1 is h2, "Expected same Handle instance on second load"

    def test_normalised_path_same_handle(self):
        """Different path spellings that normalise to the same key share a handle."""
        manager, _ = _fresh_manager()
        # Use os.path separator variations on this platform
        path_a = os.path.join("assets", "test.fake")
        path_b = "assets/test.fake"
        h1 = manager.load(path_a)
        h2 = manager.load(path_b)
        assert h1 is h2, "Normalised paths must yield the same Handle"

    def test_different_paths_different_handles(self):
        fake = _FakeLoadersModule()
        fake.register_loader(".fake", lambda p: {"p": p})
        manager = __import__(
            "fire_engine.resources.manager", fromlist=["ResourceManager"]
        ).ResourceManager(loaders_module=fake)
        h1 = manager.load("a.fake")
        h2 = manager.load("b.fake")
        assert h1 is not h2


# ---------------------------------------------------------------------------
# Headless: refcount lifecycle
# ---------------------------------------------------------------------------


class TestRefcount:
    """acquire/release adjust refcount correctly; never goes negative."""

    def test_initial_refcount_is_zero(self):
        manager, _ = _fresh_manager()
        h = manager.load("x.fake")
        assert h.refcount == 0

    def test_acquire_increments(self):
        manager, _ = _fresh_manager()
        h = manager.load("x.fake")
        manager.acquire(h)
        assert h.refcount == 1

    def test_multiple_acquires(self):
        manager, _ = _fresh_manager()
        h = manager.load("x.fake")
        manager.acquire(h)
        manager.acquire(h)
        manager.acquire(h)
        assert h.refcount == 3

    def test_release_decrements(self):
        manager, _ = _fresh_manager()
        h = manager.load("x.fake")
        manager.acquire(h)
        manager.release(h)
        assert h.refcount == 0

    def test_release_never_negative(self):
        manager, _ = _fresh_manager()
        h = manager.load("x.fake")
        manager.release(h)  # refcount already 0 — must not go negative
        assert h.refcount == 0

    def test_acquire_returns_handle(self):
        manager, _ = _fresh_manager()
        h = manager.load("x.fake")
        returned = manager.acquire(h)
        assert returned is h

    def test_acquire_release_balance(self):
        manager, _ = _fresh_manager()
        h = manager.load("x.fake")
        for _ in range(5):
            manager.acquire(h)
        for _ in range(5):
            manager.release(h)
        assert h.refcount == 0


# ---------------------------------------------------------------------------
# Headless: unload_unreferenced
# ---------------------------------------------------------------------------


class TestUnloadUnreferenced:
    """Only zero-ref handles are evicted; non-zero handles stay."""

    def test_evicts_zero_ref(self):
        manager, _ = _fresh_manager()
        h = manager.load("evict_me.fake")
        # refcount is 0 — should be evicted
        count = manager.unload_unreferenced()
        assert count == 1
        s = manager.stats()
        assert s["cache_size"] == 0

    def test_keeps_nonzero_ref(self):
        manager, _ = _fresh_manager()
        h = manager.load("keep_me.fake")
        manager.acquire(h)  # refcount → 1
        count = manager.unload_unreferenced()
        assert count == 0
        s = manager.stats()
        assert s["cache_size"] == 1

    def test_partial_eviction(self):
        """Load three handles; acquire one; unload_unreferenced evicts the other two."""
        fake = _FakeLoadersModule()
        fake.register_loader(".fake", lambda p: p)
        from fire_engine.resources.manager import ResourceManager

        manager = ResourceManager(loaders_module=fake)

        h_keep = manager.load("keep.fake")
        h_ev1 = manager.load("evict1.fake")
        h_ev2 = manager.load("evict2.fake")

        manager.acquire(h_keep)  # refcount → 1

        count = manager.unload_unreferenced()
        assert count == 2

        # The kept handle is still reachable via load (cache hit)
        h_again = manager.load("keep.fake")
        assert h_again is h_keep

    def test_evicted_handle_reloaded_on_next_load(self):
        """After eviction, the same path produces a NEW Handle on next load."""
        manager, _ = _fresh_manager()
        h1 = manager.load("reload.fake")
        manager.unload_unreferenced()
        h2 = manager.load("reload.fake")
        # Different Handle objects (re-loaded)
        assert h1 is not h2

    def test_cleanup_called_on_eviction(self):
        """If the resource has a .cleanup() method it is called on eviction."""
        cleanup_called = []

        class _Cleanable:
            def cleanup(self):
                cleanup_called.append(True)

        fake = _FakeLoadersModule()
        fake.register_loader(".fake", lambda p: _Cleanable())
        from fire_engine.resources.manager import ResourceManager

        manager = ResourceManager(loaders_module=fake)

        manager.load("clean.fake")  # refcount 0
        manager.unload_unreferenced()
        assert cleanup_called == [True]


# ---------------------------------------------------------------------------
# Headless: suffix dispatch
# ---------------------------------------------------------------------------


class TestSuffixDispatch:
    """dispatch() routes to the correct loader based on file suffix."""

    def test_fake_suffix_routes_correctly(self):
        fake = _FakeLoadersModule()
        call_log = []
        fake.register_loader(".fake", lambda p: call_log.append(p) or "result")
        from fire_engine.resources.manager import ResourceManager

        manager = ResourceManager(loaders_module=fake)
        h = manager.load("some/path/asset.fake")
        assert h.resource == "result"
        assert len(call_log) == 1

    def test_two_suffixes_route_independently(self):
        fake = _FakeLoadersModule()
        fake.register_loader(".alpha", lambda p: "alpha_result")
        fake.register_loader(".beta", lambda p: "beta_result")
        from fire_engine.resources.manager import ResourceManager

        manager = ResourceManager(loaders_module=fake)
        ha = manager.load("file.alpha")
        hb = manager.load("file.beta")
        assert ha.resource == "alpha_result"
        assert hb.resource == "beta_result"


# ---------------------------------------------------------------------------
# Headless: unknown suffix
# ---------------------------------------------------------------------------


class TestUnknownSuffix:
    """Unregistered suffix raises UnknownResourceFormatError."""

    def test_unknown_suffix_raises(self):
        from fire_engine.resources.loaders import UnknownResourceFormatError

        manager, _ = _fresh_manager()
        with pytest.raises(UnknownResourceFormatError) as exc_info:
            manager.load("model.xyz")
        assert ".xyz" in str(exc_info.value)

    def test_error_has_correct_suffix_attribute(self):
        from fire_engine.resources.loaders import UnknownResourceFormatError

        manager, _ = _fresh_manager()
        try:
            manager.load("sound.mp3")
        except UnknownResourceFormatError as e:
            assert e.suffix == ".mp3"
        else:
            pytest.fail("Expected UnknownResourceFormatError was not raised")

    def test_known_but_unregistered_suffix_raises(self):
        """
        A suffix in the known dispatch table (e.g. ".egg") but with no loader
        registered yet should also raise UnknownResourceFormatError.
        """
        from fire_engine.resources.loaders import UnknownResourceFormatError

        # Use a fresh fake loaders module that knows nothing about .egg
        manager, _ = _fresh_manager()
        with pytest.raises(UnknownResourceFormatError):
            manager.load("model.egg")  # .egg not in _FakeLoadersModule


# ---------------------------------------------------------------------------
# Headless: stats()
# ---------------------------------------------------------------------------


class TestStats:
    """stats() returns sane numbers reflecting cache state."""

    def test_empty_cache_stats(self):
        manager, _ = _fresh_manager()
        s = manager.stats()
        assert s["cache_size"] == 0
        assert s["total_handles"] == 0
        assert s["zero_ref"] == 0
        assert s["nonzero_ref"] == 0
        assert s["max_refcount"] == 0
        assert s["total_refcount"] == 0

    def test_stats_after_load(self):
        manager, _ = _fresh_manager()
        manager.load("a.fake")
        s = manager.stats()
        assert s["cache_size"] == 1
        assert s["zero_ref"] == 1  # refcount is 0 at load time
        assert s["nonzero_ref"] == 0

    def test_stats_after_acquire(self):
        manager, _ = _fresh_manager()
        h = manager.load("a.fake")
        manager.acquire(h)
        s = manager.stats()
        assert s["nonzero_ref"] == 1
        assert s["zero_ref"] == 0
        assert s["max_refcount"] == 1
        assert s["total_refcount"] == 1

    def test_stats_mixed(self):
        fake = _FakeLoadersModule()
        fake.register_loader(".fake", lambda p: p)
        from fire_engine.resources.manager import ResourceManager

        manager = ResourceManager(loaders_module=fake)

        h1 = manager.load("a.fake")
        h2 = manager.load("b.fake")
        manager.acquire(h1)
        manager.acquire(h1)  # refcount 2
        manager.acquire(h2)  # refcount 1

        s = manager.stats()
        assert s["cache_size"] == 2
        assert s["nonzero_ref"] == 2
        assert s["zero_ref"] == 0
        assert s["max_refcount"] == 2
        assert s["total_refcount"] == 3

    def test_stats_after_unload(self):
        manager, _ = _fresh_manager()
        manager.load("a.fake")
        manager.unload_unreferenced()
        s = manager.stats()
        assert s["cache_size"] == 0


# ---------------------------------------------------------------------------
# Headless: module-level default instance
# ---------------------------------------------------------------------------


class TestDefaultManager:
    """The module-level default_manager and convenience functions work correctly."""

    def test_default_manager_exists(self):
        from fire_engine.resources import ResourceManager, default_manager

        assert isinstance(default_manager, ResourceManager)

    def test_register_loader_on_global(self):
        """register_loader wires into the global loaders module dispatch table."""
        import fire_engine.resources.loaders as _loaders

        _loaders.register_loader(".testfmt", lambda p: "test_resource")
        from fire_engine.resources.loaders import dispatch

        result = dispatch("something.testfmt")
        assert result == "test_resource"
        # Clean up
        _loaders._LOADERS.pop(".testfmt", None)


# ---------------------------------------------------------------------------
# Window test: real Panda3D loader
# ---------------------------------------------------------------------------


@pytest.mark.window
def test_load_triangle_egg_with_panda3d():
    """
    Load tests/fixtures/triangle.egg through the real Panda3D loader.

    This test requires a Panda3D ShowBase (graphics pipe) — it is excluded
    from the default headless run via ``addopts = -m "not window"`` in
    pytest.ini and must be explicitly invoked::

        pytest tests/test_resources.py -m window -q

    If no display is available (e.g. headless CI), this test will still be
    collected but may fail at ShowBase initialisation — that is expected and
    documented.
    """
    # Locate the fixture relative to this test file
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "triangle.egg")
    assert os.path.exists(fixture_path), (
        f"Test fixture not found: {fixture_path!r}. Ensure tests/fixtures/triangle.egg is present."
    )

    # We need a ShowBase before Panda3D's global loader is available.
    # Use an offscreen window to avoid needing a display.
    from panda3d.core import loadPrcFileData  # type: ignore[import]

    loadPrcFileData("", "window-type offscreen\naudio-library-name null")

    from direct.showbase.ShowBase import ShowBase  # type: ignore[import]

    base = ShowBase()

    try:
        # Create a fresh ResourceManager backed by the real loaders module
        # but with a clean private loaders state to avoid polluting the global.
        from fire_engine.render.resource_adapter import register_panda_loaders
        from fire_engine.resources.manager import ResourceManager

        manager = ResourceManager()
        register_panda_loaders(manager)

        handle = manager.load(fixture_path)
        assert handle is not None, "load() returned None"
        assert handle.resource is not None, "Handle.resource is None"

        # A Panda3D NodePath should have a getName method
        nodepath = handle.resource
        assert hasattr(nodepath, "getName") or hasattr(nodepath, "getChildren"), (
            f"Expected a Panda3D NodePath, got {type(nodepath)}"
        )
    finally:
        base.destroy()
