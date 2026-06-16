"""
tests/test_resources_loaders.py — Characterization / golden-master tests for
fire_engine.resources.loaders and the ResourceManager + module-level
convenience functions in fire_engine.resources.manager.

Philosophy
----------
These tests PIN CURRENT BEHAVIOUR.  They do NOT fix bugs.  If a test reveals
an unexpected behaviour (e.g. case insensitivity, silent re-register overwrite)
the comment marks it "SUSPECTED BUG" and the test asserts whatever the code
actually does today so future regressions are caught.

Headless only — no panda3d imports, no real file I/O.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Helpers — snapshot + restore the real _LOADERS table so these tests don't
# pollute the global registry used by TestDefaultManagerConvenience or other
# test modules.
# ---------------------------------------------------------------------------


def _loaders_module():
    import fire_engine.resources.loaders as lm

    return lm


@pytest.fixture()
def clean_loaders():
    """
    Snapshot the real _LOADERS dict, yield, then restore it.
    Each test that touches the global registry uses this fixture.
    """
    import fire_engine.resources.loaders as lm

    original = dict(lm._LOADERS)
    yield lm
    lm._LOADERS.clear()
    lm._LOADERS.update(original)


# ---------------------------------------------------------------------------
# SECTION 1 — loaders.py: dispatch + register_loader
# ---------------------------------------------------------------------------


class TestRegisterAndDispatch:
    """register_loader then dispatch routes to the right loader by suffix."""

    def test_dispatch_calls_registered_loader(self, clean_loaders):
        lm = clean_loaders
        calls = []
        lm.register_loader(".zap", lambda p: calls.append(p) or "zap_result")
        result = lm.dispatch("assets/some.zap")
        assert result == "zap_result"
        assert calls == [os.path.normpath("assets/some.zap") if False else "assets/some.zap"]
        # dispatch passes the path through as-is (no normpath inside loaders.dispatch)
        assert calls[0] == "assets/some.zap"

    def test_dispatch_returns_loader_return_value(self, clean_loaders):
        lm = clean_loaders
        sentinel = object()
        lm.register_loader(".zap", lambda p: sentinel)
        result = lm.dispatch("file.zap")
        assert result is sentinel

    def test_dispatch_unregistered_suffix_raises(self, clean_loaders):
        from fire_engine.resources.loaders import UnknownResourceFormatError

        lm = clean_loaders
        with pytest.raises(UnknownResourceFormatError) as exc_info:
            lm.dispatch("model.xyz")
        err = exc_info.value
        assert isinstance(err, UnknownResourceFormatError)

    def test_dispatch_error_has_path_attribute(self, clean_loaders):
        from fire_engine.resources.loaders import UnknownResourceFormatError

        lm = clean_loaders
        path = "assets/unknown.xyz"
        with pytest.raises(UnknownResourceFormatError) as exc_info:
            lm.dispatch(path)
        assert exc_info.value.path == path

    def test_dispatch_error_has_suffix_attribute(self, clean_loaders):
        from fire_engine.resources.loaders import UnknownResourceFormatError

        lm = clean_loaders
        with pytest.raises(UnknownResourceFormatError) as exc_info:
            lm.dispatch("assets/unknown.xyz")
        assert exc_info.value.suffix == ".xyz"

    def test_known_suffix_with_none_loader_raises(self, clean_loaders):
        """
        .egg is in the table but registered as None at boot.
        dispatch should raise UnknownResourceFormatError, not call None.
        """
        from fire_engine.resources.loaders import UnknownResourceFormatError

        lm = clean_loaders
        # Confirm .egg starts as None in the module-level table
        assert ".egg" in lm._LOADERS
        assert lm._LOADERS[".egg"] is None
        with pytest.raises(UnknownResourceFormatError):
            lm.dispatch("model.egg")

    def test_two_suffixes_route_independently(self, clean_loaders):
        lm = clean_loaders
        lm.register_loader(".aaa", lambda p: "aaa")
        lm.register_loader(".bbb", lambda p: "bbb")
        assert lm.dispatch("x.aaa") == "aaa"
        assert lm.dispatch("x.bbb") == "bbb"


# ---------------------------------------------------------------------------
# SECTION 2 — loaders.py: registered_suffixes()
# ---------------------------------------------------------------------------


class TestRegisteredSuffixes:
    """registered_suffixes() returns sorted list of non-None suffixes."""

    def test_empty_when_all_none(self, clean_loaders):
        """With only None-valued entries, nothing is returned."""
        lm = clean_loaders
        # All built-in entries should be None at boot
        for suffix in list(lm._LOADERS):
            lm._LOADERS[suffix] = None
        result = lm.registered_suffixes()
        assert result == []

    def test_returns_registered_suffix(self, clean_loaders):
        lm = clean_loaders
        for suffix in list(lm._LOADERS):
            lm._LOADERS[suffix] = None
        lm.register_loader(".zap", lambda p: None)
        result = lm.registered_suffixes()
        assert ".zap" in result

    def test_result_is_sorted(self, clean_loaders):
        lm = clean_loaders
        for suffix in list(lm._LOADERS):
            lm._LOADERS[suffix] = None
        for s in [".zzz", ".aaa", ".mmm"]:
            lm.register_loader(s, lambda p: None)
        result = lm.registered_suffixes()
        assert result == sorted(result), "registered_suffixes() must return a sorted list"

    def test_none_slots_excluded(self, clean_loaders):
        """Suffixes whose loader is None must NOT appear in the result."""
        lm = clean_loaders
        lm.register_loader(".present", lambda p: "x")
        # Keep at least one None slot (.egg)
        lm._LOADERS[".egg"] = None
        result = lm.registered_suffixes()
        assert ".egg" not in result
        assert ".present" in result


# ---------------------------------------------------------------------------
# SECTION 3 — loaders.py: case sensitivity of dispatch
# ---------------------------------------------------------------------------


class TestCaseSensitivity:
    """
    Pin the case-handling behaviour of dispatch().

    Current implementation lowercases the extracted suffix before lookup
    (path[dot_idx:].lower()).  So ".EGG" and ".egg" resolve to the same
    slot.  This section pins that behaviour.
    """

    def test_uppercase_suffix_dispatches_to_lowercase_entry(self, clean_loaders):
        """
        PINNED: dispatch() lowercases the suffix, so ".EGG" hits the ".egg" slot.
        """
        lm = clean_loaders
        lm.register_loader(".egg", lambda p: "egg_result")
        # Path with uppercase extension
        result = lm.dispatch("model.EGG")
        # Pin: lowercasing means .EGG resolves to .egg loader
        assert result == "egg_result", (
            "dispatch() must be case-insensitive: .EGG should route to .egg loader"
        )

    def test_register_loader_lowercases_suffix_key(self, clean_loaders):
        """
        register_loader() stores under the lowercase key, so querying
        registered_suffixes() shows the lowercase form even when registered
        with uppercase.
        """
        lm = clean_loaders
        for s in list(lm._LOADERS):
            lm._LOADERS[s] = None
        lm.register_loader(".PNG", lambda p: "png_result")
        result = lm.registered_suffixes()
        # The stored key should be ".png" (lowercased), not ".PNG"
        assert ".png" in result
        assert ".PNG" not in result


# ---------------------------------------------------------------------------
# SECTION 4 — loaders.py: re-registering the same suffix
# ---------------------------------------------------------------------------


class TestReRegister:
    """
    Pin the behaviour when the same suffix is registered twice.
    Current code silently overwrites the previous entry.
    """

    def test_reregister_overwrites_silently(self, clean_loaders):
        """
        PINNED: register_loader with the same suffix does NOT raise; it overwrites.
        """
        lm = clean_loaders
        lm.register_loader(".zap", lambda p: "first")
        lm.register_loader(".zap", lambda p: "second")
        result = lm.dispatch("file.zap")
        # Pin: the second registration wins
        assert result == "second", "Re-registering a suffix should overwrite silently (no error)."

    def test_reregister_none_removes_from_registered_suffixes(self, clean_loaders):
        """
        Overwriting an entry back to None hides it from registered_suffixes().
        NOTE: register_loader() only accepts a LoaderCallable, not None, by
        type annotation, but the runtime behaviour when None is stored directly
        is pinned here via direct _LOADERS mutation.
        """
        lm = clean_loaders
        lm.register_loader(".zap", lambda p: "x")
        assert ".zap" in lm.registered_suffixes()
        # Direct mutation to simulate "de-register"
        lm._LOADERS[".zap"] = None
        assert ".zap" not in lm.registered_suffixes()


# ---------------------------------------------------------------------------
# SECTION 5 — loaders.py: path with no suffix
# ---------------------------------------------------------------------------


class TestNoSuffix:
    """
    Pin the behaviour when dispatch() receives a path with no '.' at all.
    """

    def test_path_without_dot_raises_unknown_error(self, clean_loaders):
        """
        PINNED: A path with no extension produces suffix="" and raises
        UnknownResourceFormatError (not e.g. ValueError or KeyError).
        """
        from fire_engine.resources.loaders import UnknownResourceFormatError

        lm = clean_loaders
        with pytest.raises(UnknownResourceFormatError) as exc_info:
            lm.dispatch("nodotfile")
        err = exc_info.value
        assert isinstance(err, UnknownResourceFormatError)
        # The suffix attribute will be "" (empty string)
        assert err.suffix == ""

    def test_path_without_dot_error_carries_path(self, clean_loaders):
        from fire_engine.resources.loaders import UnknownResourceFormatError

        lm = clean_loaders
        with pytest.raises(UnknownResourceFormatError) as exc_info:
            lm.dispatch("nodotfile")
        assert exc_info.value.path == "nodotfile"


# ---------------------------------------------------------------------------
# Helpers for ResourceManager tests — isolated fake loaders
# ---------------------------------------------------------------------------


class _FakeLoaders:
    """
    Minimal fake loaders module for ResourceManager isolation.
    Mirrors the headless pattern from test_resources.py.
    """

    def __init__(self):
        self._table: dict = {}

    def register_loader(self, suffix: str, fn) -> None:
        self._table[suffix.lower()] = fn

    def dispatch(self, path: str) -> object:
        from fire_engine.resources.loaders import UnknownResourceFormatError

        dot = path.rfind(".")
        suffix = path[dot:].lower() if dot != -1 else ""
        if suffix not in self._table or self._table[suffix] is None:
            raise UnknownResourceFormatError(path, suffix)
        return self._table[suffix](path)


def _make_manager(extra_suffixes=None):
    """
    Return a (manager, fake_loaders) pair with a .fake loader pre-registered.
    extra_suffixes: list of (suffix, fn) pairs to additionally register.
    """
    from fire_engine.resources.manager import ResourceManager

    fake = _FakeLoaders()
    fake.register_loader(".fake", lambda p: {"p": p})
    if extra_suffixes:
        for suf, fn in extra_suffixes:
            fake.register_loader(suf, fn)
    return ResourceManager(loaders_module=fake), fake


# ---------------------------------------------------------------------------
# SECTION 6 — manager.py: load() identity and path normalisation
# ---------------------------------------------------------------------------


class TestManagerLoad:
    """load() returns a Handle with refcount 0; identity and normalisation."""

    def test_load_returns_handle(self):
        from fire_engine.resources.manager import Handle

        manager, _ = _make_manager()
        h = manager.load("x.fake")
        assert isinstance(h, Handle)

    def test_initial_refcount_zero(self):
        manager, _ = _make_manager()
        h = manager.load("x.fake")
        assert h.refcount == 0

    def test_same_path_same_handle_identity(self):
        manager, _ = _make_manager()
        h1 = manager.load("assets/x.fake")
        h2 = manager.load("assets/x.fake")
        assert h1 is h2

    def test_different_paths_different_handles(self):
        manager, _ = _make_manager()
        h1 = manager.load("a.fake")
        h2 = manager.load("b.fake")
        assert h1 is not h2

    def test_forward_backslash_same_handle(self):
        """
        assets/x.fake and assets\\x.fake must normalise to the same Handle.
        """
        manager, _ = _make_manager()
        h1 = manager.load("assets/x.fake")
        h2 = manager.load("assets\\x.fake")
        assert h1 is h2, "Forward-slash and back-slash variants must share the same Handle"

    def test_case_variants_same_handle_on_windows(self):
        """
        PINNED: On Windows, normcase() lowercases paths, so 'A.fake' and
        'a.fake' share the same cache entry.  On other platforms they are
        separate entries.  Pin current platform behaviour.
        """
        manager, _ = _make_manager()
        h1 = manager.load("A.fake")
        h2 = manager.load("a.fake")
        if os.path.normcase("A") == "a":
            # Windows: case-insensitive → same handle
            assert h1 is h2
        else:
            # POSIX: case-sensitive → different handles
            assert h1 is not h2

    def test_handle_path_is_normalised(self):
        """handle.path stores the normalised (normcase+normpath) cache key."""
        manager, _ = _make_manager()
        h = manager.load("assets/x.fake")
        expected_key = os.path.normcase(os.path.normpath("assets/x.fake"))
        assert h.path == expected_key


# ---------------------------------------------------------------------------
# SECTION 7 — manager.py: acquire / release / refcount floor
# ---------------------------------------------------------------------------


class TestManagerRefcount:
    """acquire increments, release decrements, never below 0."""

    def test_acquire_increments(self):
        manager, _ = _make_manager()
        h = manager.load("x.fake")
        manager.acquire(h)
        assert h.refcount == 1

    def test_acquire_returns_same_handle(self):
        manager, _ = _make_manager()
        h = manager.load("x.fake")
        ret = manager.acquire(h)
        assert ret is h

    def test_release_decrements(self):
        manager, _ = _make_manager()
        h = manager.load("x.fake")
        manager.acquire(h)
        manager.release(h)
        assert h.refcount == 0

    def test_release_does_not_go_below_zero(self):
        """PINNED: release() clamps at 0; never negative."""
        manager, _ = _make_manager()
        h = manager.load("x.fake")
        # refcount is already 0 — release should be a no-op
        manager.release(h)
        assert h.refcount == 0, "refcount must not go below 0"

    def test_double_release_stays_zero(self):
        manager, _ = _make_manager()
        h = manager.load("x.fake")
        manager.acquire(h)  # → 1
        manager.release(h)  # → 0
        manager.release(h)  # → must stay 0, not -1
        assert h.refcount == 0


# ---------------------------------------------------------------------------
# SECTION 8 — manager.py: unload_unreferenced
# ---------------------------------------------------------------------------


class TestManagerUnload:
    """unload_unreferenced evicts refcount==0 handles; keeps referenced ones."""

    def test_evicts_zero_ref_handle(self):
        manager, _ = _make_manager()
        manager.load("evict.fake")  # refcount stays 0
        evicted = manager.unload_unreferenced()
        assert evicted == 1
        assert manager.stats()["cache_size"] == 0

    def test_keeps_nonzero_ref_handle(self):
        manager, _ = _make_manager()
        h = manager.load("keep.fake")
        manager.acquire(h)  # refcount → 1
        evicted = manager.unload_unreferenced()
        assert evicted == 0
        assert manager.stats()["cache_size"] == 1

    def test_mixed_eviction(self):
        """Three handles loaded; two zero-ref evicted; one kept."""
        fake = _FakeLoaders()
        fake.register_loader(".fake", lambda p: p)
        from fire_engine.resources.manager import ResourceManager

        manager = ResourceManager(loaders_module=fake)
        h_keep = manager.load("keep.fake")
        manager.load("ev1.fake")
        manager.load("ev2.fake")
        manager.acquire(h_keep)  # only this one survives
        evicted = manager.unload_unreferenced()
        assert evicted == 2
        # h_keep is still in cache and returns the same handle
        assert manager.load("keep.fake") is h_keep

    def test_evicted_path_reloads_as_new_handle(self):
        """After eviction, the same path produces a NEW Handle on next load."""
        manager, _ = _make_manager()
        h1 = manager.load("reload.fake")
        manager.unload_unreferenced()
        h2 = manager.load("reload.fake")
        assert h1 is not h2, "Re-loading an evicted path must return a fresh Handle"

    def test_cleanup_called_on_eviction(self):
        """resource.cleanup() is called if present."""
        cleaned = []

        class _R:
            def cleanup(self):
                cleaned.append(True)

        fake = _FakeLoaders()
        fake.register_loader(".fake", lambda p: _R())
        from fire_engine.resources.manager import ResourceManager

        manager = ResourceManager(loaders_module=fake)
        manager.load("clean.fake")
        manager.unload_unreferenced()
        assert cleaned == [True]


# ---------------------------------------------------------------------------
# SECTION 9 — manager.py: module-level convenience functions on default_manager
# ---------------------------------------------------------------------------


class TestDefaultManagerConvenience:
    """
    Module-level load/acquire/release/unload_unreferenced delegate to
    default_manager.  We register a fake loader into the REAL loaders module,
    restore it afterwards.
    """

    @pytest.fixture(autouse=True)
    def _register_and_restore(self):
        import fire_engine.resources.loaders as lm
        import fire_engine.resources.manager as mm

        # Register a test suffix on the real module
        original_loaders = dict(lm._LOADERS)
        original_cache = dict(mm.default_manager._cache)
        lm.register_loader(".convtest", lambda p: f"loaded:{p}")
        yield
        # Restore loaders table
        lm._LOADERS.clear()
        lm._LOADERS.update(original_loaders)
        # Restore default_manager cache (remove only the key we added)
        for key in list(mm.default_manager._cache):
            if key not in original_cache:
                mm.default_manager._cache.pop(key, None)

    def test_module_load_returns_handle(self):
        import fire_engine.resources.manager as mm
        from fire_engine.resources.manager import Handle

        h = mm.load("something.convtest")
        assert isinstance(h, Handle)

    def test_module_load_same_path_same_handle(self):
        import fire_engine.resources.manager as mm

        h1 = mm.load("x.convtest")
        h2 = mm.load("x.convtest")
        assert h1 is h2

    def test_module_acquire_increments_refcount(self):
        import fire_engine.resources.manager as mm

        h = mm.load("acq.convtest")
        mm.acquire(h)
        assert h.refcount == 1
        mm.release(h)  # clean up

    def test_module_release_decrements_refcount(self):
        import fire_engine.resources.manager as mm

        h = mm.load("rel.convtest")
        mm.acquire(h)
        mm.release(h)
        assert h.refcount == 0

    def test_module_unload_unreferenced_evicts(self):
        import fire_engine.resources.manager as mm

        mm.load("unload_conv.convtest")
        # refcount is 0 — should be evicted
        count = mm.unload_unreferenced()
        assert count >= 1  # may include other zero-ref handles in default_manager

    def test_module_convenience_operate_on_default_manager(self):
        """Verify the module-level fns really go through default_manager."""
        import fire_engine.resources.manager as mm

        h = mm.load("dm.convtest")
        # The handle must be in default_manager's cache
        assert h in mm.default_manager._cache.values()
