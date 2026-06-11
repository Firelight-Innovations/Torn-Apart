"""
procedural/registry.py — Global registry and cache for ProceduralDef instances.

The registry is the single point of access for all procedurally-generated
content.  It:

1. Stores ``ProceduralDef`` instances by name.
2. Lazily generates content on first request.
3. Caches results by ``(name, world_seed, sorted_params)`` so the same
   inputs always return the **same object** (identity equality, not just
   equality by value).
4. Injects a deterministic ``numpy.random.Generator`` via
   ``core.rng.for_domain("procedural", name, params_digest)`` — the caller
   never touches the RNG directly.

Cache invalidation
------------------
The cache key includes the current world seed (read at call time from
``core.rng._world_seed``).  If ``set_world_seed`` is called with a new value
between two ``get`` calls, the cache miss causes fresh generation with the
new seed.

Thread safety
-------------
Not thread-safe.  The engine runs generation on a single thread.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import register, get, clear_cache

    set_world_seed(42)

    # If "wasteland_ground" is already registered (imported from its module):
    arr = get("wasteland_ground")          # (256, 256, 4) uint8
    arr2 = get("wasteland_ground")
    assert arr2 is arr                      # same cached object

    clear_cache()
    arr3 = get("wasteland_ground")
    assert arr3 is not arr                  # regenerated after cache clear
"""

from __future__ import annotations

import hashlib
from typing import Any

from fire_engine.procedural.defs import ProceduralDef

__all__ = ["register", "get", "clear_cache", "reset_registry"]

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# name → ProceduralDef instance
_registry: dict[str, ProceduralDef] = {}

# (name, world_seed, params_digest_str) → generated result
_cache: dict[tuple[str, int, str], Any] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_digest(params: dict) -> str:
    """
    Return a stable, canonical string digest of *params*.

    The digest is used both as a human-readable cache-key component and as
    part of the RNG domain keys.  It is computed with blake2b so it is
    cross-process stable (no Python ``hash()``).

    Parameters
    ----------
    params : dict
        Keyword parameters passed to ``get()``.

    Returns
    -------
    str
        Hex-encoded blake2b-8 digest of the sorted, repr'd params.
    """
    # Canonical form: sorted by key so {"b":1,"a":2} == {"a":2,"b":1}
    canonical = repr(sorted(params.items()))
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
    return digest.hex()


def _current_world_seed() -> int:
    """Return the current world seed from ``core.rng`` module state."""
    # Import here to avoid a circular import at module init time and to
    # always read the *current* value (which changes on set_world_seed).
    from fire_engine.core import rng as _rng  # noqa: PLC0415
    return _rng._world_seed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register(def_instance: ProceduralDef) -> None:
    """
    Register a ``ProceduralDef`` instance in the global registry.

    The instance is stored under ``def_instance.name``.  Registering a
    second definition with the same name replaces the first and clears any
    cached results for that name.

    Parameters
    ----------
    def_instance : ProceduralDef
        A concrete (non-abstract) ``ProceduralDef`` instance with ``name``
        set.  Creating an instance and passing it here is the standard
        pattern; the ``@register_def`` decorator does this automatically.

    Raises
    ------
    TypeError
        If *def_instance* is not a ``ProceduralDef``.
    AttributeError
        If *def_instance* has no ``name`` attribute.

    Example
    -------
    ::

        from fire_engine.procedural.defs import ProceduralDef
        from fire_engine.procedural.registry import register

        class MyDef(ProceduralDef):
            name = "my_def"
            def generate(self, rng, **params):
                return {"value": rng.integers(0, 100)}

        register(MyDef())
    """
    if not isinstance(def_instance, ProceduralDef):
        raise TypeError(
            f"register() expects a ProceduralDef instance, got {type(def_instance)}"
        )
    name: str = def_instance.name
    # Evict stale cache entries for this name (any seed/params combo).
    keys_to_drop = [k for k in _cache if k[0] == name]
    for k in keys_to_drop:
        del _cache[k]
    _registry[name] = def_instance


def get(name: str, **params) -> Any:
    """
    Generate (or return cached) content for the named definition.

    The result is cached by ``(name, world_seed, params_digest)``.  Calling
    ``get`` twice with identical arguments returns the **same object**.

    The ``numpy.random.Generator`` passed to ``def.generate()`` is derived
    from ``core.rng.for_domain("procedural", name, params_digest)`` — fully
    deterministic from the world seed.

    Parameters
    ----------
    name : str
        Registered definition name, e.g. ``"wasteland_ground"``.
    **params : any JSON-serialisable value
        Optional generation parameters.  Must be stable-repr-able (ints,
        floats, strings, tuples of those).

    Returns
    -------
    Any
        The generated result.  For ``ProceduralTextureDef`` subclasses this
        is a ``numpy.ndarray`` of shape ``(H, W, 4)`` and dtype ``uint8``.

    Raises
    ------
    KeyError
        If *name* is not registered.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural.registry import get

        set_world_seed(1337)
        arr = get("wasteland_ground")          # 256×256 RGBA uint8
        arr2 = get("wasteland_ground")
        assert arr2 is arr                      # cached — same object
    """
    if name not in _registry:
        raise KeyError(
            f"ProceduralDef '{name}' is not registered. "
            "Did you import the module that defines and registers it?"
        )

    world_seed = _current_world_seed()
    p_digest = _params_digest(params)
    cache_key = (name, world_seed, p_digest)

    if cache_key in _cache:
        return _cache[cache_key]

    # Cache miss — generate fresh.
    from fire_engine.core.rng import for_domain  # noqa: PLC0415
    rng = for_domain("procedural", name, p_digest)
    result = _registry[name].generate(rng, **params)
    _cache[cache_key] = result
    return result


def clear_cache() -> None:
    """
    Discard all cached generated results.

    The registry of ``ProceduralDef`` instances is preserved; only the
    generated-output cache is flushed.  The next ``get()`` call for any name
    will re-generate from scratch.

    Call this:
    - After ``set_world_seed()`` if you need the cache to reflect the new seed
      immediately (the cache key includes the world seed, so stale entries are
      naturally skipped — ``clear_cache`` is an optimisation to reclaim memory).
    - In tests between independent test cases.

    Example
    -------
    ::

        from fire_engine.procedural.registry import clear_cache, get
        arr1 = get("wasteland_ground")
        clear_cache()
        arr2 = get("wasteland_ground")
        assert arr2 is not arr1   # freshly generated
    """
    _cache.clear()


def reset_registry() -> None:
    """
    Discard **both** the registered definitions and the cache.

    Use this only in tests that need a completely clean slate (e.g. when
    testing ``register`` itself without interference from other registered
    defs).  Do NOT call in production code — it will break any system that
    holds a reference to a cached result by name.

    Example
    -------
    ::

        from fire_engine.procedural.registry import reset_registry, register, get
        reset_registry()
        # Now register only the def you want to test…
    """
    _registry.clear()
    _cache.clear()
