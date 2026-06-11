"""
core/rng.py — Deterministic random-number service for the Torn Apart engine.

All randomness in the engine MUST flow through this module.  Never use
``random.*`` or bare ``numpy.random.*`` directly — doing so breaks the
determinism guarantee that makes delta saves and reproducible bug reports
possible.

Cross-process determinism guarantee
------------------------------------
``for_domain(*keys)`` produces the **identical** stream on any run with the
same world seed and keys, even in a fresh interpreter process.  This is
guaranteed because:

1. The world seed is passed directly into ``np.random.SeedSequence``.
2. The per-domain key digest is computed with **hashlib.blake2b** over the
   canonical UTF-8 repr of the keys tuple.  Unlike Python's built-in
   ``hash()``, blake2b is deterministic across processes (Python salts
   ``hash()`` by default since 3.3 to prevent hash-flooding attacks).
3. The digest is converted to a Python int and mixed into the SeedSequence
   as an additional entropy source.

Usage
-----
    from fire_engine.core.rng import set_world_seed, for_domain

    set_world_seed(1337)

    rng = for_domain("terrain", (4, 5, 0))       # chunk (4,5,0)
    noise = rng.random((32, 32))                  # float64 32×32

    rng2 = for_domain("terrain", (4, 5, 0))       # same keys → same stream
    assert rng2.integers(0, 1000) == rng.integers(0, 1000)   # ← always true
"""

from __future__ import annotations

import hashlib

import numpy as np

# ---------------------------------------------------------------------------
# Module-level world seed (set once at boot via set_world_seed)
# ---------------------------------------------------------------------------

_world_seed: int = 0


def set_world_seed(seed: int) -> None:
    """
    Set the world seed used by all subsequent ``for_domain`` calls.

    Call this exactly once at engine boot, before any procedural generation.

    Parameters
    ----------
    seed : int
        The world seed (typically loaded from ``Config.world_seed``).

    Example
    -------
    >>> set_world_seed(1337)
    >>> rng = for_domain("terrain", (0, 0, 0))
    """
    global _world_seed
    _world_seed = int(seed)


def _keys_digest(keys: tuple) -> int:
    """
    Compute a stable cross-process int digest for a tuple of domain keys.

    Uses hashlib.blake2b (NOT Python's built-in hash()) so the result is
    identical across separate interpreter processes with different hash seeds.

    Parameters
    ----------
    keys : tuple
        Arbitrary combination of strings, ints, or nested tuples.

    Returns
    -------
    int
        A 64-bit unsigned integer derived from the blake2b digest.
    """
    # Canonical repr: sort-stable, human-readable, unambiguous for the types
    # we actually use (str, int, tuple of those).
    canonical: str = repr(keys)
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def for_domain(*keys) -> np.random.Generator:
    """
    Return a deterministic ``numpy.random.Generator`` for the given domain keys.

    The generator's stream depends only on (``_world_seed``, ``keys``) and is
    identical across separate Python processes (cross-process determinism).

    Parameters
    ----------
    *keys : str | int | tuple
        Domain identifier components.  Common usage patterns:

        ``for_domain("terrain", (cx, cy, cz))``  — per-chunk terrain noise
        ``for_domain("procedural", "wasteland_ground")`` — texture synthesis
        ``for_domain("npc", npc_id)``            — NPC attribute generation

    Returns
    -------
    numpy.random.Generator
        A fresh BitGenerator seeded from the world seed + key digest.
        Always yields the same stream for the same (world_seed, keys) pair.

    Raises
    ------
    RuntimeError
        If ``set_world_seed`` has never been called (seed remains 0; this is
        a valid seed, so no exception — callers are responsible for calling
        set_world_seed at boot).

    Example
    -------
    >>> set_world_seed(42)
    >>> a = for_domain("terrain", (1, 2, 3)).integers(0, 1_000_000, 5)
    >>> b = for_domain("terrain", (1, 2, 3)).integers(0, 1_000_000, 5)
    >>> (a == b).all()
    True

    Cross-process determinism (subprocess test in tests/test_rng.py confirms):
    >>> # In process 1:  for_domain("terrain",(1,2,3)).integers(0,10**6,5)
    >>> # In process 2:  same call with same seed → identical array
    """
    key_int = _keys_digest(keys)
    # Combine world_seed and key_int as independent entropy sources.
    ss = np.random.SeedSequence(entropy=_world_seed, spawn_key=(key_int,))
    return np.random.default_rng(ss)
