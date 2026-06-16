"""
save/save_manager.py — SaveManager: delta saves encoded as msgpack+zlib.

Overview
--------
``SaveManager`` is the single point of persistence for the entire game world.
It follows the pattern mandated by ARCHITECTURE.md §5.12 and §4a.4:

    SAVE:  header + per-system get_delta() → encode → zlib → disk  (atomic)
    LOAD:  read → validate header → reset clock → apply_delta per system

The world seed fully determines the procedural baseline.  Saves store only
*deviations from that baseline*, so an untouched world costs ~0 bytes of
delta storage.

On-disk Layout
--------------
A single msgpack-encoded outer envelope is written to disk (after zlib
compression of the per-system blobs):

    {
        "header": {
            "format_version": 1,           # int — bump when layout changes
            "world_seed":     <int>,        # must match Config.world_seed on load
            "config_digest":  <hex str>,    # blake2b of canonical config fields
            "game_clock":     <dict>        # Clock.get_state() plain dict
        },
        "systems": {
            "<save_key>": <bytes>,          # zlib(msgpack(encoded_delta))
            ...                             # one entry per registered Saveable
        }
    }

The outer dict itself is NOT zlib-compressed — only the per-system blobs are,
so the header remains cheaply readable.

Numpy & Tuple-Key Encoding
--------------------------
msgpack does not natively support numpy arrays or tuple dict-keys.  Two
transformations are applied before msgpack encoding and reversed after decoding:

1. **Numpy arrays** are encoded as a 3-element list:
       ["__ndarray__", dtype_str, shape_list, raw_bytes_b64]
   where ``raw_bytes_b64`` is the array's raw bytes encoded as a msgpack
   ``bytes`` object (msgpack supports raw bytes natively when ``use_bin_type``
   is True).  dtype_str is e.g. ``"uint8"``.

2. **Tuple dict-keys** (e.g. terrain's ``{(cx, cy, cz): array}``) cannot be
   JSON/msgpack keys.  The delta dict is therefore serialised as a list of
   ``[key, value]`` pairs:
       [[[0, 0, 0], <encoded_value>], [[1, 0, 2], <encoded_value>], ...]
   where each key-list is reconstructed into a tuple of ints on decode.

   Only the *top-level* dict of a system delta uses this pairing; nested dicts
   use normal msgpack string keys.  The distinction is made by wrapping the
   delta in a tagged envelope:
       {"__delta_type__": "kv_pairs", "pairs": [[key, value], ...]}
   If the top-level keys are all strings, the dict is encoded directly
   (backward-compatible with future string-keyed systems).

Config Digest
-------------
``config_digest`` is a lowercase hex ``blake2b`` (digest_size=16) of the
canonical string representation of the fields that, if changed, would make
the save incompatible:

    world_seed, voxel_size, chunk_size, light_grid_scale

The digest is computed as:
    blake2b(f"{world_seed}:{voxel_size}:{chunk_size}:{light_grid_scale}").hexdigest()

Debug flags (show_fps, show_chunk_borders, show_light_grid) and
view_distance_chunks are intentionally excluded — changing those does not
make a save file invalid.

Atomic Write
------------
``save(path)`` writes to ``<path>.tmp`` then calls ``os.replace(<path>.tmp, path)``
so the destination is never left in a half-written state (Hard Rule: atomic).

Example
-------
    from fire_engine.core import load_config, Clock, EventBus
    from fire_engine.core.rng import set_world_seed
    from fire_engine.world.terrain import ChunkManager
    from fire_engine.save import SaveManager

    cfg = load_config()
    set_world_seed(cfg.world_seed)
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    cm = ChunkManager(cfg, bus)

    sm = SaveManager(cfg, clock)
    sm.register(cm)              # register in ARCHITECTURE §4a.4 order

    # Save
    sm.save("saves/quick.ta")

    # Load (fresh world same seed — baseline regen happens inside apply_delta)
    sm.load("saves/quick.ta")

Docs: docs/systems/save.md
"""

from __future__ import annotations

import os
import zlib
from pathlib import Path
from typing import Any

import msgpack  # type: ignore[import-untyped]  # msgpack has no py.typed or stubs

from fire_engine.core.clock import Clock
from fire_engine.core.config import Config
from fire_engine.core.log import get_logger
from fire_engine.save._codec import (
    compute_config_digest,
    decode_delta,
    decode_value,
    encode_delta,
    encode_value,
)
from fire_engine.save.saveable import Saveable, SaveIncompatibleError

_log = get_logger("save.save_manager")

# Bump this only when the on-disk layout changes in a backward-incompatible way.
_FORMAT_VERSION: int = 1

# ---------------------------------------------------------------------------
# Re-exports for backward-compatible private import paths
# (tests historically import these private names directly from save_manager)
# ---------------------------------------------------------------------------
_compute_config_digest = compute_config_digest  # backward-compat re-export
_encode_delta = encode_delta  # backward-compat re-export
_decode_delta = decode_delta  # backward-compat re-export
_encode_value = encode_value  # backward-compat re-export
_decode_value = decode_value  # backward-compat re-export


class SaveManager:
    """
    Delta save manager: coordinates serialisation of registered Saveable systems.

    Saves a world as a seed + config header + compressed delta blobs, one per
    registered Saveable system.  The clock state lives in the header (per
    ARCHITECTURE.md §4a.4) and is the authoritative source of time on load.

    Parameters
    ----------
    config : Config
        The current engine config (provides ``world_seed`` and fields for the
        ``config_digest``).
    clock : Clock
        The game clock.  Its state is stored in the save header on ``save()``,
        and restored from the header on ``load()`` (authoritative).

    Attributes
    ----------
    config : Config
        Engine config reference.
    clock : Clock
        Game clock reference.

    Registration Order
    ------------------
    ``apply_delta`` is called on registered Saveables in registration order (the
    order ``register()`` was called), matching the ARCHITECTURE §4a.4 sequence
    diagram:
        1. terrain (ChunkManager)
        2. ai (stub)
        3. economy / politics (stubs)

    Wire-up in main.py
    ------------------
    After creating the engine systems, wire like this:

        sm = SaveManager(cfg, clock)
        sm.register(chunk_manager)   # "terrain" — always first
        # sm.register(ai_manager)    # "ai" — Phase 8+
        # sm.register(economy)       # "economy" — Phase 9+

        # F5:
        sm.save("saves/quick.ta")

        # F9 (must despawn/respawn chunks after):
        sm.load("saves/quick.ta")
        # After load, iterate cm.chunks (all now dirty) and re-upload meshes,
        # OR call cm.stream_frame() to let the normal streaming pipeline handle it.

    Example
    -------
    >>> from fire_engine.core import load_config, Clock, EventBus
    >>> from fire_engine.core.rng import set_world_seed
    >>> from fire_engine.world.terrain import ChunkManager
    >>> from fire_engine.save import SaveManager
    >>> cfg = load_config()
    >>> set_world_seed(cfg.world_seed)
    >>> bus = EventBus()
    >>> clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    >>> cm = ChunkManager(cfg, bus)
    >>> sm = SaveManager(cfg, clock)
    >>> sm.register(cm)
    >>> sm.save("/tmp/test.ta")
    >>> sm.load("/tmp/test.ta")

    Docs: docs/systems/save.md
    """

    def __init__(self, config: Config, clock: Clock) -> None:
        self.config = config
        self.clock = clock
        self._saveables: list[Saveable] = []

    def register(self, saveable: Saveable) -> None:
        """
        Register a Saveable system for delta persistence.

        Systems are applied in registration order on load (ARCHITECTURE §4a.4).
        Call once per system at boot, after the system is fully initialised.

        Parameters
        ----------
        saveable : Saveable
            Any object implementing the Saveable protocol (has ``save_key``,
            ``get_delta()``, ``apply_delta()``).

        Raises
        ------
        TypeError
            If ``saveable`` does not satisfy the Saveable protocol.

        Docs: docs/systems/save.md
        """
        if not isinstance(saveable, Saveable):
            raise TypeError(
                f"{type(saveable).__name__!r} does not implement the Saveable "
                "protocol (needs save_key, get_delta, apply_delta)."
            )
        self._saveables.append(saveable)
        _log.debug("Registered saveable %r", saveable.save_key)

    def save(self, path: str | os.PathLike[str]) -> None:
        """
        Save the world to disk as a header + compressed per-system delta blobs.

        The write is **atomic**: data is first written to ``<path>.tmp`` and
        then renamed to ``path`` via ``os.replace``.  The destination is never
        left in a half-written state.

        On-disk format (msgpack outer envelope, then per-system zlib blobs):

        .. code-block:: text

            {
                "header": {
                    "format_version": 1,
                    "world_seed":     <int>,
                    "config_digest":  <str>,    # blake2b hex, 32 chars
                    "game_clock":     <dict>     # Clock.get_state()
                },
                "systems": {
                    "<save_key>": <bytes>,       # zlib(msgpack(encoded_delta))
                    ...
                }
            }

        Parameters
        ----------
        path : str or Path
            Destination file path.  Parent directory must exist.

        Raises
        ------
        OSError
            If the parent directory is not writable.

        Docs: docs/systems/save.md
        """
        path = Path(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        # Build the header.
        header = {
            "format_version": _FORMAT_VERSION,
            "world_seed": self.config.world_seed,
            "config_digest": compute_config_digest(self.config),
            "game_clock": self.clock.get_state(),
        }

        # Encode each system delta.
        systems: dict[str, bytes] = {}
        for sv in self._saveables:
            delta = sv.get_delta()
            raw_msgpack = encode_delta(delta)
            compressed = zlib.compress(raw_msgpack)
            systems[sv.save_key] = compressed
            _log.debug(
                "save: %r → %d bytes (raw) / %d bytes (compressed)",
                sv.save_key,
                len(raw_msgpack),
                len(compressed),
            )

        envelope = {"header": header, "systems": systems}
        data = msgpack.packb(envelope, use_bin_type=True)

        # Atomic write: tmp then replace.
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp_path.write_bytes(data)
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        _log.info("Saved world to %s (%d bytes)", path, len(data))

    def load(self, path: str | os.PathLike[str]) -> None:
        """
        Load a save file, validate the header, and apply deltas.

        Validation is performed **before** any state change.  If validation
        fails a ``SaveIncompatibleError`` is raised and the engine state is
        completely unchanged (no partial load).

        Validation rules (raise ``SaveIncompatibleError`` if any fail):
        - ``format_version`` > ``_FORMAT_VERSION``: engine is too old.
        - ``world_seed`` != ``config.world_seed``: wrong world.
        - ``config_digest`` != current digest: geometry-affecting config changed.

        On success:
        1. ``clock.set_state(header["game_clock"])`` — clock is authoritative.
        2. For each registered Saveable (in registration order): if its
           ``save_key`` is present in the file, call ``apply_delta(decoded)``.
           If absent, the system retains its freshly-generated state.

        Parameters
        ----------
        path : str or Path
            Path to the ``.ta`` save file.

        Raises
        ------
        SaveIncompatibleError
            If the save cannot be loaded (version / seed / digest mismatch).
        FileNotFoundError
            If ``path`` does not exist.

        Docs: docs/systems/save.md
        """
        path = Path(path)
        raw_data = path.read_bytes()
        envelope = msgpack.unpackb(raw_data, raw=False)

        # Decode bytes keys at envelope level (msgpack may use bytes keys).
        if isinstance(envelope, dict):
            envelope = {(k.decode() if isinstance(k, bytes) else k): v for k, v in envelope.items()}

        header_raw = envelope.get("header", {})
        if isinstance(header_raw, dict):
            header: dict[str, Any] = {
                (k.decode() if isinstance(k, bytes) else k): v for k, v in header_raw.items()
            }
        else:
            header = {}

        systems_raw = envelope.get("systems", {})
        if isinstance(systems_raw, dict):
            systems: dict[str, bytes] = {
                (k.decode() if isinstance(k, bytes) else k): v for k, v in systems_raw.items()
            }
        else:
            systems = {}

        # --- Validate header BEFORE any state change ---
        saved_version = header.get("format_version", 0)
        if saved_version > _FORMAT_VERSION:
            raise SaveIncompatibleError(
                f"Save file format version {saved_version} is newer than this "
                f"engine supports (max {_FORMAT_VERSION}).  Please update the engine."
            )

        saved_seed = header.get("world_seed")
        if saved_seed != self.config.world_seed:
            raise SaveIncompatibleError(
                f"Save file world_seed={saved_seed!r} does not match the current "
                f"config world_seed={self.config.world_seed!r}.  The save was "
                "created for a different world."
            )

        saved_digest = header.get("config_digest")
        current_digest = compute_config_digest(self.config)
        if saved_digest != current_digest:
            raise SaveIncompatibleError(
                f"Save file config_digest={saved_digest!r} does not match the "
                f"current config digest={current_digest!r}.  A geometry-affecting "
                "config field (voxel_size, chunk_size, or light_grid_scale) has "
                "changed since this save was created."
            )

        # --- Validation passed — now apply state changes ---

        # 1. Restore clock from header (authoritative per §4a.4).
        clock_state_raw = header.get("game_clock", {})
        if isinstance(clock_state_raw, dict):
            clock_state: dict[str, Any] = {
                (k.decode() if isinstance(k, bytes) else k): v for k, v in clock_state_raw.items()
            }
        else:
            clock_state = {}
        self.clock.set_state(clock_state)

        # 2. Apply deltas in registration order.
        for sv in self._saveables:
            blob = systems.get(sv.save_key)
            if blob is None:
                _log.debug(
                    "load: save_key %r absent — system keeps baseline state",
                    sv.save_key,
                )
                continue
            decompressed = zlib.decompress(blob)
            delta = decode_delta(decompressed)
            sv.apply_delta(delta)
            _log.debug(
                "load: applied delta for %r (%d entries)",
                sv.save_key,
                len(delta) if isinstance(delta, dict) else -1,
            )

        _log.info("Loaded world from %s", path)
