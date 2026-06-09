"""
tools/dump_save.py — Human-readable inspection of a Torn Apart save file.

Usage
-----
    python tools/dump_save.py <save_file>

Output
------
Prints:
  - Save file path and total size on disk.
  - Header: format version, world seed, config digest, game clock state.
  - Per-system: save key, number of delta entries, compressed size, and
    uncompressed msgpack size (so you can see how well zlib is compressing).

This tool never modifies the save file.  It is the "viewable save" requirement
from ARCHITECTURE.md §4a.4.

No panda3d imports — runs headlessly from the repo root.

Example output
--------------
    Save file: saves/quick.ta  (1234 bytes on disk)

    === HEADER ===
    format_version : 1
    world_seed     : 1337
    config_digest  : a3b4c5d6e7f8a1b2c3d4e5f6a7b8c9d0
    game_clock:
      game_day         : 0
      game_time_of_day : 0.0
      total_real_time  : 12.34
      accumulator      : 0.0

    === SYSTEMS ===
    [terrain]
      delta entries : 3
      compressed    : 1 024 bytes
      uncompressed  : 98 304 bytes
      compression   : 98.96 %
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

# Ensure the repo root is on sys.path so torn_apart imports work.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import msgpack  # noqa: E402 (after sys.path fixup)

from torn_apart.save.save_manager import _decode_delta  # noqa: E402


def _decode_bytes_keys(obj):
    """Recursively decode bytes keys in a dict (msgpack may use bytes)."""
    if isinstance(obj, dict):
        return {
            (k.decode() if isinstance(k, bytes) else k): _decode_bytes_keys(v)
            for k, v in obj.items()
        }
    return obj


def dump(path: str | Path) -> None:
    """
    Print a human-readable summary of the save file at ``path``.

    Parameters
    ----------
    path : str or Path
        Path to a ``.ta`` save file produced by ``SaveManager.save()``.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    path = Path(path)
    raw_data = path.read_bytes()
    total_bytes = len(raw_data)

    envelope = msgpack.unpackb(raw_data, raw=False)
    envelope = _decode_bytes_keys(envelope)

    header = _decode_bytes_keys(envelope.get("header", {}))
    systems = _decode_bytes_keys(envelope.get("systems", {}))

    print(f"\nSave file: {path}  ({total_bytes:,} bytes on disk)")

    # ---- Header ----
    print("\n=== HEADER ===")
    print(f"  format_version : {header.get('format_version', '?')}")
    print(f"  world_seed     : {header.get('world_seed', '?')}")
    print(f"  config_digest  : {header.get('config_digest', '?')}")
    clock_state = header.get("game_clock", {})
    if isinstance(clock_state, dict):
        clock_state = _decode_bytes_keys(clock_state)
    print("  game_clock:")
    for k, v in (clock_state.items() if isinstance(clock_state, dict) else []):
        print(f"    {k:<20} : {v}")

    # ---- Systems ----
    print("\n=== SYSTEMS ===")
    if not systems:
        print("  (no systems saved)")
        return

    for save_key, compressed_blob in systems.items():
        if not isinstance(compressed_blob, (bytes, bytearray)):
            print(f"  [{save_key}]  (not a bytes blob — unexpected)")
            continue

        compressed_size = len(compressed_blob)

        try:
            decompressed = zlib.decompress(compressed_blob)
            uncompressed_size = len(decompressed)
        except zlib.error as exc:
            print(f"  [{save_key}]  ERROR decompressing: {exc}")
            continue

        try:
            delta = _decode_delta(decompressed)
            entry_count = len(delta) if isinstance(delta, dict) else "?"
        except Exception as exc:
            entry_count = f"ERROR decoding: {exc}"

        compression_pct = (
            (1.0 - compressed_size / uncompressed_size) * 100.0
            if uncompressed_size > 0
            else 0.0
        )

        print(f"\n  [{save_key}]")
        print(f"    delta entries  : {entry_count}")
        print(f"    compressed     : {compressed_size:>10,} bytes")
        print(f"    uncompressed   : {uncompressed_size:>10,} bytes")
        print(f"    compression    : {compression_pct:.2f} %")


def main() -> None:
    """Entry point for ``python tools/dump_save.py <file>``."""
    if len(sys.argv) < 2:
        print("Usage: python tools/dump_save.py <save_file>", file=sys.stderr)
        sys.exit(1)
    dump(sys.argv[1])


if __name__ == "__main__":
    main()
