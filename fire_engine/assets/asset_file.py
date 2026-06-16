"""Versioned ``.asset`` file IO — :func:`save_asset` / :func:`load_asset`.

Reads and writes the on-disk .asset format: UTF-8 JSON, ``indent=2``,
``sort_keys=True``, trailing newline — byte-stable so re-saving an unchanged
asset is a no-op git diff. The loader checks the ``fire_asset`` spec version and
migrates older revisions forward before building a :class:`Prefab`.

Docs: docs/systems/assets.md
"""

from __future__ import annotations

import json
import os
from typing import Any

from fire_engine.assets.constants import FIRE_ASSET_VERSION
from fire_engine.assets.prefab import Prefab
from fire_engine.assets.types import AssetError, AssetVersionError


def save_asset(path: str | os.PathLike[str], prefab: Prefab) -> None:
    """Write ``prefab`` to ``path`` as a .asset file (UTF-8, sorted, +newline).

    Output is byte-stable for a given prefab — sorted keys and a fixed layout, so
    re-saving unchanged content produces an identical file.

    Docs: docs/systems/assets.md
    """
    text = json.dumps(prefab.to_envelope(), indent=2, sort_keys=True, ensure_ascii=False)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.write("\n")


def load_asset(path: str | os.PathLike[str]) -> Prefab:
    """Read a .asset file, migrate it to the current spec, return a :class:`Prefab`.

    Raises:
        AssetError: if the file is missing or not valid .asset JSON.
        AssetVersionError: if the file's ``fire_asset`` is newer than this build.

    Docs: docs/systems/assets.md
    """
    try:
        with open(path, encoding="utf-8") as fh:
            env = json.load(fh)
    except FileNotFoundError as e:
        raise AssetError(f"no asset file at {path!r}") from e
    except (OSError, json.JSONDecodeError) as e:
        raise AssetError(f"unreadable asset file {path!r}: {e}") from e
    if not isinstance(env, dict):
        raise AssetError(f"asset file {path!r} is not a JSON object")
    return Prefab.from_envelope(_migrate(env))


def _migrate(env: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an envelope to :data:`FIRE_ASSET_VERSION`.

    v1 is the first version, so there are no upgrade steps yet — this is the seam
    where future ``while raw < FIRE_ASSET_VERSION: ...`` passes go. A
    newer-than-known file is a hard error (forward-incompatible).

    Docs: docs/systems/assets.md
    """
    raw = env.get("fire_asset")
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise AssetError(f"asset envelope missing integer 'fire_asset' version (got {raw!r})")
    if raw > FIRE_ASSET_VERSION:
        raise AssetVersionError(
            f"asset spec version {raw} is newer than this build supports ({FIRE_ASSET_VERSION})"
        )
    # Future migrations land here, one step per version bump.
    return env
