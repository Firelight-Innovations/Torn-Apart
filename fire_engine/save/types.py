"""
save/types.py — Exception types for the save subsystem.

Groups the trivial exception types used by fire_engine.save so that
saveable.py can hold exactly one public class (the Saveable protocol).

Docs: docs/systems/save.md
"""

from __future__ import annotations


class SaveIncompatibleError(Exception):
    """
    Raised by ``SaveManager.load`` when a save file cannot be loaded safely.

    Triggers:
    - ``format_version`` in the file is newer than the engine supports.
    - ``world_seed`` in the save header does not match the current
      ``Config.world_seed``.
    - ``config_digest`` in the save header does not match the digest of the
      current ``Config``.

    When this exception is raised, **no partial load has occurred**: the
    engine state is unchanged (the save file was validated before any
    ``apply_delta`` calls).

    Attributes
    ----------
    message : str
        Human-readable explanation of the incompatibility.

    Example
    -------
        from fire_engine.save import SaveIncompatibleError

        try:
            save_manager.load("saves/quick.ta")
        except SaveIncompatibleError as exc:
            print(f"Cannot load save: {exc}")

    Docs: docs/systems/save.md
    """
