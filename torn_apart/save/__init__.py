"""
torn_apart.save — Delta save / load subsystem.

Exports the public API for the save package.  Import from here rather than
from submodules directly.

Contents
--------
SaveManager
    The central persistence manager.  Register Saveables at boot, then call
    ``save(path)`` / ``load(path)`` for F5/F9 or autosave.

Saveable
    Runtime-checkable structural protocol that any saveable system implements
    (``save_key`` attribute, ``get_delta()`` → dict, ``apply_delta(dict)``).

SaveIncompatibleError
    Raised by ``SaveManager.load`` when a save file cannot be safely loaded
    (version newer than supported, world-seed mismatch, or config-digest
    mismatch).  No partial load occurs when this is raised.

Usage
-----
    from torn_apart.save import SaveManager, Saveable, SaveIncompatibleError
"""

from torn_apart.save.saveable import Saveable, SaveIncompatibleError
from torn_apart.save.save_manager import SaveManager

__all__ = [
    "SaveManager",
    "Saveable",
    "SaveIncompatibleError",
]
