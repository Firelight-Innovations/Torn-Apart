"""vulture_whitelist.py — names vulture must NOT report as dead code.

Vulture is static and cannot see names reached only through dynamic dispatch:
the Unity-style lifecycle (called by the engine's update loop by name) and the
``Saveable`` protocol (called by the save system by name). Referencing them here
marks them as "used". Add a name here ONLY when it is genuinely invoked
dynamically — never to silence real dead code.

Docs: docs/systems/standards.md#code-quality
"""

from typing import Any

_obj: Any = None

# --- Unity-style lifecycle hooks (invoked by name by the world update loop) ---
_obj.awake
_obj.start
_obj.update
_obj.fixed_update
_obj.late_update
_obj.on_enable
_obj.on_disable
_obj.on_destroy

# --- Saveable protocol (invoked by name by the save system) ---
_obj.get_delta
_obj.apply_delta
_obj.mark_baseline
