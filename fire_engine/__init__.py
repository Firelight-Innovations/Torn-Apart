"""Torn Apart — custom headless engine on Panda3D (rendering SDK only).

See ``docs/ARCHITECTURE.md`` (design authority) and ``docs/DEVELOPMENT_PLAN.md``
(sequencing authority). Only ``world/`` and ``lighting/`` may import panda3d.
"""

# Engine version. The Fire Editor daemon asserts compatibility against this at
# boot (EDITOR_PRD §5.6). Bump the minor when public engine APIs the editor
# binds to change in a backward-compatible way; bump the major on a break.
__version__ = "0.1.0"
