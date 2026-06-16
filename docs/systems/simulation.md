# simulation — System Doc
keywords: simulation, agency, macro-simulation, Layer-4, Layer-5, NPC, ai, economy, politics, player, grouping package, stub, agent, faction, good, tradeable, archetype

> One doc per code package; filename matches the package exactly (`docs/systems/simulation.md` ↔ `fire_engine/simulation/`).

## Role

`simulation/` is the **grouping package** for the engine's Layer-4/5 agency and macro-simulation systems (ARCHITECTURE.md §5.8–5.11).  It exports nothing of its own; all real APIs live in its sub-packages.

**Current milestone state:** all sub-packages except `player/` are session-1 stubs — the classes exist with the correct public interface so the rest of the engine can import and register them, but behaviour is not yet implemented (methods raise `NotImplementedError`).

Sub-packages:

| Sub-package | Purpose |
|---|---|
| `simulation.ai` | 3-tier NPC AI (Active / Regional / World-Map). |
| `simulation.economy` | Per-settlement supply/demand pricing and trade routes. |
| `simulation.politics` | Faction ownership, allegiance, and world-map events. |
| `simulation.player` | Human-control layer: same interface as an NPC agent. |

`simulation/` deliberately does NOT: export any symbol from `__init__.py`; import panda3d (fully headless-testable); or implement any behaviour beyond what is noted in each sub-package.

## Public API

`fire_engine.simulation` itself exports nothing (`__all__` is not defined; `__init__.py` only holds the module docstring).  Import sub-package APIs directly:

```python
from fire_engine.simulation.player import FlyController
from fire_engine.simulation.ai import NPCArchetype
from fire_engine.simulation.economy import GoodDef
from fire_engine.simulation.politics import FactionDef
```

## Imports Allowed

Per ARCHITECTURE.md §4a.2, `simulation/` as a grouping package has no direct imports.  Each sub-package observes its own import rules — see the individual sub-package docs:

- `simulation.ai` → `world`, `procedural`, `core`
- `simulation.economy` → `procedural`, `core`
- `simulation.politics` → `procedural`, `core`
- `simulation.player` → `render`, `core`

No panda3d imports anywhere in `simulation/`.

## Events

### Published
Sub-packages publish their own events.  None are published from `simulation/__init__.py` itself.

### Subscribed
None at this level.

## Units & Invariants

- All sub-packages are headless (no panda3d imports).
- All randomness in sub-packages must use `core.rng.for_domain(*keys)` — never `random.*` or unseeded `np.random.*`.
- Stub methods raise `NotImplementedError` with a message pointing at the relevant ARCHITECTURE.md section.

## Examples

```python
# Import sub-package APIs directly — never import from simulation itself.
from fire_engine.simulation.player import FlyController
from fire_engine.simulation.ai import NPCArchetype
from fire_engine.simulation.economy import GoodDef
from fire_engine.simulation.politics import FactionDef
```

## Gotchas

1. **`simulation/` is a grouping package only** — it has no public API of its own.  Always import from a sub-package (`simulation.player`, `simulation.ai`, etc.).
2. **Sub-packages are stubs** (except `player/`).  Calling `.generate()` on `NPCArchetype`, `GoodDef`, or `FactionDef` raises `NotImplementedError` until the relevant session implements them.
3. **Player is the exception** — `simulation.player.FlyController` is fully implemented in Session 1.
