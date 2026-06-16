# simulation.politics — System Doc
keywords: politics, political, faction, factiondef, allegiance, morale, settlement, ownership, world map, world-map events, war, conflict, diplomacy, stub, generate

> One doc per code package; filename matches the package exactly (`docs/systems/simulation.politics.md` ↔ `fire_engine/simulation/politics/`).

## Role

`simulation/politics/` is the **Politics API** (Layer 4 — ARCHITECTURE.md §5.10).  It will implement faction-based political simulation across the world map:

- Factions own settlements and compete for territory through war, diplomacy, and economic pressure.
- Political events (wars, coups, faction collapses) are consumed by other systems:
  - **Economy** — war disrupts trade routes between settlements.
  - **AI** — faction allegiance and morale affect NPC behaviour and decision-making.
- The politics layer runs at world-map scale — decisions update infrequently (minutes of game time) via numpy array operations over all factions.

**Current milestone state (Session 1 stub):** only `FactionDef` exists, as the authoring interface that politics content scripts subclass.  All methods raise `NotImplementedError`.  Settlement ownership simulation, war logic, and event publishing are future scope.

`simulation/politics/` deliberately does NOT: import panda3d; issue render commands; directly control NPCs (it publishes events; AI subscribes).

## Public API

All symbols below are re-exported from `fire_engine.simulation.politics` (`__init__.py`).

| Symbol | Description |
|---|---|
| `FactionDef` | Base class (stub) for faction definitions. Subclass and implement `generate(rng, **params)` to define a named faction (name, ideology, starting settlements, relationships with other factions, tech level). |
| `FactionDef.generate(rng, **params) -> None` | Generate a faction definition instance from `rng` and keyword parameters. Raises `NotImplementedError` in Session 1. |

## Imports Allowed

Per ARCHITECTURE.md §4a.2, `simulation/politics/` may import:

- `fire_engine.procedural` (noise, registry — for content generation)
- `fire_engine.core` (Config, EventBus, `for_domain`, get_logger)
- `numpy`, Python standard library

**No panda3d imports.** Never import from `render/`, `lighting/`, `world/`, `simulation/ai/`, or `simulation/economy/` — politics publishes events upward; it never calls those layers directly.

## Events

### Published
None (stub).  The full politics system will publish:

| Event | When |
|---|---|
| `FactionWarStartedEvent(attacker, defender)` | Two factions enter open conflict. |
| `FactionWarEndedEvent(winner, loser)` | Conflict resolves. |
| `SettlementCapturedEvent(settlement, new_faction)` | Ownership changes. |
| `FactionCollapsedEvent(faction)` | Faction is eliminated from the map. |

### Subscribed
None (stub).  The full system may subscribe to economic collapse signals (future design — TBD).

## Units & Invariants

- All randomness must use `core.rng.for_domain(*keys)` — never `random.*` or unseeded `np.random.*`.
- Political state must be deterministic for the same seed + game time — same seed produces the same initial faction configuration.
- `FactionDef` content classes are Python scripts (like `ProceduralTextureDef`) — they are registered and callable by AI content agents.
- Politics updates run at world-map granularity — bulk numpy array operations over all factions, not per-faction Python loops.

## Examples

```python
# Authoring a new faction (future session):
import numpy as np
from fire_engine.simulation.politics import FactionDef

class WastelandRaidersFactionDef(FactionDef):
    def generate(self, rng: np.random.Generator, **params) -> None:
        # Set ideology, starting settlements, relationships, tech level...
        pass
```

```python
# Session 1 — importing the stub:
from fire_engine.simulation.politics import FactionDef
faction = FactionDef()
# faction.generate(rng) raises NotImplementedError until implemented.
```

## Gotchas

1. **Session 1 stub only** — `FactionDef.generate` raises `NotImplementedError`.  Full politics implementation is future scope (ARCHITECTURE.md §5.10).
2. **Publish events; never call AI or Economy directly** — politics sits at the same layer as AI and Economy.  Communicate sideways via the Event Bus only (ARCHITECTURE.md §4a rule 2).
3. **No panda3d imports** — the politics package is headless-testable.
4. **Faction-collapse is irreversible in Session 1 design** — eliminated factions do not respawn.  If you add respawn logic, record the decision in `DECISIONS.md`.
