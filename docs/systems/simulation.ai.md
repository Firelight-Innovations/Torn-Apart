# simulation.ai — System Doc
keywords: ai, npc, agent, archetype, npcarchetype, tier, active tier, regional tier, world map tier, promotion, demotion, 10k agents, world map pass, numpy pass, simulation, headless, stub, generate

> One doc per code package; filename matches the package exactly (`docs/systems/simulation.ai.md` ↔ `fire_engine/simulation/ai/`).

## Role

`simulation/ai/` is the **NPC AI package** (Layer 4 — ARCHITECTURE.md §5.8).  It will implement 3-tier simulation for 10,000+ named agents:

- **Active tier** — full agent update for NPCs within a few chunks of the player (schedules, pathfinding, dialogue).
- **Regional tier** — simplified update for NPCs in the loaded world but not near the player.
- **World-Map tier** — single bulk numpy pass over *all* agents in the world (never 10k Python objects); NPCs accumulate resources, travel, fight, and die as array operations.

Promotion/demotion between tiers is driven by player proximity.

**Current milestone state (Session 1 stub):** only `NPCArchetype` exists, as the authoring interface that AI content scripts subclass.  All methods raise `NotImplementedError`.  The tier machinery and per-agent update logic are future scope.

`simulation/ai/` deliberately does NOT: import panda3d; issue render commands; read terrain directly (it receives queries through the world layer's public API).

## Public API

All symbols below are re-exported from `fire_engine.simulation.ai` (`__init__.py`).

| Symbol | Description |
|---|---|
| `NPCArchetype` | Base class (stub) for procedural NPC character type definitions. Subclass and implement `generate(rng, **params)` to define a named archetype (skills, backstory, schedule template, faction allegiance). |
| `NPCArchetype.generate(rng, **params) -> None` | Generate an archetype instance from `rng` and keyword parameters. Raises `NotImplementedError` in Session 1. |

## Imports Allowed

Per ARCHITECTURE.md §4a.2, `simulation/ai/` may import:

- `fire_engine.world` (terrain, weather — for agent queries)
- `fire_engine.procedural` (noise, registry — for content generation)
- `fire_engine.core` (Config, EventBus, `for_domain`, Vec3, get_logger)
- `numpy`, Python standard library

**No panda3d imports.** Never import from `render/`, `lighting/`, or any layer above `simulation/`.

## Events

### Published
None (stub).  The full AI system will publish events such as `NPCDiedEvent`, `NPCFactionChangedEvent` (future sessions).

### Subscribed
None (stub).  The full system will subscribe to `TerrainEditedEvent` (for pathfinding invalidation) and political events (for allegiance changes).

## Units & Invariants

- All randomness must use `core.rng.for_domain(*keys)` — never `random.*` or unseeded `np.random.*`.
- The World-Map tier must be a **single array-based numpy pass** over all agents — never 10k Python objects in a loop (Hard Rule 4).
- NPCArchetype content classes are Python scripts (like `ProceduralTextureDef`) — they are registered and callable by AI content agents.

## Examples

```python
# Authoring a new NPC archetype (future session):
import numpy as np
from fire_engine.simulation.ai import NPCArchetype

class BanditArchetype(NPCArchetype):
    def generate(self, rng: np.random.Generator, **params) -> None:
        # Generate skills, backstory, schedule, faction allegiance...
        pass
```

```python
# Session 1 — importing the stub:
from fire_engine.simulation.ai import NPCArchetype
archetype = NPCArchetype()
# archetype.generate(rng) raises NotImplementedError until implemented.
```

## Gotchas

1. **Session 1 stub only** — `NPCArchetype.generate` raises `NotImplementedError`.  Full AI implementation is future scope (ARCHITECTURE.md §5.8).
2. **World-Map tier must be vectorised** — the design requires a single numpy array pass over all world agents, not a Python loop.  When implementing, profile array shapes before adding any per-agent Python logic.
3. **No panda3d imports** — the AI package is headless-testable.  All rendering of NPC state (health bars, speech bubbles) is handled by `render/` reading from the AI layer's public data.
