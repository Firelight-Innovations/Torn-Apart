# simulation.economy — System Doc
keywords: economy, economic, goods, gooddef, tradeable, trade, supply, demand, pricing, price, settlement, trade route, arbitrage, manager, hired manager, market, stub, generate

> One doc per code package; filename matches the package exactly (`docs/systems/simulation.economy.md` ↔ `fire_engine/simulation/economy/`).

## Role

`simulation/economy/` is the **Economy API** (Layer 4 — ARCHITECTURE.md §5.9).  It will implement per-settlement supply/demand pricing for all tradeable goods in the world:

- Prices vary by settlement location, current supply, faction technology level, and trade-route connectivity.
- NPCs and the player use the **identical pricing path** — the player has no market privilege.
- Trade routes allow merchants and hired managers to arbitrage price differences across settlements.

**Current milestone state (Session 1 stub):** only `GoodDef` exists, as the authoring interface that economy content scripts subclass.  All methods raise `NotImplementedError`.  Settlement markets, price simulation, and trade-route logic are future scope.

`simulation/economy/` deliberately does NOT: import panda3d; issue render commands; manage UI or inventory directly (those are player/render concerns).

## Public API

All symbols below are re-exported from `fire_engine.simulation.economy` (`__init__.py`).

| Symbol | Description |
|---|---|
| `GoodDef` | Base class (stub) for tradeable good definitions. Subclass and implement `generate(rng, **params)` to define a named good (name, base value, weight, stack size, faction-specific modifiers). |
| `GoodDef.generate(rng, **params) -> None` | Generate a good definition instance from `rng` and keyword parameters. Raises `NotImplementedError` in Session 1. |

## Imports Allowed

Per ARCHITECTURE.md §4a.2, `simulation/economy/` may import:

- `fire_engine.procedural` (noise, registry — for content generation)
- `fire_engine.core` (Config, EventBus, `for_domain`, get_logger)
- `numpy`, Python standard library

**No panda3d imports.** Never import from `render/`, `lighting/`, `world/`, or any layer not listed above.

## Events

### Published
None (stub).  The full economy system will publish events such as `PriceChangedEvent`, `TradeRouteOpenedEvent` (future sessions).

### Subscribed
None (stub).  The full system will subscribe to `FactionWarEvent` / political events (war disrupts trade routes) and `SettlementDestroyedEvent`.

## Units & Invariants

- All randomness must use `core.rng.for_domain(*keys)` — never `random.*` or unseeded `np.random.*`.
- Prices should be deterministic for the same seed + world state — same seed must always produce the same initial market configuration.
- `GoodDef` content classes are Python scripts (like `ProceduralTextureDef`) — they are registered and callable by AI content agents.

## Examples

```python
# Authoring a new tradeable good (future session):
import numpy as np
from fire_engine.simulation.economy import GoodDef

class IronIngotDef(GoodDef):
    def generate(self, rng: np.random.Generator, **params) -> None:
        # Set base value, weight, faction modifiers...
        pass
```

```python
# Session 1 — importing the stub:
from fire_engine.simulation.economy import GoodDef
good = GoodDef()
# good.generate(rng) raises NotImplementedError until implemented.
```

## Gotchas

1. **Session 1 stub only** — `GoodDef.generate` raises `NotImplementedError`.  Full economy implementation is future scope (ARCHITECTURE.md §5.9).
2. **NPC/player parity** — the design requires NPCs and the player use identical market access.  Do not add special player-only pricing paths.
3. **Politics coupling** — war (from `simulation/politics/`) disrupts trade routes; implement the subscription to political events before enabling trade-route simulation.
4. **No panda3d imports** — the economy package is headless-testable.
