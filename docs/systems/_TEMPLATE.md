# <package> — System Doc
keywords: <comma-separated synonyms an agent might grep for — e.g. for terrain: voxel, chunk, mesh, mesher, brush, crater, heightmap, octree, raycast>

> One doc per code package; filename matches the package exactly (`docs/systems/world.terrain.md` ↔ `fire_engine/world/terrain/`).
> Every system doc uses these exact H2 headings, always in this order, so structured greps work:
> `grep -rA5 "## Events" docs/systems/` · `grep -rl "apply_brush" docs/` · `grep -A10 "## Units" docs/systems/world.terrain.md`

## Role
What this package does in one paragraph, and what it deliberately does NOT do.

## Public API
The exports from `__init__.py`, with one-line descriptions. Spell identifiers exactly as in code (`apply_brush`, not "the brush function") — greps must hit code and docs with the same query.

## Imports Allowed
Which packages this one may import (per the dependency map, ARCHITECTURE.md §4a.2). Anything else is a review failure.

## Events
Published: `*Event` types this package emits, and when.
Subscribed: events it listens to, and what it does in response.

## Units & Invariants
Meters/seconds/voxels, coordinate conventions, value ranges, determinism guarantees.

## Examples
Minimal working snippets (these are AI-agent prompt context — write for a reader who has never seen the codebase).

## Gotchas
Known traps, performance cliffs, ordering requirements.
