# devtools._tools — Private Implementation Sub-package
keywords: dev tool, DevTool, PerformanceTool, InspectorTool, ActionsTool, CallbackTool, ClockTool, tool base, tool plugin, panel builder

> Private internals sub-package created during standards remediation to satisfy the
> one-public-class-per-module rule while keeping the public surface of
> `fire_engine.devtools.tools` unchanged.  Import from `fire_engine.devtools.tools`
> or `fire_engine.devtools`, not directly from here.

## Role
Holds one module per concrete DevTool subclass so the structure gate is satisfied.
Does NOT define any new public API — every class here is re-exported via
`fire_engine.devtools.tools` and `fire_engine.devtools.__init__`.

## Public API
All exports are re-exported from `fire_engine.devtools.tools`:
- `DevTool` — base class for all dev-overlay panels (`base.py`)
- `PerformanceTool` — live engine stats read-out (`performance.py`)
- `InspectorTool` — editable inspector for the selected GameObject (`inspector.py`)
- `ActionsTool` — panel of one-shot action buttons (`actions.py`)
- `CallbackTool` — panel from a supplied build function (`callback.py`)
- `ClockTool` — game calendar read-out (`clock.py`)

## Imports Allowed
`fire_engine.devtools.types`, `fire_engine.devtools.enums`,
`fire_engine.devtools.introspect`, `fire_engine.devtools.selection`,
`fire_engine.core.*`.  No panda3d.

## Events
None — this sub-package contains no event publishers or subscribers.

## Units & Invariants
Same as parent package `fire_engine.devtools`: no panda3d, headless-testable,
no game-clock or RNG coupling.

## Examples
    # Do not import directly from _tools — use the public surface:
    from fire_engine.devtools import PerformanceTool, InspectorTool

## Gotchas
This is a private (`_`-prefixed) sub-package; its internal module paths may
change without notice.  Always import via `fire_engine.devtools.tools` or
`fire_engine.devtools`.
