# Torn Apart — Engine Docs

Grep-first knowledge base for the **Firelight engine**. Search here before reading code.

- **[ARCHITECTURE](ARCHITECTURE.md)** — design authority.
- **[DEVELOPMENT_PLAN](DEVELOPMENT_PLAN.md)** — sequencing authority.
- **[systems/](systems/standards.md)** — one doc per code package (mirrors the package tree).
  Every system doc uses the same H2 schema: Role / Public API / Imports Allowed /
  Events / Units & Invariants / Examples / Gotchas.
- **[content/](content/tree_species_authoring.md)** — authoring guides for AI content agents.

## Standards gate

The repo enforces a machine-checked code-quality, structure, docs & testing
gate. See **[systems/standards.md](systems/standards.md)** for the full rule set,
toolchain, and how to run it (`pytest -q tests/standards/`).

This site is built with `mkdocs build --strict`, which doubles as the gate's
broken-link / dead-nav checker.
