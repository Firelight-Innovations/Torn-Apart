// Contract guard: every element id the viewport script dereferences via
// getElementById must exist in the shared markup (VIEWPORT_BODY_HTML), so the
// markup stays usable by BOTH the VS Code panel and the browser harness. The id
// list is extracted from the actual sceneView.ts source so it can't drift.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { VIEWPORT_BODY_HTML } from "../webview/viewportMarkup";

const SOURCE = join(process.cwd(), "src", "webview", "sceneView.ts");

function dereferencedIds(): string[] {
  const src = readFileSync(SOURCE, "utf8");
  const ids = new Set<string>();
  // Direct getElementById("...") and the brush palette's val("...") helper,
  // which reads form controls by id (val = (id) => getElementById(id).value).
  const patterns = [
    /getElementById\(\s*["'`]([^"'`]+)["'`]\s*\)/g,
    /\bval\(\s*["'`]([^"'`]+)["'`]\s*\)/g,
  ];
  for (const re of patterns) {
    let m: RegExpExecArray | null;
    while ((m = re.exec(src)) !== null) ids.add(m[1]);
  }
  return [...ids];
}

test("sceneView dereferences at least the known control ids", () => {
  const ids = dereferencedIds();
  // Sanity: the extraction found the palette/gizmo controls (not an empty set).
  for (const expected of ["stats", "gizmoMove", "brushShape"]) {
    assert.ok(ids.includes(expected), `expected sceneView to reference #${expected}`);
  }
});

test("every id sceneView dereferences exists in the shared markup", () => {
  for (const id of dereferencedIds()) {
    assert.ok(
      VIEWPORT_BODY_HTML.includes(`id="${id}"`),
      `VIEWPORT_BODY_HTML is missing #${id} (sceneView.ts dereferences it)`
    );
  }
});
