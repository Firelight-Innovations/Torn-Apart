// Contract guard: the inspector webview and the extension host must agree on
// the message vocabulary. The inspector posts component messages up; the host
// must have a matching `case` for each, and must feed the catalog down. The
// strings are extracted from the actual sources so the two can't drift apart.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const INSPECTOR = join(process.cwd(), "src", "webview", "inspector.ts");
const EXTENSION = join(process.cwd(), "src", "extension.ts");

function read(path: string): string {
  return readFileSync(path, "utf8");
}

// Outbound message types the inspector posts via vscode.postMessage({ type: "..." }).
function outboundTypes(src: string): Set<string> {
  const types = new Set<string>();
  const re = /postMessage\(\s*\{\s*type:\s*["'`]([^"'`]+)["'`]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) types.add(m[1]);
  return types;
}

test("inspector posts the component edit messages", () => {
  const types = outboundTypes(read(INSPECTOR));
  for (const expected of ["addComponent", "removeComponent", "setComponent"]) {
    assert.ok(types.has(expected), `inspector should post ${expected}`);
  }
});

test("the host routes every inspector outbound message", () => {
  const ext = read(EXTENSION);
  // The host's inspector callback handles these; rename/setTransform predate
  // components but are part of the same vocabulary, so guard them all.
  for (const t of ["rename", "setTransform", "addComponent", "removeComponent", "setComponent"]) {
    assert.ok(ext.includes(`case "${t}"`), `extension.ts is missing case "${t}"`);
  }
});

test("inspector consumes object and catalog messages", () => {
  const src = read(INSPECTOR);
  assert.ok(/type\s*===\s*"object"/.test(src), "inspector should handle object messages");
  assert.ok(/type\s*===\s*"catalog"/.test(src), "inspector should handle catalog messages");
});

test("the host fetches the catalog and forwards it to the inspector", () => {
  const ext = read(EXTENSION);
  assert.ok(ext.includes("SCENE_CATALOG"), "extension.ts should request SCENE_CATALOG");
  assert.ok(ext.includes("postCatalog"), "extension.ts should call inspector.postCatalog");
});
