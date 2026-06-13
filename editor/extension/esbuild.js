// Build script for the Fire Editor extension.
//   node esbuild.js              -> bundle the extension to dist/extension.js
//   node esbuild.js --watch      -> rebuild on change
//   node esbuild.js --production -> minified build
//   node esbuild.js --tests      -> also bundle node:test suites to dist/test/
const esbuild = require("esbuild");
const fs = require("fs");
const path = require("path");

const watch = process.argv.includes("--watch");
const production = process.argv.includes("--production");
const tests = process.argv.includes("--tests");

const base = {
  bundle: true,
  platform: "node",
  target: "node18",
  format: "cjs",
  sourcemap: true,
  logLevel: "info",
  external: ["vscode"], // provided by the VS Code runtime
};

function listTestEntries() {
  const dir = path.join(__dirname, "src", "test");
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".test.ts"))
    .map((f) => path.join(dir, f));
}

// Webview bundles run in the browser context: bundle three.js, no node
// externals. Multi-entry: each webview (3D viewport, inspector form) gets its
// own IIFE bundle in media/.
const webviewConfig = {
  bundle: true,
  platform: "browser",
  target: "es2020",
  format: "iife",
  sourcemap: true,
  logLevel: "info",
  entryPoints: [
    "src/webview/sceneView.ts",
    "src/webview/inspector.ts",
    "src/webview/harnessBoot.ts",
  ],
  outdir: "media",
  minify: production,
};

async function main() {
  const ctx = await esbuild.context({
    ...base,
    entryPoints: ["src/extension.ts"],
    outfile: "dist/extension.js",
    minify: production,
  });
  const webviewCtx = await esbuild.context(webviewConfig);

  if (tests) {
    await esbuild.build({
      ...base,
      entryPoints: listTestEntries(),
      outdir: "dist/test",
      // node: builtins (node:test, node:assert) stay external on platform=node.
    });
  }

  if (watch) {
    await ctx.watch();
    await webviewCtx.watch();
    console.log("[esbuild] watching…");
  } else {
    await ctx.rebuild();
    await webviewCtx.rebuild();
    await ctx.dispose();
    await webviewCtx.dispose();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
