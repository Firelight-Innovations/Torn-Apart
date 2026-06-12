// Inspector webview (properties panel) — plain DOM, no three.js.
// Bundled to media/inspector.js (esbuild, IIFE). Lives in the fireEditor
// sidebar below the Hierarchy. Shows the selected scene object's properties
// (name, kind/id, position, rotation as XYZ Euler degrees, scale) and commits
// edits back to the daemon via the extension host:
//   inbound : {type:"object", object: SceneObjectDTO | null}
//   outbound: {type:"rename", id, name}
//             {type:"setTransform", id, position?, rotation?, scale?}
// Rotation wire format is scalar-first (w,x,y,z); see inspectorMath.ts.
import { eulerDegToQuat, quatToEulerDeg } from "./inspectorMath";

declare function acquireVsCodeApi(): { postMessage(msg: unknown): void };
const vscode = acquireVsCodeApi();

interface SceneObjectDTO {
  id: number;
  name: string;
  kind: string;
  parent: number | null;
  position: number[];
  rotation: number[];
  scale: number[];
}

let current: SceneObjectDTO | null = null;
let uniformScale = true;

// --- build the static form once ---
document.body.innerHTML = `
<style>
  body { padding: 4px 8px; font: 12px var(--vscode-font-family); color: var(--vscode-foreground); }
  #empty { opacity: 0.65; padding: 12px 4px; }
  .row { display: flex; align-items: center; gap: 4px; margin: 3px 0; }
  .row > .lbl { width: 52px; flex: none; opacity: 0.85; }
  input[type="text"], input[type="number"] {
    flex: 1; min-width: 0; box-sizing: border-box; padding: 2px 4px;
    background: var(--vscode-input-background); color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, transparent); border-radius: 2px;
  }
  input[type="number"] { width: 30%; }
  .kindline { opacity: 0.7; margin: 2px 0 8px; }
  .axis { width: 10px; flex: none; text-align: center; opacity: 0.8; }
  .x { color: #e0604f; } .y { color: #53d769; } .z { color: #4f9fe0; }
  #lock { margin-left: 2px; }
  .sect { margin-top: 8px; font-weight: 600; opacity: 0.9; }
</style>
<div id="empty">No object selected.</div>
<div id="form" style="display:none">
  <div class="row"><span class="lbl">Name</span><input type="text" id="name" /></div>
  <div class="kindline"><span id="kind"></span> · id <span id="id"></span></div>
  <div class="sect">Position (m)</div>
  <div class="row"><span class="axis x">X</span><input type="number" step="0.1" id="px" />
                   <span class="axis y">Y</span><input type="number" step="0.1" id="py" />
                   <span class="axis z">Z</span><input type="number" step="0.1" id="pz" /></div>
  <div class="sect">Rotation (°)</div>
  <div class="row"><span class="axis x">X</span><input type="number" step="1" id="rx" />
                   <span class="axis y">Y</span><input type="number" step="1" id="ry" />
                   <span class="axis z">Z</span><input type="number" step="1" id="rz" /></div>
  <div class="sect">Scale <label title="uniform scale"><input type="checkbox" id="lock" checked />🔗</label></div>
  <div class="row"><span class="axis x">X</span><input type="number" step="0.1" id="sx" />
                   <span class="axis y">Y</span><input type="number" step="0.1" id="sy" />
                   <span class="axis z">Z</span><input type="number" step="0.1" id="sz" /></div>
</div>`;

const el = (id: string) => document.getElementById(id) as HTMLInputElement;
const FIELD_IDS = ["name", "px", "py", "pz", "rx", "ry", "rz", "sx", "sy", "sz"] as const;

function num(id: string): number {
  const v = parseFloat(el(id).value);
  return Number.isFinite(v) ? v : 0;
}

function show(obj: SceneObjectDTO | null): void {
  current = obj;
  document.getElementById("empty")!.style.display = obj ? "none" : "";
  document.getElementById("form")!.style.display = obj ? "" : "none";
  if (!obj) return;
  const focused = document.activeElement?.id;
  const set = (id: string, value: string) => {
    // Echo guard: never clobber the field the user is typing in.
    if (id !== focused) el(id).value = value;
  };
  set("name", obj.name);
  document.getElementById("kind")!.textContent = obj.kind;
  document.getElementById("id")!.textContent = String(obj.id);
  const f = (v: number) => String(Math.round(v * 1000) / 1000);
  set("px", f(obj.position[0])); set("py", f(obj.position[1])); set("pz", f(obj.position[2]));
  const e = quatToEulerDeg(obj.rotation as [number, number, number, number]);
  const fd = (v: number) => String(Math.round(v * 100) / 100);
  set("rx", fd(e[0])); set("ry", fd(e[1])); set("rz", fd(e[2]));
  set("sx", f(obj.scale[0])); set("sy", f(obj.scale[1])); set("sz", f(obj.scale[2]));
}

function commit(changedId: string): void {
  if (!current) return;
  if (changedId === "name") {
    const name = el("name").value.trim();
    if (name && name !== current.name) {
      vscode.postMessage({ type: "rename", id: current.id, name });
    }
    return;
  }
  if (changedId.startsWith("s") && uniformScale) {
    // Uniform lock: the edited axis drives all three.
    const v = num(changedId);
    for (const id of ["sx", "sy", "sz"]) {
      if (id !== changedId && document.activeElement?.id !== id) el(id).value = String(v);
    }
    el("sx").value = el("sy").value = el("sz").value = String(v);
  }
  vscode.postMessage({
    type: "setTransform",
    id: current.id,
    position: [num("px"), num("py"), num("pz")],
    rotation: eulerDegToQuat([num("rx"), num("ry"), num("rz")]),
    scale: [num("sx"), num("sy"), num("sz")],
  });
}

for (const id of FIELD_IDS) {
  const input = el(id);
  input.addEventListener("change", () => commit(id));
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      commit(id);
      input.blur();
    }
  });
}
el("lock").addEventListener("change", () => {
  uniformScale = el("lock").checked;
});

window.addEventListener("message", (event) => {
  const msg = event.data;
  if (msg?.type === "object") show((msg.object ?? null) as SceneObjectDTO | null);
});

vscode.postMessage({ type: "ready" });
