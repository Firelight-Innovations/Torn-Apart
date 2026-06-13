// Inspector webview (properties panel) — plain DOM, no three.js.
// Bundled to media/inspector.js (esbuild, IIFE). Lives in the fireEditor
// sidebar below the Hierarchy. Shows the selected scene object's Transform
// (name, kind/id, position, rotation as XYZ Euler degrees, scale) PLUS a
// Unity-style component stack (Mesh, Light, ... — built from the scene.catalog
// the host forwards) with Add / Remove Component. Commits edits back to the
// daemon via the extension host:
//   inbound : {type:"object", object: SceneObjectDTO | null}
//             {type:"catalog", catalog: ComponentTypeSpec[]}
//   outbound: {type:"rename", id, name}
//             {type:"setTransform", id, position, rotation, scale}
//             {type:"addComponent", id, componentType}
//             {type:"removeComponent", id, index}
//             {type:"setComponent", id, index, params?, enabled?}
// Rotation wire format is scalar-first (w,x,y,z); see inspectorMath.ts.
import { eulerDegToQuat, quatToEulerDeg } from "./inspectorMath";

declare function acquireVsCodeApi(): { postMessage(msg: unknown): void };
const vscode = acquireVsCodeApi();

interface ComponentDTO {
  type: string;
  enabled: boolean;
  params: Record<string, unknown>;
}

interface SceneObjectDTO {
  id: number;
  name: string;
  kind: string;
  parent: number | null;
  position: number[];
  rotation: number[];
  scale: number[];
  components?: ComponentDTO[];
}

interface FieldSpec {
  name: string;
  ui_type: "float" | "color" | "vec3" | "enum" | "bool";
  default: unknown;
  label?: string;
  min?: number;
  max?: number;
  choices?: string[];
}

interface ComponentTypeSpec {
  type: string;
  label: string;
  multiple: boolean;
  fields: FieldSpec[];
}

let current: SceneObjectDTO | null = null;
let uniformScale = true;
let catalog: ComponentTypeSpec[] = [];
let catalogByType = new Map<string, ComponentTypeSpec>();
let renderedSig = ""; // structural signature of the component sections shown

// --- build the static form once ---
document.body.innerHTML = `
<style>
  body { padding: 4px 8px; font: 12px var(--vscode-font-family); color: var(--vscode-foreground); }
  #empty { opacity: 0.65; padding: 12px 4px; }
  .row { display: flex; align-items: center; gap: 4px; margin: 3px 0; }
  .row > .lbl { width: 52px; flex: none; opacity: 0.85; }
  input[type="text"], input[type="number"], select {
    flex: 1; min-width: 0; box-sizing: border-box; padding: 2px 4px;
    background: var(--vscode-input-background); color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, transparent); border-radius: 2px;
  }
  input[type="number"] { width: 30%; }
  input[type="color"] { flex: none; width: 36px; height: 20px; padding: 0; border: none; background: none; }
  .kindline { opacity: 0.7; margin: 2px 0 8px; }
  .axis { width: 10px; flex: none; text-align: center; opacity: 0.8; }
  .x { color: #e0604f; } .y { color: #53d769; } .z { color: #4f9fe0; }
  #lock { margin-left: 2px; }
  .sect { margin-top: 10px; font-weight: 600; opacity: 0.9; }
  .comp { margin: 6px 0; border: 1px solid var(--vscode-panel-border, rgba(128,128,128,0.3));
          border-radius: 3px; }
  .comphdr { display: flex; align-items: center; gap: 6px; padding: 3px 6px;
             background: var(--vscode-sideBarSectionHeader-background, rgba(128,128,128,0.12)); }
  .comphdr .compEnabled { flex: none; }
  .complabel { flex: 1; font-weight: 600; }
  .compRemove { flex: none; cursor: pointer; background: none; border: none;
                color: var(--vscode-foreground); opacity: 0.6; font-size: 12px; }
  .compRemove:hover { opacity: 1; color: var(--vscode-errorForeground, #e0604f); }
  .comprows { padding: 4px 6px; }
  .comprows .flbl { width: 76px; flex: none; opacity: 0.85; }
  .muted { opacity: 0.55; padding: 2px 0; }
  #addbar { display: flex; gap: 4px; margin-top: 10px; }
  #addBtn { flex: none; cursor: pointer; padding: 2px 8px;
            background: var(--vscode-button-background); color: var(--vscode-button-foreground);
            border: none; border-radius: 2px; }
  #addBtn:disabled { opacity: 0.5; cursor: default; }
</style>
<div id="empty">No object selected.</div>
<div id="form" style="display:none">
  <div class="row"><span class="lbl">Name</span><input type="text" id="name" /></div>
  <div class="kindline"><span id="kind"></span> · id <span id="id"></span></div>
  <div class="sect">Transform</div>
  <div class="row"><span class="axis x">X</span><input type="number" step="0.1" id="px" />
                   <span class="axis y">Y</span><input type="number" step="0.1" id="py" />
                   <span class="axis z">Z</span><input type="number" step="0.1" id="pz" /></div>
  <div class="row"><span class="axis x">X</span><input type="number" step="1" id="rx" />
                   <span class="axis y">Y</span><input type="number" step="1" id="ry" />
                   <span class="axis z">Z</span><input type="number" step="1" id="rz" /></div>
  <div class="row"><span class="axis x">X</span><input type="number" step="0.1" id="sx" />
                   <span class="axis y">Y</span><input type="number" step="0.1" id="sy" />
                   <span class="axis z">Z</span><input type="number" step="0.1" id="sz" />
                   <label title="uniform scale"><input type="checkbox" id="lock" checked />🔗</label></div>
  <div class="sect">Components</div>
  <div id="components"></div>
  <div id="addbar"><select id="addType"></select><button id="addBtn">+ Add</button></div>
</div>`;

const el = (id: string) => document.getElementById(id) as HTMLInputElement;
const FIELD_IDS = ["name", "px", "py", "pz", "rx", "ry", "rz", "sx", "sy", "sz"] as const;

function num(id: string): number {
  const v = parseFloat(el(id).value);
  return Number.isFinite(v) ? v : 0;
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function rgbToHex(rgb: number[]): string {
  const h = (v: number) => Math.round(clamp01(v) * 255).toString(16).padStart(2, "0");
  return `#${h(rgb[0] ?? 0)}${h(rgb[1] ?? 0)}${h(rgb[2] ?? 0)}`;
}

function hexToRgb(hex: string): number[] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return [1, 1, 1];
  return [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255];
}

// --- Transform section (the intrinsic, non-removable component) ---
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
  const f = (v: number) => String(round3(v));
  set("px", f(obj.position[0])); set("py", f(obj.position[1])); set("pz", f(obj.position[2]));
  const e = quatToEulerDeg(obj.rotation as [number, number, number, number]);
  const fd = (v: number) => String(Math.round(v * 100) / 100);
  set("rx", fd(e[0])); set("ry", fd(e[1])); set("rz", fd(e[2]));
  set("sx", f(obj.scale[0])); set("sy", f(obj.scale[1])); set("sz", f(obj.scale[2]));
  renderComponents(obj);
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

// --- Component stack ---
function componentsSig(obj: SceneObjectDTO): string {
  const types = (obj.components ?? []).map((c) => c.type).join("|");
  const cat = catalog.map((s) => s.type).join(",");
  return `${types}::${cat}`;
}

function renderComponents(obj: SceneObjectDTO): void {
  const host = document.getElementById("components")!;
  const sig = componentsSig(obj);
  if (sig !== renderedSig) {
    const comps = obj.components ?? [];
    host.innerHTML = comps.length
      ? comps.map((c, i) => sectionHtml(c, i)).join("")
      : `<div class="muted">No components.</div>`;
    renderedSig = sig;
  }
  refreshAddOptions(obj);
  patchComponentValues(obj);
}

function sectionHtml(comp: ComponentDTO, i: number): string {
  const spec = catalogByType.get(comp.type);
  const label = spec?.label ?? comp.type;
  const fields = (spec?.fields ?? []).map((f) => fieldRow(i, f)).join("");
  const checked = comp.enabled === false ? "" : "checked";
  return `<div class="comp">
    <div class="comphdr">
      <input type="checkbox" id="comp-${i}-enabled" class="compEnabled" data-ci="${i}" ${checked} title="enabled" />
      <span class="complabel">${escapeHtml(label)}</span>
      <button class="compRemove" data-ci="${i}" title="remove component">✕</button>
    </div>
    <div class="comprows">${fields || '<div class="muted">No properties.</div>'}</div>
  </div>`;
}

function fieldRow(i: number, f: FieldSpec): string {
  const id = `comp-${i}-${f.name}`;
  const lbl = `<span class="flbl">${escapeHtml(f.label ?? f.name)}</span>`;
  const data = `data-ci="${i}" data-field="${escapeHtml(f.name)}" data-ftype="${f.ui_type}"`;
  if (f.ui_type === "color") {
    return `<div class="row">${lbl}<input type="color" id="${id}" ${data} /></div>`;
  }
  if (f.ui_type === "bool") {
    return `<div class="row">${lbl}<input type="checkbox" id="${id}" ${data} /></div>`;
  }
  if (f.ui_type === "enum") {
    const opts = (f.choices ?? [])
      .map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`)
      .join("");
    return `<div class="row">${lbl}<select id="${id}" ${data}>${opts}</select></div>`;
  }
  if (f.ui_type === "vec3") {
    const axes = [0, 1, 2]
      .map((a) => `<input type="number" step="0.1" id="${id}-${a}" ${data} data-axis="${a}" />`)
      .join("");
    return `<div class="row">${lbl}${axes}</div>`;
  }
  // float
  const step = stepFor(f);
  return `<div class="row">${lbl}<input type="number" step="${step}" id="${id}" ${data} /></div>`;
}

function stepFor(f: FieldSpec): string {
  if (f.max !== undefined && f.min !== undefined && f.max - f.min <= 4) return "0.05";
  return "0.1";
}

function patchComponentValues(obj: SceneObjectDTO): void {
  const focused = document.activeElement?.id;
  (obj.components ?? []).forEach((comp, i) => {
    const enId = `comp-${i}-enabled`;
    const en = document.getElementById(enId) as HTMLInputElement | null;
    if (en && enId !== focused) en.checked = comp.enabled !== false;
    const spec = catalogByType.get(comp.type);
    (spec?.fields ?? []).forEach((f) => {
      const v = comp.params?.[f.name];
      if (f.ui_type === "vec3") {
        [0, 1, 2].forEach((a) => {
          const id = `comp-${i}-${f.name}-${a}`;
          const inp = document.getElementById(id) as HTMLInputElement | null;
          if (inp && id !== focused) inp.value = String(round3(Number((v as number[])?.[a] ?? 0)));
        });
        return;
      }
      const id = `comp-${i}-${f.name}`;
      const inp = document.getElementById(id) as HTMLInputElement | null;
      if (!inp || id === focused) return;
      if (f.ui_type === "color") inp.value = rgbToHex((v as number[]) ?? [1, 1, 1]);
      else if (f.ui_type === "bool") inp.checked = Boolean(v);
      else if (f.ui_type === "enum") inp.value = String(v ?? f.default ?? "");
      else inp.value = String(round3(Number(v ?? 0)));
    });
  });
}

function refreshAddOptions(obj: SceneObjectDTO): void {
  const sel = document.getElementById("addType") as HTMLSelectElement | null;
  const bar = document.getElementById("addbar");
  const btn = document.getElementById("addBtn") as HTMLButtonElement | null;
  if (!sel || !bar || !btn) return;
  const present = new Set((obj.components ?? []).map((c) => c.type));
  const avail = catalog.filter((s) => s.multiple || !present.has(s.type));
  sel.innerHTML = avail
    .map((s) => `<option value="${escapeHtml(s.type)}">${escapeHtml(s.label)}</option>`)
    .join("");
  btn.disabled = avail.length === 0;
  bar.style.display = avail.length === 0 ? "none" : "";
}

// Event delegation for the (re-rendered) component sections.
const compHost = document.getElementById("components")!;
compHost.addEventListener("change", (ev) => {
  if (!current) return;
  const t = ev.target as HTMLElement;
  if (t.classList.contains("compEnabled")) {
    vscode.postMessage({
      type: "setComponent",
      id: current.id,
      index: Number(t.dataset.ci),
      enabled: (t as unknown as HTMLInputElement).checked,
    });
    return;
  }
  const ci = t.dataset.ci;
  const field = t.dataset.field;
  const ftype = t.dataset.ftype;
  if (ci === undefined || field === undefined) return;
  const index = Number(ci);
  const input = t as unknown as HTMLInputElement;
  let value: unknown;
  if (ftype === "color") value = hexToRgb(input.value);
  else if (ftype === "bool") value = input.checked;
  else if (ftype === "enum") value = input.value;
  else if (ftype === "vec3") {
    value = [0, 1, 2].map((a) => {
      const inp = document.getElementById(`comp-${index}-${field}-${a}`) as HTMLInputElement | null;
      const n = inp ? parseFloat(inp.value) : 0;
      return Number.isFinite(n) ? n : 0;
    });
  } else {
    const n = parseFloat(input.value);
    value = Number.isFinite(n) ? n : 0;
  }
  vscode.postMessage({ type: "setComponent", id: current.id, index, params: { [field]: value } });
});
compHost.addEventListener("keydown", (ev) => {
  if ((ev as KeyboardEvent).key === "Enter") (ev.target as HTMLElement).blur();
});
compHost.addEventListener("click", (ev) => {
  if (!current) return;
  const t = ev.target as HTMLElement;
  if (t.classList.contains("compRemove")) {
    vscode.postMessage({ type: "removeComponent", id: current.id, index: Number(t.dataset.ci) });
  }
});

document.getElementById("addBtn")!.addEventListener("click", () => {
  if (!current) return;
  const sel = document.getElementById("addType") as HTMLSelectElement;
  if (sel.value) {
    vscode.postMessage({ type: "addComponent", id: current.id, componentType: sel.value });
  }
});

function setCatalog(types: ComponentTypeSpec[]): void {
  catalog = types ?? [];
  catalogByType = new Map(catalog.map((s) => [s.type, s]));
  renderedSig = ""; // force a structural rebuild now that fields are known
  if (current) renderComponents(current);
}

window.addEventListener("message", (event) => {
  const msg = event.data;
  if (msg?.type === "object") show((msg.object ?? null) as SceneObjectDTO | null);
  else if (msg?.type === "catalog") setCatalog((msg.catalog ?? []) as ComponentTypeSpec[]);
});

vscode.postMessage({ type: "ready" });
