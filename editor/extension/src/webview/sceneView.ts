// Scene View webview (EDITOR_PRD F1) — three.js WebGL viewport.
// Bundled to media/sceneView.js (esbuild, IIFE). Runs in the VS Code webview
// browser context; talks to the extension host via postMessage.
//
// - Z-up (THREE.Object3D.DEFAULT_UP = (0,0,1)), matching the engine.
// - Fly camera: WASD move, Q/E down/up, mouse-look on pointer lock, Shift = 5x
//   (same bindings spirit as the game).
// - Meshes arrive as decoded MESH payloads; positions are absolute world meters,
//   so every chunk mesh is attached at the origin (DECISIONS: world-space verts).
// - Vertex colours already bake greyscale x sunlight, so MeshBasicMaterial shows
//   the engine's lighting without double-shading.
import * as THREE from "three";
import { decodeMeshPayload, chunkKey } from "../protocol/meshPayload";

declare function acquireVsCodeApi(): {
  postMessage(msg: unknown): void;
  getState(): unknown;
  setState(s: unknown): void;
};

const vscode = acquireVsCodeApi();

THREE.Object3D.DEFAULT_UP = new THREE.Vector3(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x101418, 1);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 1.0)); // baked lighting is in vertex colors

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 4000);
camera.position.set(20, -20, 24);

// --- chunk geometry registry ---
const chunks = new Map<string, THREE.Mesh>();
const borders = new Map<string, THREE.LineSegments>();
let chunkMeters = 16;
let showWireframe = false;
let showBorders = false;
let totalVerts = 0;
let totalTris = 0;

const material = new THREE.MeshBasicMaterial({ vertexColors: true });
const borderMaterial = new THREE.LineBasicMaterial({ color: 0x3a8fd0 });

function upsertChunk(payload: Uint8Array): void {
  const m = decodeMeshPayload(payload);
  const key = chunkKey(m.coord);
  removeChunk(key);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(m.positions, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(m.normals, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(m.colors, 4));
  geo.setAttribute("uv", new THREE.BufferAttribute(m.uvs, 2));
  geo.setIndex(new THREE.BufferAttribute(m.indices, 1));

  const mesh = new THREE.Mesh(geo, material);
  scene.add(mesh);
  chunks.set(key, mesh);
  totalVerts += m.vertexCount;
  totalTris += m.indexCount / 3;

  if (showBorders) addBorder(key, m.coord);
  updateStats();
}

function removeChunk(key: string): void {
  const mesh = chunks.get(key);
  if (mesh) {
    const g = mesh.geometry as THREE.BufferGeometry;
    totalVerts -= g.getAttribute("position").count;
    const idx = g.getIndex();
    if (idx) totalTris -= idx.count / 3;
    scene.remove(mesh);
    g.dispose();
    chunks.delete(key);
  }
  const b = borders.get(key);
  if (b) {
    scene.remove(b);
    (b.geometry as THREE.BufferGeometry).dispose();
    borders.delete(key);
  }
}

function addBorder(key: string, coord: [number, number, number]): void {
  const size = chunkMeters;
  const min = new THREE.Vector3(coord[0] * size, coord[1] * size, coord[2] * size);
  const box = new THREE.Box3(min, min.clone().addScalar(size));
  const helper = new THREE.Box3Helper(box, 0x3a8fd0) as unknown as THREE.LineSegments;
  helper.material = borderMaterial;
  scene.add(helper);
  borders.set(key, helper);
}

function rebuildBorders(): void {
  for (const [key, b] of borders) {
    scene.remove(b);
    (b.geometry as THREE.BufferGeometry).dispose();
    borders.delete(key);
  }
  if (showBorders) {
    for (const key of chunks.keys()) {
      const [cx, cy, cz] = key.split(",").map(Number) as [number, number, number];
      addBorder(key, [cx, cy, cz]);
    }
  }
}

function clearAll(): void {
  for (const key of [...chunks.keys()]) removeChunk(key);
  totalVerts = 0;
  totalTris = 0;
  updateStats();
}

// --- fly camera ---
const keys = new Set<string>();
let yaw = Math.PI * 0.75; // facing toward origin-ish
let pitch = -0.5;
let pointerLocked = false;

function brushSettings(): { shape: string; mode: string; radius: number; material: number } {
  const val = (id: string) => (document.getElementById(id) as HTMLInputElement | HTMLSelectElement).value;
  return {
    shape: val("brushShape"),
    mode: val("brushMode"),
    radius: parseFloat(val("brushSize")),
    material: parseInt(val("brushMaterial"), 10),
  };
}

renderer.domElement.addEventListener("mousedown", (e) => {
  if (!pointerLocked) {
    renderer.domElement.requestPointerLock();
    return;
  }
  if (e.button !== 0) return; // left click = carve at the crosshair
  const o = camera.position;
  const d = forwardVector();
  vscode.postMessage({
    type: "edit",
    ox: o.x, oy: o.y, oz: o.z,
    dx: d.x, dy: d.y, dz: d.z,
    brush: brushSettings(),
  });
});
document.addEventListener("pointerlockchange", () => {
  pointerLocked = document.pointerLockElement === renderer.domElement;
});
document.addEventListener("mousemove", (e) => {
  if (!pointerLocked) return;
  yaw -= e.movementX * 0.0025;
  pitch -= e.movementY * 0.0025;
  pitch = Math.max(-1.5, Math.min(1.5, pitch));
});
window.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.code === "KeyZ") {
    vscode.postMessage({ type: "undo" });
    e.preventDefault();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && (e.code === "KeyY" || (e.shiftKey && e.code === "KeyZ"))) {
    vscode.postMessage({ type: "redo" });
    e.preventDefault();
    return;
  }
  keys.add(e.code);
  if (e.code === "KeyG") {
    showWireframe = !showWireframe;
    material.wireframe = showWireframe;
  }
  if (e.code === "KeyB") {
    showBorders = !showBorders;
    rebuildBorders();
  }
});
window.addEventListener("keyup", (e) => keys.delete(e.code));

function forwardVector(): THREE.Vector3 {
  // Z-up spherical -> direction. yaw about Z, pitch from horizon.
  const cp = Math.cos(pitch);
  return new THREE.Vector3(Math.cos(yaw) * cp, Math.sin(yaw) * cp, Math.sin(pitch));
}

let lastCenterSent = new THREE.Vector3(Infinity, Infinity, Infinity);
let lastTime = performance.now();
let fps = 0;

function tick(): void {
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;
  fps = fps * 0.9 + (1 / Math.max(dt, 1e-4)) * 0.1;

  const fwd = forwardVector();
  const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
  const speed = (keys.has("ShiftLeft") || keys.has("ShiftRight") ? 50 : 10) * dt;
  const move = new THREE.Vector3();
  if (keys.has("KeyW")) move.add(fwd);
  if (keys.has("KeyS")) move.sub(fwd);
  if (keys.has("KeyD")) move.add(right);
  if (keys.has("KeyA")) move.sub(right);
  if (keys.has("KeyE")) move.z += 1;
  if (keys.has("KeyQ")) move.z -= 1;
  if (move.lengthSq() > 0) camera.position.addScaledVector(move.normalize(), speed);

  camera.up.set(0, 0, 1);
  camera.lookAt(camera.position.clone().add(fwd));

  // Re-center streaming when the camera crosses ~half a chunk.
  if (camera.position.distanceTo(lastCenterSent) > chunkMeters * 0.5) {
    lastCenterSent = camera.position.clone();
    vscode.postMessage({
      type: "camera",
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
    });
  }

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

// --- stats overlay ---
const statsEl = document.getElementById("stats")!;
function updateStats(): void {
  statsEl.textContent =
    `chunks ${chunks.size}  verts ${totalVerts}  tris ${Math.round(totalTris)}  ` +
    `fps ${fps.toFixed(0)}  [G]wire [B]borders`;
}
setInterval(updateStats, 250);

// --- host messages ---
window.addEventListener("message", (event) => {
  const msg = event.data;
  switch (msg.type) {
    case "mesh": {
      const buf = msg.payload instanceof ArrayBuffer ? new Uint8Array(msg.payload) : (msg.payload as Uint8Array);
      upsertChunk(buf);
      break;
    }
    case "unload":
      removeChunk(chunkKey(msg.coord));
      break;
    case "config":
      if (msg.config && typeof msg.config.chunk_meters === "number") {
        chunkMeters = msg.config.chunk_meters;
      }
      break;
    case "reset":
      clearAll();
      break;
    case "editState": {
      const st = (msg.state ?? {}) as { edited_chunks?: number };
      const dirtyEl = document.getElementById("dirty");
      if (dirtyEl) {
        dirtyEl.textContent = st.edited_chunks ? `● ${st.edited_chunks} edited` : "";
      }
      break;
    }
  }
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

vscode.postMessage({ type: "ready" });
requestAnimationFrame(tick);
