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
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { decodeMeshPayload, chunkKey } from "../protocol/meshPayload";
import { decodeTexturePayload } from "../protocol/texturePayload";
import { makeGroundMaterial } from "./groundMaterial";
import { host } from "./host";

THREE.Object3D.DEFAULT_UP = new THREE.Vector3(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x101418, 1);
document.body.appendChild(renderer.domElement);

// VS Code webviews only deliver keyboard events to the iframe once its document
// holds focus, and a bare <canvas> isn't focusable — so without this the WASD
// fly keys and the W/E/R/F/G/B/Esc hotkeys (all `window` keydown handlers below)
// silently never fire. Make the canvas focusable and claim focus on load and on
// every pointer interaction so keystrokes reach the viewport.
renderer.domElement.tabIndex = 0;
renderer.domElement.style.outline = "none";
const grabViewportFocus = (): void => renderer.domElement.focus({ preventScroll: true });
renderer.domElement.addEventListener("pointerdown", grabViewportFocus);
window.addEventListener("focus", grabViewportFocus);
grabViewportFocus();

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

// Boot fallback: greyscale baked-light vertex colours. Replaced by the
// procedural ground ShaderMaterial once the daemon ships the palette LUT
// (world.ground_lut → "groundLut" message).
const material = new THREE.MeshBasicMaterial({ vertexColors: true });
let groundMaterial: THREE.ShaderMaterial | null = null;
let terrainMaterial: THREE.Material = material;
let groundSeed = 0;
let groundTexelsPerM = 16;
const borderMaterial = new THREE.LineBasicMaterial({ color: 0x3a8fd0 });

function applyGroundLut(payload: Uint8Array): void {
  const tex = decodeTexturePayload(payload);
  groundMaterial?.dispose();
  groundMaterial = makeGroundMaterial(tex.data, tex.width, tex.height, {
    seed: groundSeed,
    texelsPerM: groundTexelsPerM,
  });
  groundMaterial.wireframe = showWireframe;
  updatePxRad();
  terrainMaterial = groundMaterial;
  for (const mesh of chunks.values()) mesh.material = terrainMaterial;
}

function updatePxRad(): void {
  if (!groundMaterial) return;
  // Vertical FOV radians / CSS pixels of viewport height = radians per pixel
  // (the same definition as the game's u_px_rad).
  groundMaterial.uniforms.u_px_rad.value =
    ((camera.fov * Math.PI) / 180) / renderer.domElement.clientHeight;
}

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

  const mesh = new THREE.Mesh(geo, terrainMaterial);
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

// --- object gizmos (Phase E2) ---
// Each placeable scene object is drawn as a coloured proxy mesh. Gizmos are
// parented to each other to mirror the hierarchy, so three.js composes world
// transforms for us. The selected object wears a yellow bounding box.
interface SceneObjectDTO {
  id: number;
  name: string;
  kind: string;
  parent: number | null;
  position: number[];
  rotation: number[];
  scale: number[];
}

const objectGroup = new THREE.Group();
scene.add(objectGroup);
const objectGizmos = new Map<number, THREE.Object3D>();
let selectedObjectId: number | null = null;

const selectionHelper = new THREE.BoxHelper(new THREE.Mesh(), 0xffcc33);
(selectionHelper.material as THREE.LineBasicMaterial).depthTest = false;
selectionHelper.renderOrder = 1000;
selectionHelper.visible = false;
scene.add(selectionHelper);

// --- transform gizmo (move / rotate / scale the selected object) ---
// three.js TransformControls: axis arrows + plane handles (translate),
// 3 rings (rotate), per-axis + uniform-center handles (scale). It edits the
// attached node's LOCAL transform under its hierarchy parent — exactly the
// semantics of scene.set_transform.
type GizmoMode = "translate" | "rotate" | "scale";
let draggingObjectId: number | null = null;
let lastTransformSent = 0;
const TRANSFORM_THROTTLE_MS = 80;

const tc = new TransformControls(camera, renderer.domElement);
tc.setSpace("local");
scene.add(tc);

function sendTransform(node: THREE.Object3D, force: boolean): void {
  const id = node.userData.id as number | undefined;
  if (id === undefined) return;
  const now = performance.now();
  if (!force && now - lastTransformSent < TRANSFORM_THROTTLE_MS) return;
  lastTransformSent = now;
  const q = node.quaternion; // three is (x,y,z,w); wire is (w,x,y,z)
  host.post({
    type: "transform",
    id,
    position: [node.position.x, node.position.y, node.position.z],
    rotation: [q.w, q.x, q.y, q.z],
    scale: [node.scale.x, node.scale.y, node.scale.z],
  });
}

tc.addEventListener("objectChange", () => {
  if (tc.object) {
    sendTransform(tc.object, false);
    refreshSelectionHelper();
  }
});
tc.addEventListener("dragging-changed", (e) => {
  if ((e as { value: unknown }).value) {
    draggingObjectId = selectedObjectId;
  } else {
    // Drag ended: send the exact final transform un-throttled; the resulting
    // scene.changed echo reconciles everyone.
    if (tc.object) sendTransform(tc.object, true);
    draggingObjectId = null;
  }
});

function setGizmoMode(mode: GizmoMode): void {
  tc.setMode(mode);
  for (const [btn, m] of [["gizmoMove", "translate"], ["gizmoRotate", "rotate"], ["gizmoScale", "scale"]] as const) {
    document.getElementById(btn)?.classList.toggle("active", m === mode);
  }
}

const KIND_COLOR: Record<string, number> = {
  empty: 0x9aa7b2,
  cube: 0x4f9fe0,
  sphere: 0xe0884f,
  light: 0xf2d54b,
  spawn: 0x53d769,
};

function makeGizmoMesh(kind: string): THREE.Mesh {
  let geo: THREE.BufferGeometry;
  switch (kind) {
    case "cube":
      geo = new THREE.BoxGeometry(1, 1, 1);
      break;
    case "sphere":
      geo = new THREE.SphereGeometry(0.6, 20, 14);
      break;
    case "light":
      geo = new THREE.IcosahedronGeometry(0.5, 0);
      break;
    case "spawn":
      geo = new THREE.ConeGeometry(0.5, 1.4, 16).rotateX(Math.PI / 2); // tip points +Z (up)
      break;
    default: // empty
      geo = new THREE.OctahedronGeometry(0.4, 0);
      break;
  }
  const mat = new THREE.MeshBasicMaterial({
    color: KIND_COLOR[kind] ?? 0xcccccc,
    transparent: true,
    opacity: kind === "empty" ? 0.55 : 0.9,
  });
  return new THREE.Mesh(geo, mat);
}

function disposeNode(node: THREE.Object3D): void {
  node.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.geometry) m.geometry.dispose();
    if (m.material) (m.material as THREE.Material).dispose();
  });
}

function setObjects(list: SceneObjectDTO[]): void {
  const ids = new Set(list.map((o) => o.id));
  for (const [id, node] of [...objectGizmos]) {
    if (!ids.has(id)) {
      node.parent?.remove(node);
      disposeNode(node);
      objectGizmos.delete(id);
    }
  }
  // tree() arrives DFS roots-first, so a parent gizmo always exists before its
  // children — safe to reparent in one pass.
  for (const o of list) {
    let node = objectGizmos.get(o.id);
    if (!node || node.userData.kind !== o.kind) {
      if (node) {
        node.parent?.remove(node);
        disposeNode(node);
      }
      node = makeGizmoMesh(o.kind);
      node.userData.kind = o.kind;
      node.userData.id = o.id;
      objectGizmos.set(o.id, node);
    }
    const desiredParent = o.parent != null ? objectGizmos.get(o.parent) ?? objectGroup : objectGroup;
    if (node.parent !== desiredParent) desiredParent.add(node);
    // Echo suppression: while a gizmo drag is live, our local node IS the
    // source of truth — daemon echoes of throttled mid-drag transforms must
    // not snap it backwards.
    if (o.id !== draggingObjectId) {
      node.position.set(o.position[0], o.position[1], o.position[2]);
      // stored quat is (w, x, y, z); three.js wants (x, y, z, w).
      node.quaternion.set(o.rotation[1], o.rotation[2], o.rotation[3], o.rotation[0]);
      node.scale.set(o.scale[0], o.scale[1], o.scale[2]);
    }
  }
  // Re-resolve the selection: detaches the gizmo if the object was deleted.
  selectObjectLocal(selectedObjectId);
}

function refreshSelectionHelper(): void {
  const node = selectedObjectId != null ? objectGizmos.get(selectedObjectId) : undefined;
  if (node) {
    objectGroup.updateMatrixWorld(true);
    selectionHelper.setFromObject(node);
    selectionHelper.visible = true;
  } else {
    selectionHelper.visible = false;
  }
}

function selectObjectLocal(id: number | null): void {
  selectedObjectId = id != null && objectGizmos.has(id) ? id : null;
  const node = selectedObjectId != null ? objectGizmos.get(selectedObjectId) : undefined;
  if (node) tc.attach(node);
  else tc.detach();
  refreshSelectionHelper();
}

function raycastObjects(): number | null {
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObjects(objectGroup.children, true);
  if (!hits.length) return null;
  let o: THREE.Object3D | null = hits[0].object;
  while (o && o.userData.id === undefined) o = o.parent;
  return o ? (o.userData.id as number) : null;
}

function frameObject(id: number): void {
  const node = objectGizmos.get(id);
  if (!node) return;
  const target = new THREE.Vector3();
  node.getWorldPosition(target);
  camera.position.copy(target).addScaledVector(forwardVector(), -10);
}

// --- Unity-style editor camera ---
// The cursor is FREE by default (no pointer lock), so it can hover the terrain
// for the brush preview and use the palette. Navigation mirrors the Unity Scene
// View:
//   • Right-drag        = look around; while held, WASD/QE fly, Shift = faster,
//                         scroll = adjust fly speed (flythrough mode).
//   • Middle-drag       = pan on the screen plane.
//   • Scroll            = dolly zoom along the view ray.
//   • Alt + Left-drag   = orbit around the hovered point.
//   • Left-click        = carve with the active brush at the hovered point.
const keys = new Set<string>();
let yaw = Math.PI * 0.75; // facing toward origin-ish
let pitch = -0.5;

type DragMode = null | "look" | "pan" | "orbit";
let dragMode: DragMode = null;
const orbitPivot = new THREE.Vector3();
let orbitDistance = 30;
let pivotDist = 30; // distance to whatever the cursor last hovered; scales pan/zoom

const LOOK_SENS = 0.0025;
const PAN_SPEED = 0.0016; // world meters per pixel, per meter of view distance
const ZOOM_SPEED = 0.0016;
let flySpeed = 14; // meters/sec while in flythrough, adjustable via scroll

const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

function brushSettings(): { shape: string; mode: string; radius: number; material: number } {
  const val = (id: string) => (document.getElementById(id) as HTMLInputElement | HTMLSelectElement).value;
  return {
    shape: val("brushShape"),
    mode: val("brushMode"),
    radius: parseFloat(val("brushSize")),
    material: parseInt(val("brushMaterial"), 10),
  };
}

// --- brush preview gizmo ---
// A wireframe shape that hugs the hovered terrain point, sized by brush radius
// and tinted by mode (green = add, red = remove). depthTest off so it reads
// through the surface like Unity's tool gizmos.
const gizmoMat = new THREE.MeshBasicMaterial({
  color: 0x53d769,
  wireframe: true,
  transparent: true,
  opacity: 0.9,
  depthTest: false,
});
const gizmo: THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial> = new THREE.Mesh(
  new THREE.SphereGeometry(1, 20, 14) as THREE.BufferGeometry,
  gizmoMat
);
gizmo.renderOrder = 999;
gizmo.visible = false;
scene.add(gizmo);
let gizmoShape = "";
let gizmoRadius = -1;

function buildGizmoGeo(shape: string, radius: number): THREE.BufferGeometry {
  switch (shape) {
    case "box":
      return new THREE.BoxGeometry(radius * 2, radius * 2, radius * 2);
    case "cylinder":
      // Cylinder is Y-up by default; rotate so its axis is world Z (up).
      return new THREE.CylinderGeometry(radius, radius, radius * 2, 24).rotateX(Math.PI / 2);
    default:
      return new THREE.SphereGeometry(radius, 24, 16);
  }
}

function updateGizmoAppearance(): void {
  const b = brushSettings();
  if (b.shape !== gizmoShape || Math.abs(b.radius - gizmoRadius) > 1e-3) {
    gizmo.geometry.dispose();
    gizmo.geometry = buildGizmoGeo(b.shape, b.radius);
    gizmoShape = b.shape;
    gizmoRadius = b.radius;
  }
  gizmoMat.color.setHex(b.mode === "add" ? 0x53d769 : 0xe0533d);
}

function pointerNDC(e: MouseEvent): void {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  ndc.y = -(((e.clientY - r.top) / r.height) * 2 - 1);
}

function raycastTerrain(): THREE.Intersection | null {
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObjects(Array.from(chunks.values()), false);
  return hits.length ? hits[0] : null;
}

function rightVector(): THREE.Vector3 {
  return new THREE.Vector3().crossVectors(forwardVector(), camera.up).normalize();
}

renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

renderer.domElement.addEventListener("mousedown", (e) => {
  pointerNDC(e);
  if (e.button === 2) {
    dragMode = "look";
    renderer.domElement.requestPointerLock();
    e.preventDefault();
  } else if (e.button === 1) {
    dragMode = "pan";
    e.preventDefault();
  } else if (e.button === 0) {
    // Transform-gizmo interaction wins: TransformControls sets `axis` while a
    // handle is hovered and `dragging` during a drag — never select/carve then.
    if (tc.dragging || tc.axis !== null) return;
    if (e.altKey) {
      const hit = raycastTerrain();
      orbitPivot.copy(hit ? hit.point : camera.position.clone().addScaledVector(forwardVector(), orbitDistance));
      orbitDistance = camera.position.distanceTo(orbitPivot);
      dragMode = "orbit";
    } else {
      // Selecting an object takes priority over carving: if a gizmo is under the
      // cursor, pick it; otherwise carve the terrain.
      const picked = raycastObjects();
      if (picked != null) {
        selectObjectLocal(picked);
        host.post({ type: "selectObject", id: picked });
        return;
      }
      // Carve: hand the daemon the ray through the cursor; it does the
      // authoritative terrain raycast + brush.
      raycaster.setFromCamera(ndc, camera);
      const o = camera.position;
      const d = raycaster.ray.direction;
      host.post({
        type: "edit",
        ox: o.x, oy: o.y, oz: o.z,
        dx: d.x, dy: d.y, dz: d.z,
        brush: brushSettings(),
      });
    }
  }
});

window.addEventListener("mouseup", () => {
  if (dragMode === "look") document.exitPointerLock();
  dragMode = null;
});
window.addEventListener("blur", () => {
  keys.clear();
  if (dragMode === "look") document.exitPointerLock();
  dragMode = null;
});

document.addEventListener("mousemove", (e) => {
  if (dragMode === "look") {
    yaw -= e.movementX * LOOK_SENS;
    pitch = clamp(pitch - e.movementY * LOOK_SENS, -1.5, 1.5);
  } else if (dragMode === "pan") {
    const k = PAN_SPEED * pivotDist;
    const right = rightVector();
    const up = new THREE.Vector3().crossVectors(right, forwardVector()).normalize();
    camera.position.addScaledVector(right, -e.movementX * k);
    camera.position.addScaledVector(up, e.movementY * k);
  } else if (dragMode === "orbit") {
    yaw -= e.movementX * LOOK_SENS;
    pitch = clamp(pitch - e.movementY * LOOK_SENS, -1.5, 1.5);
    camera.position.copy(orbitPivot).addScaledVector(forwardVector(), -orbitDistance);
  } else {
    // Free hover: drive the brush preview off the terrain under the cursor —
    // unless the transform gizmo owns the pointer (hover or drag).
    pointerNDC(e);
    if (tc.dragging || tc.axis !== null) {
      gizmo.visible = false;
      return;
    }
    const hit = raycastTerrain();
    if (hit) {
      pivotDist = hit.distance;
      gizmo.position.copy(hit.point);
      updateGizmoAppearance();
      gizmo.visible = true;
    } else {
      gizmo.visible = false;
    }
  }
});

renderer.domElement.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    if (dragMode === "look") {
      flySpeed = clamp(flySpeed * (e.deltaY < 0 ? 1.1 : 0.9), 1, 250);
    } else {
      const step = -e.deltaY * ZOOM_SPEED * Math.max(4, pivotDist);
      camera.position.addScaledVector(forwardVector(), step);
    }
  },
  { passive: false }
);

window.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.code === "KeyZ") {
    host.post({ type: "undo" });
    e.preventDefault();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && (e.code === "KeyY" || (e.shiftKey && e.code === "KeyZ"))) {
    host.post({ type: "redo" });
    e.preventDefault();
    return;
  }
  keys.add(e.code);
  // W/E/R switch the transform-gizmo mode (Unity bindings) — only when not
  // flying (the WASD fly keys are only consumed during a right-drag look).
  if (dragMode !== "look" && selectedObjectId != null) {
    if (e.code === "KeyW") setGizmoMode("translate");
    if (e.code === "KeyE") setGizmoMode("rotate");
    if (e.code === "KeyR") setGizmoMode("scale");
  }
  if (e.code === "KeyG") {
    showWireframe = !showWireframe;
    material.wireframe = showWireframe;
    if (groundMaterial) groundMaterial.wireframe = showWireframe;
  }
  if (e.code === "KeyB") {
    showBorders = !showBorders;
    rebuildBorders();
  }
  if (e.code === "KeyF" && selectedObjectId != null) {
    frameObject(selectedObjectId);
  }
  if (e.code === "Escape" && selectedObjectId != null) {
    selectObjectLocal(null);
    host.post({ type: "selectObject", id: null });
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
let lastFocusSent = 0;
let fps = 0;

// Place the camera deterministically (harness screenshots). Accepts an explicit
// position and either yaw/pitch or a look-at target (default: origin). Forces a
// fresh stream-center so chunks load around the new pose.
function applyCameraPose(p: {
  x?: number; y?: number; z?: number;
  yaw?: number; pitch?: number;
  tx?: number; ty?: number; tz?: number;
}): void {
  if (typeof p.x === "number" && typeof p.y === "number" && typeof p.z === "number") {
    camera.position.set(p.x, p.y, p.z);
  }
  if (typeof p.yaw === "number") yaw = p.yaw;
  if (typeof p.pitch === "number") pitch = p.pitch;
  if (p.yaw === undefined && p.pitch === undefined) {
    const dir = new THREE.Vector3(
      (p.tx ?? 0) - camera.position.x,
      (p.ty ?? 0) - camera.position.y,
      (p.tz ?? 0) - camera.position.z
    );
    if (dir.lengthSq() > 1e-6) {
      yaw = Math.atan2(dir.y, dir.x);
      pitch = Math.atan2(dir.z, Math.hypot(dir.x, dir.y));
    }
  }
  lastCenterSent = new THREE.Vector3(Infinity, Infinity, Infinity);
}

// Numeric viewport state for harness assertions / Chrome MCP readbacks. Not used
// by VS Code; harmless there. Read it with `window.__fireSceneDebug.snapshot()`.
const fireSceneDebug = {
  snapshot() {
    return {
      chunks: chunks.size,
      verts: totalVerts,
      tris: Math.round(totalTris),
      objects: objectGizmos.size,
      selected: selectedObjectId,
      hasGround: groundMaterial !== null,
      fps: Math.round(fps),
      camera: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
    };
  },
};
(window as unknown as { __fireSceneDebug?: unknown }).__fireSceneDebug = fireSceneDebug;

function tick(): void {
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;
  fps = fps * 0.9 + (1 / Math.max(dt, 1e-4)) * 0.1;

  // Flythrough movement only while looking (right-drag held) — keeps the editor
  // from drifting like a free debug camera when the user isn't navigating.
  if (dragMode === "look") {
    const fwd = forwardVector();
    const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
    const fast = keys.has("ShiftLeft") || keys.has("ShiftRight") ? 4 : 1;
    const speed = flySpeed * fast * dt;
    const move = new THREE.Vector3();
    if (keys.has("KeyW")) move.add(fwd);
    if (keys.has("KeyS")) move.sub(fwd);
    if (keys.has("KeyD")) move.add(right);
    if (keys.has("KeyA")) move.sub(right);
    if (keys.has("KeyE")) move.z += 1;
    if (keys.has("KeyQ")) move.z -= 1;
    if (move.lengthSq() > 0) camera.position.addScaledVector(move.normalize(), speed);
  }

  camera.up.set(0, 0, 1);
  camera.lookAt(camera.position.clone().add(forwardVector()));

  // Re-center streaming when the camera crosses ~half a chunk.
  if (camera.position.distanceTo(lastCenterSent) > chunkMeters * 0.5) {
    lastCenterSent = camera.position.clone();
    host.post({
      type: "camera",
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
    });
  }

  // Report a spawn focus point (where the camera looks) for new-object placement.
  if (now - lastFocusSent > 250) {
    lastFocusSent = now;
    const focus = camera.position.clone().addScaledVector(forwardVector(), 18);
    host.post({ type: "focus", x: focus.x, y: focus.y, z: focus.z });
  }

  if (groundMaterial) {
    (groundMaterial.uniforms.u_cam_pos.value as THREE.Vector3).copy(camera.position);
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
      if (msg.config && typeof msg.config.ground_seed === "number") {
        groundSeed = msg.config.ground_seed;
        groundTexelsPerM = Number(msg.config.ground_texels_per_m) || 16;
        if (groundMaterial) {
          groundMaterial.uniforms.u_ground_seed.value = groundSeed;
          groundMaterial.uniforms.u_ground_texels_per_m.value = groundTexelsPerM;
        }
      }
      break;
    case "groundLut": {
      const buf = msg.payload instanceof ArrayBuffer ? new Uint8Array(msg.payload) : (msg.payload as Uint8Array);
      applyGroundLut(buf);
      break;
    }
    case "reset":
      clearAll();
      setObjects([]);
      selectObjectLocal(null);
      break;
    case "objects":
      setObjects((msg.objects ?? []) as SceneObjectDTO[]);
      break;
    case "select":
      selectObjectLocal(msg.id === null || msg.id === undefined ? null : Number(msg.id));
      break;
    case "frame":
      frameObject(Number(msg.id));
      break;
    case "cameraPose":
      applyCameraPose(msg);
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
  updatePxRad();
});

// Gizmo mode buttons in the palette (W/E/R keyboard equivalents).
document.getElementById("gizmoMove")?.addEventListener("click", () => setGizmoMode("translate"));
document.getElementById("gizmoRotate")?.addEventListener("click", () => setGizmoMode("rotate"));
document.getElementById("gizmoScale")?.addEventListener("click", () => setGizmoMode("scale"));
setGizmoMode("translate");

host.post({ type: "ready" });
requestAnimationFrame(tick);
