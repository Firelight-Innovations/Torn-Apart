// Inspector quat <-> Euler conversion — round-trip and gimbal cases.
import assert from "node:assert/strict";
import { test } from "node:test";

import { eulerDegToQuat, quatToEulerDeg, QuatWXYZ } from "../webview/inspectorMath";

function quatClose(a: QuatWXYZ, b: QuatWXYZ, eps = 1e-6): boolean {
  // q and -q are the same rotation; compare via |dot| ~ 1.
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  return Math.abs(Math.abs(dot) - 1) < eps;
}

test("identity round-trips", () => {
  const e = quatToEulerDeg([1, 0, 0, 0]);
  for (const v of e) assert.ok(Math.abs(v) < 1e-9, `expected 0, got ${v}`);
  assert.ok(quatClose(eulerDegToQuat([0, 0, 0]), [1, 0, 0, 0]));
});

test("single-axis rotations round-trip in degrees", () => {
  for (const [axis, idx] of [["x", 0], ["y", 1], ["z", 2]] as const) {
    for (const deg of [-90, -45, 30, 60, 89]) {
      const e: [number, number, number] = [0, 0, 0];
      e[idx] = deg;
      const q = eulerDegToQuat(e);
      const back = quatToEulerDeg(q);
      assert.ok(
        Math.abs(back[idx] - deg) < 1e-4,
        `${axis}=${deg} came back as ${back[idx]}`
      );
    }
  }
});

test("arbitrary rotation round-trips euler -> quat -> euler -> quat", () => {
  const e: [number, number, number] = [24.5, -31.25, 142.0];
  const q1 = eulerDegToQuat(e);
  const q2 = eulerDegToQuat(quatToEulerDeg(q1));
  assert.ok(quatClose(q1, q2), `${q1} vs ${q2}`);
});

test("gimbal lock (pitch ±90°) stays stable", () => {
  for (const pitch of [90, -90]) {
    const q1 = eulerDegToQuat([35, pitch, 20]);
    const q2 = eulerDegToQuat(quatToEulerDeg(q1));
    assert.ok(quatClose(q1, q2, 1e-5), `pitch ${pitch}: ${q1} vs ${q2}`);
  }
});

test("unnormalised input is tolerated", () => {
  const q: QuatWXYZ = [2, 0, 0, 0]; // 2x identity
  for (const v of quatToEulerDeg(q)) {
    assert.ok(Math.abs(v) < 1e-9, `expected 0, got ${v}`);
  }
});
