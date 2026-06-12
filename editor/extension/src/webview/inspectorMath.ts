// Quaternion <-> Euler conversion for the Inspector's rotation fields.
// Wire format is scalar-first (w, x, y, z) — fire_engine.core.math3d.Quat /
// SceneObject.rotation. Display is intrinsic XYZ Euler in DEGREES (the same
// convention as three.js Euler order "XYZ", so viewport gizmo rotations and
// inspector numbers agree). Pure functions, no DOM/three.js — unit-tested by
// src/test/inspectorMath.test.ts.

export type QuatWXYZ = [number, number, number, number];
export type EulerDeg = [number, number, number];

const RAD2DEG = 180 / Math.PI;
const DEG2RAD = Math.PI / 180;

/** Quaternion (w,x,y,z) -> intrinsic XYZ Euler angles in degrees. */
export function quatToEulerDeg(q: QuatWXYZ): EulerDeg {
  const [w, x, y, z] = normalize(q);
  // Rotation-matrix elements (row-major) of the unit quaternion.
  const m11 = 1 - 2 * (y * y + z * z);
  const m12 = 2 * (x * y - w * z);
  const m13 = 2 * (x * z + w * y);
  const m23 = 2 * (y * z - w * x);
  const m33 = 1 - 2 * (x * x + y * y);
  const m22 = 1 - 2 * (x * x + z * z);
  const m32 = 2 * (y * z + w * x);

  const ey = Math.asin(clamp(m13, -1, 1));
  let ex: number;
  let ez: number;
  if (Math.abs(m13) < 0.9999999) {
    ex = Math.atan2(-m23, m33);
    ez = Math.atan2(-m12, m11);
  } else {
    // Gimbal lock: pitch at ±90° — fold the lost axis into X.
    ex = Math.atan2(m32, m22);
    ez = 0;
  }
  return [ex * RAD2DEG, ey * RAD2DEG, ez * RAD2DEG];
}

/** Intrinsic XYZ Euler angles in degrees -> quaternion (w,x,y,z). */
export function eulerDegToQuat(e: EulerDeg): QuatWXYZ {
  const hx = (e[0] * DEG2RAD) / 2;
  const hy = (e[1] * DEG2RAD) / 2;
  const hz = (e[2] * DEG2RAD) / 2;
  const c1 = Math.cos(hx), s1 = Math.sin(hx);
  const c2 = Math.cos(hy), s2 = Math.sin(hy);
  const c3 = Math.cos(hz), s3 = Math.sin(hz);
  return normalize([
    c1 * c2 * c3 - s1 * s2 * s3,
    s1 * c2 * c3 + c1 * s2 * s3,
    c1 * s2 * c3 - s1 * c2 * s3,
    c1 * c2 * s3 + s1 * s2 * c3,
  ]);
}

function normalize(q: QuatWXYZ): QuatWXYZ {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
