// Procedural-ground material for the editor viewport — a full-bright port of
// the game's terrain.frag world-space ground albedo (fire_engine/world/
// shaders/terrain.frag @ HEAD). Same hash, same octaves, same 4-corner
// texel-coverage filtering, same palette LUT — so the editor ground matches
// the in-game look pixel-for-pixel per seed, WITHOUT any lighting (the owner
// wants no lighting engine in the editor; v_color.rgb is baked vertex data,
// kept so facets stay readable).
//
// Gotchas carried over from the game shader (docs/systems/world.md):
//  - mpp (world meters per screen pixel) is ANALYTIC — never fwidth(); quad
//    derivatives explode across facet edges of the faceted mesh (gotcha 22).
//  - Posterise per corner, never the averaged noise (gotcha 21).
//  - The LUT row is clamped so blocky-mesher meshes (color.a == 1.0 → mat 255)
//    degrade to the last palette row instead of sampling garbage.
import * as THREE from "three";

const VERT = /* glsl */ `
// ShaderMaterial with vertexColors=false: three.js injects position/normal but
// NOT color, so this vec4 declaration binds our itemSize-4 "color" attribute
// (alpha carries the material id, not transparency).
in vec4 color;

out vec3 v_world;
out vec3 v_normal;
out vec4 v_color;

void main() {
    v_world  = position;          // chunk meshes are world-space at the origin
    v_normal = normal;
    v_color  = color;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAG = /* glsl */ `
precision highp float;

in vec3 v_world;
in vec3 v_normal;
in vec4 v_color;

uniform sampler2D u_ground_lut;      // row = material id, 256 cols of palette
uniform float u_ground_seed;         // per-world hash offset (determinism)
uniform float u_ground_texels_per_m; // virtual texels per world meter (~16)
uniform float u_ground_lut_rows;     // LUT height, for the row coordinate
uniform vec3  u_cam_pos;             // camera world position (meters)
uniform float u_px_rad;              // view angle per screen pixel (radians)

out vec4 fragColor;

// ---- verbatim port of terrain.frag's ground noise (HEAD) ------------------
float groundHash(ivec2 p, float seed) {
    uint h = uint(p.x) * 0x8da6b343u
           ^ uint(p.y) * 0xd8163841u
           ^ uint(int(seed)) * 0xcb1ab31fu;
    h ^= h >> 15; h *= 0x2c1b3c6du;
    h ^= h >> 12; h *= 0x297a2d39u;
    h ^= h >> 15;
    return float(h & 0x00ffffffu) / 16777215.0;
}

float groundOctave(vec2 sp, float texels, float seedoff, float mpp) {
    float v = groundHash(ivec2(floor(sp * texels)), u_ground_seed + seedoff);
    return mix(v, 0.5, smoothstep(0.5, 1.4, mpp * texels));
}

float groundNoise(vec2 sp, float mpp) {
    float fine = u_ground_texels_per_m;
    return 0.45 * groundOctave(sp, fine,           0.0, mpp)
         + 0.30 * groundOctave(sp, fine * 0.25,   41.0, mpp)
         + 0.25 * groundOctave(sp, fine * 0.0625, 91.0, mpp);
}
// ---------------------------------------------------------------------------

void main() {
    vec3 n  = normalize(v_normal);
    vec3 an = abs(n);

    // Analytic screen footprint (world m / screen px) — see header.
    float dist = length(v_world - u_cam_pos);
    vec3  vdir = (u_cam_pos - v_world) / max(dist, 1e-4);
    float cosi = max(abs(dot(vdir, n)), 0.18);
    float mpp  = dist * u_px_rad / cosi;

    // Planar projection onto the dominant-normal-axis plane (mesher UV axis).
    vec2 pw;
    if (an.x >= an.y && an.x >= an.z)      pw = v_world.yz;
    else if (an.y >= an.z)                 pw = v_world.xz;
    else                                   pw = v_world.xy;

    // 4-corner texel-coverage filtering, posterised per corner (gotcha 21).
    int   mat  = int(v_color.a * 255.0 + 0.5);
    float row  = clamp(float(mat), 0.0, u_ground_lut_rows - 1.0);
    float lrow = (row + 0.5) / u_ground_lut_rows;
    float ftex = u_ground_texels_per_m;
    vec2  spt  = pw * ftex - 0.5;
    vec2  bt   = floor(spt);
    vec2  fr   = spt - bt;
    float cov  = clamp(mpp * ftex, 1e-4, 1.0);
    vec2  w    = clamp((fr - 0.5) / cov + 0.5, 0.0, 1.0);
    vec2 p00 = (bt + vec2(0.5, 0.5)) / ftex;
    vec2 p10 = (bt + vec2(1.5, 0.5)) / ftex;
    vec2 p01 = (bt + vec2(0.5, 1.5)) / ftex;
    vec2 p11 = (bt + vec2(1.5, 1.5)) / ftex;
    float g00 = clamp(groundNoise(p00, mpp), 0.0, 1.0);
    float g10 = clamp(groundNoise(p10, mpp), 0.0, 1.0);
    float g01 = clamp(groundNoise(p01, mpp), 0.0, 1.0);
    float g11 = clamp(groundNoise(p11, mpp), 0.0, 1.0);
    vec3 c00 = texture(u_ground_lut, vec2((g00 * 255.0 + 0.5) / 256.0, lrow)).rgb;
    vec3 c10 = texture(u_ground_lut, vec2((g10 * 255.0 + 0.5) / 256.0, lrow)).rgb;
    vec3 c01 = texture(u_ground_lut, vec2((g01 * 255.0 + 0.5) / 256.0, lrow)).rgb;
    vec3 c11 = texture(u_ground_lut, vec2((g11 * 255.0 + 0.5) / 256.0, lrow)).rgb;
    vec3 alb = mix(mix(c00, c10, w.x), mix(c01, c11, w.x), w.y);

    // Full-bright compose: sRGB palette colour x baked facet/sun shading.
    // No linearisation/tonemap — the game only decodes to linear because it
    // re-encodes through its tonemapper; the editor shows the palette as-is.
    fragColor = vec4(alb * v_color.rgb, 1.0);
}
`;

export interface GroundParams {
  seed: number;
  texelsPerM: number;
}

/** Build the viewport ground material from a decoded LUT (RGBA8, rows x 256). */
export function makeGroundMaterial(
  lutData: Uint8Array,
  lutWidth: number,
  lutHeight: number,
  params: GroundParams
): THREE.ShaderMaterial {
  const lut = new THREE.DataTexture(
    lutData as Uint8Array<ArrayBuffer>,
    lutWidth,
    lutHeight,
    THREE.RGBAFormat
  );
  lut.magFilter = THREE.NearestFilter;
  lut.minFilter = THREE.NearestFilter;
  lut.generateMipmaps = false;
  lut.needsUpdate = true;

  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms: {
      u_ground_lut: { value: lut },
      u_ground_seed: { value: params.seed },
      u_ground_texels_per_m: { value: params.texelsPerM },
      u_ground_lut_rows: { value: lutHeight },
      u_cam_pos: { value: new THREE.Vector3() },
      u_px_rad: { value: 0.001 },
    },
    // vertexColors stays FALSE: we declare the vec4 color attribute ourselves
    // (three's injected declaration is vec3 and would fight the alpha id).
    vertexColors: false,
  });
}
