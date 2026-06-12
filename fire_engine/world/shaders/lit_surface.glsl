// lit_surface.glsl — THE surface-lighting contract (radiance cascades).
//
// Every shader that draws a lit surface — terrain, grass, flora, trees,
// and future buildings/NPCs — includes this file instead of hand-copying
// the lighting code (three drifted copies of it are how foliage went
// washed-out once; never again).  GLSL has no includes, so the engine's
// loader (`core/shader_source.py::load_glsl`) expands the directive:
//
//     #version 330 core
//     #define LIT_REFINE 1            // optional: compile the shadow march
//     //#include "lit_surface.glsl"
//
// AUTHORING RECIPE — to light any object:
//   1. Pass a world-space position from your vertex stage (the fragment's
//      own position for real meshes; the instance base for billboards).
//   2. Quantise it to the light-pixel grid:
//        float eff = litQuantSize(dist * u_px_rad);   // or u_quant_m flat
//        vec3  wq  = litQuantPos(world_pos, eff);
//   3. Sample the field:    sampleCascades(wq, radiance, vis, occ);
//   4. (LIT_REFINE only) re-resolve celestial shadow edges wherever the
//      trilinear vis is in the penumbra band, gated by the u_refine knob:
//        if (u_refine > 0.5 && u_sun_dir.z > -0.05 &&
//            vis.r > 0.02 && vis.r < 0.98)
//            vis.r = refineVis(probe, u_sun_dir);      // or refineVisSoft
//   5. Compose:
//        vec3 direct = u_sun_radiance  * (vis.r * max(dot(n, u_sun_dir), 0.0))
//                    + u_moon_radiance * (vis.g * max(dot(n, u_moon_dir), 0.0));
//        float ao  = litAo(occ, occFar);   // occFar from a 2nd sampleCascades,
//                                          // or pass occ twice for 1-tap AO
//        vec3 hdr  = albedo * (direct + radiance * ao);
//   6. Finish:  frag_color = vec4(litFinish(litFog(hdr, dist)), alpha);
//
// All uniforms below are bound/refreshed engine-wide by
// `lighting/gpu.py::GpuLightingPipeline.bind_surface_inputs` /
// `update_surface_inputs` — a shader only has to declare them by including
// this file; scene-graph inheritance does the rest.  `u_refine` is the one
// per-object knob: bind it on your object's root (terrain pins 1.0,
// foliage binds the `gfx_foliage_shadow_refine` config value).

// --- radiance cascades (lighting/gpu.py contract) -----------------------
uniform sampler3D u_c0_radiance;
uniform sampler3D u_c0_vis;       // r sun, g moon, b sky visibility
uniform sampler3D u_c0_geom;      // rgb albedo, a occupancy
uniform vec3  u_c0_origin_m;
uniform float u_c0_cell_m;
uniform float u_c0_cells;
uniform sampler3D u_c1_radiance;
uniform sampler3D u_c1_vis;
uniform sampler3D u_c1_geom;
uniform vec3  u_c1_origin_m;
uniform float u_c1_cell_m;
uniform float u_c1_cells;
uniform sampler3D u_c2_radiance;  // coarse FAR cascade (8 m cells, 512 m box)
uniform sampler3D u_c2_vis;
uniform sampler3D u_c2_geom;
uniform vec3  u_c2_origin_m;
uniform float u_c2_cell_m;
uniform float u_c2_cells;

uniform vec3  u_sun_dir;
uniform vec3  u_sun_radiance;
uniform vec3  u_moon_dir;
uniform vec3  u_moon_radiance;
uniform vec3  u_sky_ambient;
uniform float u_quant_m;          // light-pixel size (0.0625 m → 8x8 per voxel)
uniform float u_px_rad;           // view angle per screen pixel (radians)
uniform float u_ao_strength;
uniform float u_exposure;
// 1.0 → output linear HDR (the post-process pass tonemaps); 0.0 → tonemap
// here (legacy path when post-processing is disabled).  Inherited from
// ``render`` so it applies to every surface shader at once.
uniform float u_hdr_output;

// --- froxel fog ----------------------------------------------------------
uniform sampler3D u_fog_integrated;  // rgb accumulated light, a transmittance
uniform float u_fog_near;
uniform float u_fog_far;
uniform float u_fog_enabled;
uniform vec2  u_viewport;
uniform vec3  u_cam_pos;

vec3 c_uv(vec3 wp, vec3 origin, float cell_m, float cells) {
    return (wp - origin) / (cell_m * cells);
}

bool inBox(vec3 uv, float pad) {
    return all(greaterThan(uv, vec3(pad))) && all(lessThan(uv, vec3(1.0 - pad)));
}

// Containment weight in [0,1] for a cascade's [0,1] uv cube: 1.0 well inside,
// fading to 0.0 over an outer band ``fade`` wide (as a fraction of the cube
// half-extent) at every face.  Component-wise min so a point near ANY face
// fades — the box is convex, so the min is the correct soft-containment test.
// 0 outside the cube entirely.  This replaces the hard ``inBox`` cliff that
// made the cascade swap pop as a moving seam ring.
float boxWeight(vec3 uv, float fade) {
    // Distance from the nearest face, in [0,0.5] (0.5 == cube centre).
    vec3 d = min(uv, 1.0 - uv);
    float m = min(d.x, min(d.y, d.z));
    // smoothstep(0, fade): 0 at the face, 1 once ``fade`` deep.  fade is a
    // fraction of the half-extent (0.5), so a 0.10 fade ~= outer 5% rim.
    return smoothstep(0.0, fade, m);
}

// Sample a cascade triple at a [0,1] uv (no containment test — caller weights).
void sampleCascadeAt(sampler3D rad, sampler3D vs, sampler3D gm, vec3 uv,
                     out vec3 radiance, out vec3 vis, out float occ) {
    radiance = texture(rad, uv).rgb;
    vis      = texture(vs, uv).rgb;
    occ      = texture(gm, uv).a;
}

// Sample the radiance/visibility/occupancy field with a SMOOTH finest-first
// handoff between the three cascades and into the sky-ambient fallback — no
// hard ``inBox`` cliffs, so nothing pops or rings as the camera moves and the
// cascade windows slide.
//
// Each cascade carries a containment weight that fades over an outer band of
// its box; we take the finest cascade whose weight is > 0 and cross-fade to
// the next coarser tier by (1 - that weight).  Cost: well inside cascade 0
// (the common case) w0 == 1, so only c0 is sampled — one branch, one triple.
// Bands cost at most two triples.  Self-contained (the AO path calls this a
// second time per fragment with a different world point).
//
// Fade bands (fraction of half-extent): c0 0.14, c1 0.12, c2 0.10.  At
// cell sizes 0.5/2/8 m and box half-extents 24/96/256 m these are ~3.4 m /
// ~11.5 m / ~25.6 m wide — each comfortably wider than the next-coarser
// cascade's cell so the blend never reveals that tier's texel grid.
void sampleCascades(vec3 wp, out vec3 radiance, out vec3 vis, out float occ) {
    vec3 r0, v0, r1, v1, r2, v2;
    float o0, o1, o2;

    vec3 uv0 = c_uv(wp, u_c0_origin_m, u_c0_cell_m, u_c0_cells);
    float w0 = boxWeight(uv0, 0.14);

    vec3 uv1 = c_uv(wp, u_c1_origin_m, u_c1_cell_m, u_c1_cells);
    float w1 = boxWeight(uv1, 0.12);

    vec3 uv2 = c_uv(wp, u_c2_origin_m, u_c2_cell_m, u_c2_cells);
    float w2 = boxWeight(uv2, 0.10);

    // Coarsest tier first, then mix finer over it: result = lerp(coarser,
    // finer, w_finer).  Start from the sky-ambient fallback (never black).
    radiance = u_sky_ambient * 0.6;
    vis      = vec3(1.0);
    occ      = 0.0;

    if (w2 > 0.0) {
        sampleCascadeAt(u_c2_radiance, u_c2_vis, u_c2_geom, uv2, r2, v2, o2);
        radiance = mix(radiance, r2, w2);
        vis      = mix(vis,      v2, w2);
        occ      = mix(occ,      o2, w2);
    }
    if (w1 > 0.0) {
        sampleCascadeAt(u_c1_radiance, u_c1_vis, u_c1_geom, uv1, r1, v1, o1);
        radiance = mix(radiance, r1, w1);
        vis      = mix(vis,      v1, w1);
        occ      = mix(occ,      o1, w1);
    }
    if (w0 > 0.0) {
        sampleCascadeAt(u_c0_radiance, u_c0_vis, u_c0_geom, uv0, r0, v0, o0);
        radiance = mix(radiance, r0, w0);
        vis      = mix(vis,      v0, w0);
        occ      = mix(occ,      o0, w0);
    }
}

#ifdef LIT_REFINE
// Dynamic occluder AABBs (dev cubes, props — objects not in the voxel
// field).  Same contract as inject.comp: the refinement march must test
// them analytically or it would erase their shadows in the penumbra band.
uniform int   u_num_boxes;
uniform vec4  u_box_min[16];
uniform vec4  u_box_max[16];
uniform float u_penumbra_tan;     // tan(celestial penumbra cone half-angle)
// Per-object runtime gate for the refinement march (terrain binds 1.0;
// foliage roots bind the ``gfx_foliage_shadow_refine`` config value, so
// the iGPU preset can turn the march off without recompiling shaders).
uniform float u_refine;

// Sample a cascade's occupancy at the CELL CENTRE containing ``p`` (nearest-
// texel semantics on a linear-filtered texture), so the refinement march
// below reads the same binary field inject.comp's marchVis sees.
float occCell(sampler3D gm, vec3 p, vec3 origin, float cell_m, float cells) {
    vec3 cc = (floor((p - origin) / cell_m) + 0.5) * cell_m + origin;
    return texture(gm, c_uv(cc, origin, cell_m, cells)).a;
}

// Analytic ray-vs-AABB shadow test against the dynamic occluder boxes
// (mirror of inject.comp's boxVis).  0.0 when any box blocks the ray.
float boxVis(vec3 ro, vec3 rd, float tmax) {
    for (int i = 0; i < u_num_boxes; i++) {
        vec3 inv = 1.0 / (rd + vec3(1e-7));
        vec3 ta = (u_box_min[i].xyz - ro) * inv;
        vec3 tb = (u_box_max[i].xyz - ro) * inv;
        vec3 tlo = min(ta, tb);
        vec3 thi = max(ta, tb);
        float tn = max(max(tlo.x, tlo.y), tlo.z);
        float tf = min(min(thi.x, thi.y), thi.z);
        if (tf >= max(tn, 0.05) && tn < tmax) return 0.0;
    }
    return 1.0;
}

// Per-fragment celestial-shadow REFINEMENT march.  The cascade ``vis``
// volumes are voxel-crisp DATA, but trilinear sampling smears the shadow
// edge over one full cell of whichever cascade covers the fragment — 2 m in
// cascade 1, 8 m in cascade 2 — which renders as big soft boxes (4x4 voxels
// per c1 cell) even though the compute side is per-voxel.  Wherever the
// sampled vis is in the penumbra band, re-resolve it by marching the
// occupancy chain (finest box first, falling through to the coarser ones)
// from the quantized light-pixel position: the edge then lands per light
// pixel — rendered resolution == computed resolution at any distance.
// Full-sun / full-shadow fragments never get here (1 texture tap as before).
float refineVis(vec3 wp, vec3 dir) {
    float vis = boxVis(wp, dir, 1e3);          // dynamic occluders (analytic)
    if (vis < 0.01) return 0.0;
    vec3 p = wp + dir * (u_c0_cell_m * 1.2);   // hop off the own surface
    for (int i = 0; i < 28; i++) {             // cascade 0: 0.5 m steps
        if (!inBox(c_uv(p, u_c0_origin_m, u_c0_cell_m, u_c0_cells), 0.0))
            break;
        vis *= 1.0 - occCell(u_c0_geom, p, u_c0_origin_m,
                             u_c0_cell_m, u_c0_cells);
        if (vis < 0.01) return 0.0;
        p += dir * u_c0_cell_m;
    }
    for (int i = 0; i < 24; i++) {             // cascade 1: 2 m steps
        if (!inBox(c_uv(p, u_c1_origin_m, u_c1_cell_m, u_c1_cells), 0.0))
            break;
        vis *= 1.0 - occCell(u_c1_geom, p, u_c1_origin_m,
                             u_c1_cell_m, u_c1_cells);
        if (vis < 0.01) return 0.0;
        p += dir * u_c1_cell_m;
    }
    for (int i = 0; i < 12; i++) {             // cascade 2: 8 m steps
        if (!inBox(c_uv(p, u_c2_origin_m, u_c2_cell_m, u_c2_cells), 0.0))
            break;
        vis *= 1.0 - occCell(u_c2_geom, p, u_c2_origin_m,
                             u_c2_cell_m, u_c2_cells);
        if (vis < 0.01) return 0.0;
        p += dir * u_c2_cell_m;
    }
    return vis;
}

// Soft-penumbra refinement: average four refineVis marches whose directions
// are jittered inside a cone of half-angle atan(u_penumbra_tan) around the
// light.  The fractional result turns the hard voxel-quantised shadow
// boundary (a re-resolved march is binary per ray) into a smooth gradient
// whose width grows with occluder distance — a real penumbra.  Runs from the
// UNQUANTISED probe so the gradient is continuous across light pixels, and
// only inside the penumbra band (the gate in main), so full-sun/full-shadow
// fragments still cost one texture tap.
float refineVisSoft(vec3 wp, vec3 dir) {
    vec3 up = abs(dir.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 t1 = normalize(cross(dir, up));
    vec3 t2 = cross(dir, t1);
    float r = u_penumbra_tan * 0.7071;         // diagonal cone offsets
    float v = 0.0;
    v += refineVis(wp, normalize(dir + ( t1 + t2) * r));
    v += refineVis(wp, normalize(dir + ( t1 - t2) * r));
    v += refineVis(wp, normalize(dir + (-t1 + t2) * r));
    v += refineVis(wp, normalize(dir + (-t1 - t2) * r));
    return v * 0.25;
}
#endif // LIT_REFINE

vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

// Light-pixel quantisation size with power-of-two LOD: when a light pixel
// would shrink below ~1.2 screen pixels (``mpp`` = world metres per screen
// pixel), snap to the next power-of-two multiple of ``u_quant_m``.  Power-
// of-two steps keep the coarser lattice EXACTLY nested in the finer one and
// world-anchored; a continuously varying cell size would re-seat every cell
// boundary as the footprint moved ("breathing" grid = its own shimmer).  Up
// close ``u_quant_m`` wins, so the chunky pixel-art lighting is unchanged.
float litQuantSize(float mpp) {
    return u_quant_m * exp2(ceil(max(0.0, log2(mpp * 1.2 / u_quant_m))));
}

// Snap a world position to the centre of its ``eff``-sized light pixel.
vec3 litQuantPos(vec3 wp, float eff) {
    return (floor(wp / eff) + 0.5) * eff;
}

// Voxel AO from cascade occupancy: ``occNear`` at the lit point, ``occFar``
// a little farther out along the normal (pass occNear twice for cheap
// single-tap AO on thin geometry like grass blades).
float litAo(float occNear, float occFar) {
    return 1.0 - u_ao_strength * clamp(0.5 * occNear + 0.7 * occFar, 0.0, 1.0);
}

// Volumetric fog composite (one tap into the integrated froxels), keyed by
// the fragment's camera distance.  Uses gl_FragCoord — fragment stage only.
vec3 litFog(vec3 hdr, float dist) {
    if (u_fog_enabled > 0.5) {
        float w = log(max(dist, u_fog_near) / u_fog_near)
                / log(u_fog_far / u_fog_near);
        vec2 suv = gl_FragCoord.xy / u_viewport;
        vec4 fog = texture(u_fog_integrated, vec3(suv, clamp(w, 0.0, 1.0)));
        hdr = hdr * fog.a + fog.rgb;
    }
    return hdr;
}

// Auto-exposure is applied HERE (so bloom downstream works on the exposed
// signal); under the HDR pipeline the post-process composite does the single
// ACES tonemap + gamma, otherwise (legacy path) tonemap + gamma now.
vec3 litFinish(vec3 hdr) {
    vec3 graded = hdr * u_exposure;
    if (u_hdr_output > 0.5)
        return graded;                                  // linear HDR
    return pow(acesTonemap(graded), vec3(1.0 / 2.2));
}
