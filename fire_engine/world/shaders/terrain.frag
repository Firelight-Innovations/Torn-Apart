#version 330 core
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

uniform sampler2D p3d_Texture0;   // albedo (unused on the procedural-ground path)
uniform sampler2D p3d_Texture1;   // tangent-space normal map
uniform sampler2D p3d_Texture2;   // emission map (linear HDR/8-bit)

// --- world-space procedural ground (non-repeating pixel-art albedo) ------
uniform sampler2D u_ground_lut;          // row = material id, 256 cols = posterised palette
uniform float u_ground_seed;             // per-world hash offset (determinism)
uniform float u_ground_texels_per_m;     // virtual texels per world meter (~16)
uniform float u_ground_lut_rows;         // LUT height, for the row coordinate

// --- radiance cascades (lighting/gpu.py contract) -----------------------
uniform sampler3D u_c0_radiance;
uniform sampler3D u_c0_vis;       // r sun, g moon, b sky visibility
uniform sampler3D u_c0_geom;      // rgb albedo, a occupancy
uniform sampler3D u_c0_emis;
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

// Dynamic occluder AABBs (dev cubes, props — objects not in the voxel
// field).  Same contract as inject.comp: the refinement march must test
// them analytically or it would erase their shadows in the penumbra band.
uniform int   u_num_boxes;
uniform vec4  u_box_min[16];
uniform vec4  u_box_max[16];

uniform vec3  u_sun_dir;
uniform vec3  u_sun_radiance;
uniform vec3  u_moon_dir;
uniform vec3  u_moon_radiance;
uniform vec3  u_sky_ambient;
uniform float u_quant_m;          // light-pixel size (0.0625 m → 8x8 per voxel)
uniform float u_penumbra_tan;     // tan(celestial penumbra cone half-angle)
uniform float u_px_rad;           // view angle per screen pixel (radians)
uniform float u_ao_strength;
uniform float u_exposure;
uniform float u_emission_scale;
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

vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

// Integer hash -> [0, 1).  A lowbias32-style finaliser over a 2-D + seed mix;
// uniform output so posterise buckets get an even spread.
float groundHash(ivec2 p, float seed) {
    uint h = uint(p.x) * 0x8da6b343u
           ^ uint(p.y) * 0xd8163841u
           ^ uint(int(seed)) * 0xcb1ab31fu;
    h ^= h >> 15; h *= 0x2c1b3c6du;
    h ^= h >> 12; h *= 0x297a2d39u;
    h ^= h >> 15;
    return float(h & 0x00ffffffu) / 16777215.0;
}

// One hash octave at ``texels`` virtual texels/m, analytically minification-
// faded toward the hash mean (0.5) as a screen pixel approaches the size of
// THIS octave's texels (its own mip transition).  ``mpp`` is world metres per
// screen pixel, so ``mpp * texels`` is *this octave's texels per pixel*
// (1.0 == one texel per pixel == the Nyquist limit).
//
// The fade MUST complete at/before Nyquist, not start there: a hard hash is
// white noise, so any octave whose texel is at or below ~1 pixel cannot be
// resolved and only aliases (sparkles/"z-fights") as the camera sweeps it
// across the pixel grid.  Fade it out from texel â‰ˆ 2 px (0.5 texels/px) to
// fully gone by texel â‰ˆ 0.7 px (1.4 texels/px).  Each octave drops out at its
// own distance, so distant ground keeps its coarse colour patches instead of
// every scale collapsing to one flat colour ("sea of green").
float groundOctave(vec2 sp, float texels, float seedoff, float mpp) {
    float v = groundHash(ivec2(floor(sp * texels)), u_ground_seed + seedoff);
    return mix(v, 0.5, smoothstep(0.5, 1.4, mpp * texels));
}

// Level-of-detail posterised ground-noise value (0..1) at a world-plane
// position: three octaves (fine 1x, mid 4x, macro 16x larger texels), each
// self-fading by its own screen footprint so detail degrades gracefully with
// distance.  A single octave tap is a hard step in world space, so sample the
// whole thing through the supersampler in main() for near-field edge AA.
float groundNoise(vec2 sp, float mpp) {
    float fine = u_ground_texels_per_m;
    return 0.45 * groundOctave(sp, fine,           0.0, mpp)
         + 0.30 * groundOctave(sp, fine * 0.25,   41.0, mpp)
         + 0.25 * groundOctave(sp, fine * 0.0625, 91.0, mpp);
}

void main() {
    // ------------------------------------------------------------------
    // Surface basis + normal map (TBN from the dominant axis, matching the
    // mesher's planar UV projection: X-facingâ†’(Y,Z), Yâ†’(X,Z), Zâ†’(X,Y)).
    // ------------------------------------------------------------------
    vec3 n  = normalize(v_normal);
    vec3 an = abs(n);
    vec3 t, b;
    if (an.x >= an.y && an.x >= an.z)      { t = vec3(0, 1, 0); b = vec3(0, 0, 1); }
    else if (an.y >= an.z)                 { t = vec3(1, 0, 0); b = vec3(0, 0, 1); }
    else                                   { t = vec3(1, 0, 0); b = vec3(0, 1, 0); }
    t = normalize(t - n * dot(n, t));
    b = normalize(b - n * dot(n, b) - t * dot(t, b));
    vec3 nm = texture(p3d_Texture1, v_uv).xyz * 2.0 - 1.0;
    vec3 N  = normalize(t * nm.x + b * nm.y + n * max(nm.z, 0.3));

    // ------------------------------------------------------------------
    // ANALYTIC screen footprint of this fragment on its surface, in world
    // metres per screen pixel: distance x (radians per pixel) / cos(angle of
    // incidence).  NEVER use fwidth()/dFdx() for this in the terrain shader:
    // derivatives are evaluated on 2x2 pixel quads, and wherever a quad
    // straddles two facets of the faceted mesh the helper pixels extrapolate
    // the wrong plane — the derivative explodes, every LOD/AA term driven by
    // it pops, and sloped dense-triangle areas (crater rims, cliffs) sparkle
    // under sub-pixel camera motion (world.md gotcha 22; measured ~7x worse
    // than flat ground before this change).  The analytic form is exact for
    // planar facets and perfectly stable.  ``cosi`` is clamped so grazing
    // facets (≈1 px wide anyway) don't request infinite coarseness.
    // ------------------------------------------------------------------
    float dist = length(v_world - u_cam_pos);
    vec3  vdir = (u_cam_pos - v_world) / max(dist, 1e-4);
    float cosi = max(abs(dot(vdir, n)), 0.18);
    float mpp  = dist * u_px_rad / cosi;       // world m per screen pixel

    // ------------------------------------------------------------------
    // Light sampling â€” positions quantised to the light-pixel grid so the
    // lighting itself is visibly pixelated (8x8x8 light pixels per voxel).
    // LOD: when a light pixel would shrink below ~1.2 screen pixels, snap to
    // the next power-of-two multiple of ``u_quant_m`` instead.  Power-of-two
    // steps keep the coarser lattice EXACTLY nested in the finer one (cells
    // merge 8-into-1 at a LOD change) and the lattice world-anchored; a
    // continuously varying cell size would re-seat every cell boundary each
    // time the footprint moved ("breathing" grid = its own shimmer).  Up
    // close ``u_quant_m`` wins, so the chunky pixel-art lighting is unchanged.
    // ------------------------------------------------------------------
    float eff = u_quant_m * exp2(ceil(max(0.0, log2(mpp * 1.2 / u_quant_m))));
    vec3 wq = (floor(v_world / eff) + 0.5) * eff;
    // Shadow/GI probes hop off the surface along the *face* normal.
    vec3 probe   = wq + n * (u_c0_cell_m * 0.75);
    vec3 radiance, vis;
    float occ;
    sampleCascades(probe, radiance, vis, occ);

    // Re-resolve the celestial shadow edge wherever the trilinear cascade
    // sample is in the penumbra band (see refineVisSoft).  The cone-jittered
    // marches start at the UNQUANTISED surface probe so the penumbra is a
    // smooth continuous gradient, not light-pixel stairs; full-sun and
    // full-shadow fragments never get here (1 texture tap as before).
    vec3 probeS = v_world + n * (u_c0_cell_m * 0.75);
    if (u_sun_dir.z > -0.05 && vis.r > 0.02 && vis.r < 0.98)
        vis.r = refineVisSoft(probeS, u_sun_dir);
    if (u_moon_dir.z > -0.05 && vis.g > 0.02 && vis.g < 0.98 &&
        dot(u_moon_radiance, vec3(1.0)) > 1e-4)
        vis.g = refineVisSoft(probeS, u_moon_dir);

    // Voxel AO: occupancy a little farther out along the normal + above.
    vec3 aoR, aoV; float occFar;
    sampleCascades(wq + n * (u_c0_cell_m * 1.6), aoR, aoV, occFar);
    float ao = 1.0 - u_ao_strength * clamp(0.5 * occ + 0.7 * occFar, 0.0, 1.0);

    // ------------------------------------------------------------------
    // Compose: direct celestial + gathered voxel GI + emission.
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // World-space procedural ground albedo (never repeats across the map).
    // Planar projection onto the dominant-normal-axis plane (same axis pick
    // as the mesher's UVs) keeps texels square; snapping to an integer texel
    // grid yields crisp pixel-art blocks rather than smooth noise.  The noise
    // value indexes a per-material posterised palette LUT, so the look matches
    // the baked grass_ground/dirt_ground art exactly.
    vec2 pw;
    if (an.x >= an.y && an.x >= an.z)      pw = v_world.yz;
    else if (an.y >= an.z)                 pw = v_world.xz;
    else                                   pw = v_world.xy;
    // Anti-alias the ground with ANALYTIC TEXEL-COVERAGE filtering.  The
    // albedo is a hard per-texel hash posterised by a palette LUT — two
    // quantisers whose edges sparkle under any naive sampling.  The scheme
    // that measures temporally silent (tools/shimmer_probe.py):
    //  1. Evaluate the noise stack ONLY at the 4 nearest fine-texel CENTRES.
    //     Texel centres are fixed world points, so every octave's hash —
    //     and therefore each corner's posterised COLOUR — is constant under
    //     camera motion.  (The previous fixed-offset supersample slid its
    //     taps continuously through the hash field; on crater walls seen
    //     head-on — small footprint, no octave fade, full-contrast texels —
    //     every tap crossing popped a quarter palette step, which the owner
    //     saw as "z-fighting in the dirt".)
    //  2. Posterise EACH corner through the LUT and blend the COLOURS by the
    //     pixel footprint's coverage of each texel (w = box-filtered texel
    //     edge, transition band ≈ 1 screen pixel).  The result is a
    //     CONTINUOUS function of surface position: texel edges render as
    //     crisp 1-px AA ramps that slide smoothly — zero popping by
    //     construction.  Interiors stay one flat palette colour (w saturates
    //     0/1 inside a texel), so the pixel-art look is intact up close.
    //  3. Posterise per corner, never the averaged noise (the LUT is a hard
    //     quantiser; quantise-after-filter re-hardens what the filter
    //     smoothed — world.md gotcha 21).
    //  4. Octave LOD inside groundNoise() still fades each octave by its own
    //     footprint before its texels alias (coverage support is only 2x2
    //     fine texels, so content must be band-limited past that).
    //  ``mpp`` is the analytic footprint from above — never fwidth() (facet-
    //  edge derivative garbage, world.md gotcha 22).
    int   mat  = int(v_color.a * 255.0 + 0.5);
    float lrow = (float(mat) + 0.5) / u_ground_lut_rows;
    float ftex = u_ground_texels_per_m;
    vec2  spt  = pw * ftex - 0.5;              // fine-texel space, centred
    vec2  bt   = floor(spt);
    vec2  fr   = spt - bt;
    float cov  = clamp(mpp * ftex, 1e-4, 1.0); // footprint in fine texels
    vec2  w    = clamp((fr - 0.5) / cov + 0.5, 0.0, 1.0);  // coverage weights
    vec3 c00, c10, c01, c11;
    {
        vec2 p00 = (bt + vec2(0.5, 0.5)) / ftex;
        vec2 p10 = (bt + vec2(1.5, 0.5)) / ftex;
        vec2 p01 = (bt + vec2(0.5, 1.5)) / ftex;
        vec2 p11 = (bt + vec2(1.5, 1.5)) / ftex;
        float g00 = clamp(groundNoise(p00, mpp), 0.0, 1.0);
        float g10 = clamp(groundNoise(p10, mpp), 0.0, 1.0);
        float g01 = clamp(groundNoise(p01, mpp), 0.0, 1.0);
        float g11 = clamp(groundNoise(p11, mpp), 0.0, 1.0);
        c00 = texture(u_ground_lut, vec2((g00 * 255.0 + 0.5) / 256.0, lrow)).rgb;
        c10 = texture(u_ground_lut, vec2((g10 * 255.0 + 0.5) / 256.0, lrow)).rgb;
        c01 = texture(u_ground_lut, vec2((g01 * 255.0 + 0.5) / 256.0, lrow)).rgb;
        c11 = texture(u_ground_lut, vec2((g11 * 255.0 + 0.5) / 256.0, lrow)).rgb;
    }
    vec3 alb = mix(mix(c00, c10, w.x), mix(c01, c11, w.x), w.y);
    vec3 base = pow(alb, vec3(2.2)) * v_color.rgb;

    vec3 direct =
        u_sun_radiance  * (vis.r * max(dot(N, u_sun_dir),  0.0)) +
        u_moon_radiance * (vis.g * max(dot(N, u_moon_dir), 0.0));

    vec3 ownEmis = vec3(0.0);
    vec3 uv0 = c_uv(v_world - n * (u_c0_cell_m * 0.25),
                    u_c0_origin_m, u_c0_cell_m, u_c0_cells);
    if (inBox(uv0, 0.0))
        ownEmis = texture(u_c0_emis, uv0).rgb * u_emission_scale;
    vec3 mapEmis = pow(texture(p3d_Texture2, v_uv).rgb, vec3(2.2)) * 4.0;

    vec3 hdr = base * (direct + radiance * ao) + ownEmis + mapEmis;

    // ------------------------------------------------------------------
    // Volumetric fog composite (one tap into the integrated froxels).
    // ------------------------------------------------------------------
    if (u_fog_enabled > 0.5) {
        float w = log(max(dist, u_fog_near) / u_fog_near)
                / log(u_fog_far / u_fog_near);
        vec2 suv = gl_FragCoord.xy / u_viewport;
        vec4 fog = texture(u_fog_integrated, vec3(suv, clamp(w, 0.0, 1.0)));
        hdr = hdr * fog.a + fog.rgb;
    }

    // Auto-exposure is applied HERE (so bloom downstream works on the exposed
    // signal); the post-process composite does the single ACES tonemap + gamma.
    vec3 graded = hdr * u_exposure;
    if (u_hdr_output > 0.5) {
        frag_color = vec4(graded, 1.0);                         // linear HDR
    } else {
        frag_color = vec4(pow(acesTonemap(graded), vec3(1.0 / 2.2)), 1.0);
    }
}
