#version 330 core
// The full lighting contract (cascade sampling, shadow refinement, AO, fog,
// exposure/tonemap finish) lives in lit_surface.glsl — shared verbatim with
// every other lit-surface shader.  Terrain compiles the refinement march.
#define LIT_REFINE 1
//#include "lit_surface.glsl"

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

// Terrain-local cascade extras: own-cell emission volume (a voxel-surface
// concept — sampled with the cascade-0 window params from lit_surface.glsl).
uniform sampler3D u_c0_emis;
uniform float u_emission_scale;

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
    // Light sampling — positions quantised to the light-pixel grid so the
    // lighting itself is visibly pixelated (8x8x8 light pixels per voxel),
    // with power-of-two LOD by screen footprint (see litQuantSize).
    // ------------------------------------------------------------------
    float eff = litQuantSize(mpp);
    vec3 wq = litQuantPos(v_world, eff);
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
    if (u_refine > 0.5 && u_sun_dir.z > -0.05 && vis.r > 0.02 && vis.r < 0.98)
        vis.r = refineVisSoft(probeS, u_sun_dir);
    if (u_refine > 0.5 && u_moon_dir.z > -0.05 && vis.g > 0.02 && vis.g < 0.98 &&
        dot(u_moon_radiance, vec3(1.0)) > 1e-4)
        vis.g = refineVisSoft(probeS, u_moon_dir);

    // Voxel AO: occupancy a little farther out along the normal + above.
    vec3 aoR, aoV; float occFar;
    sampleCascades(wq + n * (u_c0_cell_m * 1.6), aoR, aoV, occFar);
    float ao = litAo(occ, occFar);

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

    // Volumetric fog composite + exposure/tonemap finish (lit_surface.glsl).
    frag_color = vec4(litFinish(litFog(hdr, dist)), 1.0);
}
