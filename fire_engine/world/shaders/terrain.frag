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

uniform vec3  u_sun_dir;
uniform vec3  u_sun_radiance;
uniform vec3  u_moon_dir;
uniform vec3  u_moon_radiance;
uniform vec3  u_sky_ambient;
uniform float u_quant_m;          // light-pixel size (0.0625 m → 8x8 per voxel)
uniform float u_ao_strength;
uniform float u_exposure;
uniform float u_emission_scale;

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

// Sample a cascade triple (radiance, vis, occupancy) with cascade-0 priority.
void sampleCascades(vec3 wp, out vec3 radiance, out vec3 vis, out float occ) {
    vec3 uv0 = c_uv(wp, u_c0_origin_m, u_c0_cell_m, u_c0_cells);
    if (inBox(uv0, 0.02)) {
        radiance = texture(u_c0_radiance, uv0).rgb;
        vis      = texture(u_c0_vis, uv0).rgb;
        occ      = texture(u_c0_geom, uv0).a;
        return;
    }
    vec3 uv1 = c_uv(wp, u_c1_origin_m, u_c1_cell_m, u_c1_cells);
    if (inBox(uv1, 0.01)) {
        radiance = texture(u_c1_radiance, uv1).rgb;
        vis      = texture(u_c1_vis, uv1).rgb;
        occ      = texture(u_c1_geom, uv1).a;
        return;
    }
    vec3 uv2 = c_uv(wp, u_c2_origin_m, u_c2_cell_m, u_c2_cells);
    if (inBox(uv2, 0.005)) {              // coarse far cascade — low-res GI/shadow
        radiance = texture(u_c2_radiance, uv2).rgb;
        vis      = texture(u_c2_vis, uv2).rgb;
        occ      = texture(u_c2_geom, uv2).a;
        return;
    }
    radiance = u_sky_ambient * 0.6;       // beyond all cascades: open sky guess
    vis      = vec3(1.0);
    occ      = 0.0;
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
    // Light sampling â€” positions quantised to the light-pixel grid so the
    // lighting itself is visibly pixelated (8x8x8 light pixels per voxel).
    // ------------------------------------------------------------------
    vec3 wq = (floor(v_world / u_quant_m) + 0.5) * u_quant_m;
    // Shadow/GI probes hop off the surface along the *face* normal.
    vec3 probe   = wq + n * (u_c0_cell_m * 0.75);
    vec3 radiance, vis;
    float occ;
    sampleCascades(probe, radiance, vis, occ);

    // Voxel AO: occupancy a little farther out along the normal + above.
    vec3 aoR, aoV; float occFar;
    sampleCascades(wq + n * (u_c0_cell_m * 1.6), aoR, aoV, occFar);
    float ao = 1.0 - u_ao_strength * clamp(0.5 * occ + 0.7 * occFar, 0.0, 1.0);

    // ------------------------------------------------------------------
    // Compose: direct celestial + flood-fill GI + emission.
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
    // Anti-alias the ground.  The albedo is a HARD per-texel hash in world
    // space, so as the camera moves a pixel sitting on a texel edge — or a
    // minified sub-pixel texel — flips hash buckets every frame and flickers
    // ("z-fights" between palette colours).  Filter it analytically:
    //  1. rotated-grid 2x2 supersample over the screen-space footprint
    //     (``fwidth(pw)`` = world metres per pixel).  When the ground is
    //     magnified all four taps fall in one texel so the pixel-art stays
    //     crisp; when minified the taps straddle the footprint and average,
    //     killing edge crawl and near-field sparkle.
    //  2. LOD: each noise octave inside groundNoise() additionally fades by
    //     its OWN footprint, so as a pixel grows the fine octave drops first,
    //     then the mid, leaving the macro patches — distant ground stays
    //     varied (large light/dark grass patches) instead of going flat green.
    vec2  fp   = vec2(fwidth(pw.x), fwidth(pw.y));        // world m / pixel
    float mpp  = max(fp.x, fp.y);                          // world m / pixel
    float gnoise = 0.25 * (
        groundNoise(pw + vec2(-0.375, -0.125) * fp, mpp) +
        groundNoise(pw + vec2( 0.125, -0.375) * fp, mpp) +
        groundNoise(pw + vec2(-0.125,  0.375) * fp, mpp) +
        groundNoise(pw + vec2( 0.375,  0.125) * fp, mpp));
    gnoise = clamp(gnoise, 0.0, 1.0);
    int  mat = int(v_color.a * 255.0 + 0.5);
    vec3 alb = texture(u_ground_lut,
                       vec2((gnoise * 255.0 + 0.5) / 256.0,
                            (float(mat) + 0.5) / u_ground_lut_rows)).rgb;
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
        float dist = length(v_world - u_cam_pos);
        float w = log(max(dist, u_fog_near) / u_fog_near)
                / log(u_fog_far / u_fog_near);
        vec2 suv = gl_FragCoord.xy / u_viewport;
        vec4 fog = texture(u_fog_integrated, vec3(suv, clamp(w, 0.0, 1.0)));
        hdr = hdr * fog.a + fog.rgb;
    }

    vec3 ldr = acesTonemap(hdr * u_exposure);
    frag_color = vec4(pow(ldr, vec3(1.0 / 2.2)), 1.0);
}
