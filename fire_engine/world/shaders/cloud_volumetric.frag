#version 330 core
// Volumetric raymarched clouds.
//
// Ray-marches a horizontal cloud slab [u_altitude, u_altitude+u_thickness]
// along the per-pixel world view direction (the cloud dome's model-space
// vertex).  Density comes from the baked tileable 3-D noise (sky/cloud_noise):
// a Perlin-Worley base eroded by Worley octaves + a detail volume, gated by a
// height profile and the weather coverage.  Each lit sample marches a few steps
// toward the sun for self-shadowing (Beer + powder), with a Henyey-Greenstein
// phase for the forward "silver lining".  Output is premultiplied
// (scattered_radiance, transmittance) and composited OVER the sky with
// blend = src + dst*srcAlpha, so a bright sun bleeds through thin cloud and
// thick cloud occludes it.
//
// M4 — SPATIAL WEATHER MAP + VIRGA
// --------------------------------
// When u_weather_map_enabled == 1 the per-step coverage/density/precip come
// from sampling the GPU weather-map texture (u_weather_map) at the march
// point's world XY instead of the flat u_coverage / u_cloud_density scalars.
// Beyond the map extent the sample edge-fades to u_weather_ambient so there is
// NO hard border.  When disabled the sample returns exactly the old scalar
// uniforms, so the pre-M4 look is bit-for-bit preserved (nothing regresses
// with the feature off).
//
// CRITICAL — the weather map already encodes cell MOTION (cells drift on the
// synoptic flow, baked into the raster each re-raster).  So the map UV MUST use
// the RAW world XY of the march point and must NEVER add u_wind.  u_wind stays
// ONLY on the noise lookups (it scrolls the procedural shape/detail volumes for
// the small-scale boil); adding it to the map UV would double-advect the storm
// and make the cell slide off its own rain.
//
// The precip channel (0..1) lowers + darkens storm-cloud BASES and, when
// u_virga_enabled == 1, adds gray VIRGA shafts: density hanging below the cloud
// base that erodes downward, lit as a desaturated gray so distant rain reads as
// the classic shaft under a storm.
uniform vec3  u_cam_pos;
uniform vec3  u_sun_dir;
uniform vec3  u_moon_dir;
uniform vec3  u_sun_radiance;     // linear HDR (SkyState contract)
uniform vec3  u_moon_radiance;
uniform vec3  u_sky_ambient;

uniform float u_altitude;         // slab bottom Z (m)
uniform float u_thickness;        // slab height (m)
uniform float u_max_dist;         // far raymarch cutoff (m)
uniform float u_coverage;         // 0..1 sky fill
uniform float u_cloud_density;    // opacity scale
uniform float u_shape_scale;      // 1/tile_m for the shape volume
uniform float u_detail_scale;     // 1/tile_m for the detail volume
uniform float u_detail_strength;  // edge erosion amount
uniform float u_sigma;            // extinction per meter at full density
uniform vec2  u_wind;             // accumulated wind offset (m)
uniform float u_time;             // s (jitter animation)
uniform int   u_steps;            // primary march samples
uniform int   u_light_steps;      // sun light-march samples
uniform float u_light_step_m;     // light-march step length (m)
uniform float u_hg;               // HG anisotropy (forward scatter)
uniform float u_hdr_output;       // 1 = emit linear HDR; 0 = tonemap (legacy)
uniform float u_exposure;         // legacy tonemap exposure

// --- M4 spatial weather map (RGBA16F: R=coverage G=density B=precip A=fog) ---
uniform sampler2D u_weather_map;       // spatial weather (bound by world/weather_renderer)
uniform vec2  u_wmap_origin;           // world XY (m) of the map's min corner
uniform float u_wmap_cell_m;           // texel edge (m)
uniform float u_wmap_cells;            // texels per axis
uniform int   u_weather_map_enabled;   // 0 ⇒ use flat u_coverage/u_cloud_density
uniform vec2  u_weather_ambient;       // (coverage, density) fallback beyond edge
uniform int   u_virga_enabled;         // 0 ⇒ no virga shafts (precip still darkens)

uniform sampler3D u_shape;
uniform sampler3D u_detail;

in vec3 v_dir;
out vec4 frag_color;

const float PI = 3.14159265358979;

float remap(float v, float a, float b, float c, float d) {
    return c + (v - a) * (d - c) / max(b - a, 1e-5);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float hgPhase(float mu, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}

vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

// Spatial weather at a world XY: (coverage, density, precip), each 0..1.
//
// RAW world XY only — the map already encodes cell motion (see header); adding
// u_wind here would double-advect the storm.  Beyond the map extent we
// edge-fade to u_weather_ambient (coverage, density; precip→0) over the outer
// ~12 % of the half-span so there is no hard border.  With the feature off we
// return exactly the old scalar uniforms (coverage, density, 0) — the pre-M4
// look is preserved bit-for-bit.
vec3 sampleWeather(vec2 worldXY) {
    if (u_weather_map_enabled == 0)
        return vec3(u_coverage, u_cloud_density, 0.0);

    float span = u_wmap_cell_m * u_wmap_cells;            // total extent (m)
    vec2 uv = (worldXY - u_wmap_origin) / span;           // 0..1 across the map
    // Sample (clamped sampler) then blend toward ambient as we approach/leave
    // the edge.  edge = 1 inside, → 0 outside, smooth over the rim band.
    vec4 wm = texture(u_weather_map, clamp(uv, 0.0, 1.0));
    vec2 d2 = abs(uv - 0.5);                              // 0 at centre, 0.5 at edge
    float outside = max(d2.x, d2.y);                      // Chebyshev distance
    float edge = 1.0 - smoothstep(0.44, 0.5, outside);    // fade over outer rim
    float cov   = mix(u_weather_ambient.x, wm.r, edge);
    float den   = mix(u_weather_ambient.y, wm.g, edge);
    float precip = wm.b * edge;                           // rain fades out past edge
    return vec3(cov, den, precip);
}

// Cloud density at world point p (0 = clear, →1 = dense).  *precip* (0..1) is
// the local rain strength returned alongside — it lowers + darkens the storm
// base and (out-param) drives the virga shaft below the slab.
float sampleDensity(vec3 p, out float precipOut) {
    vec3 wx = sampleWeather(p.xy);          // (coverage, density, precip), spatial
    float coverage = wx.x;
    float density  = wx.y;
    precipOut      = wx.z;

    float hf = clamp((p.z - u_altitude) / u_thickness, 0.0, 1.0);

    // VIRGA: precip pulls density DOWN below the cloud base (negative hf) as a
    // thinning shaft.  Sampled below the slab bottom, hf is < 0; we let a
    // precip-scaled band hang under the base, eroded by the detail noise so it
    // reads as ragged falling rain rather than a solid curtain.
    if (hf <= 0.0) {
        if (u_virga_enabled == 0 || precipOut <= 0.02) return 0.0;
        // Depth below the base in slab-thickness units (0 at base, 1 a slab
        // thickness down).  Shaft reaches ~0.8 thickness at full precip.
        float below = clamp((u_altitude - p.z) / u_thickness, 0.0, 1.0);
        float reach = 0.15 + 0.65 * precipOut;
        float shaft = precipOut * (1.0 - smoothstep(0.0, reach, below));
        if (shaft <= 0.0) return 0.0;
        vec3 wp = p + vec3(u_wind, 0.0);
        float det = texture(u_detail, wp * u_detail_scale).r;
        shaft = clamp(shaft - det * 0.6, 0.0, 1.0);       // ragged erosion
        return shaft * 0.55 * density;                    // thin (not a wall)
    }

    // Rounded height profile: soft bottoms, tapered tops (cumulus-ish).
    // Precip LOWERS the storm base: the lower lobe of the profile starts nearer
    // the slab bottom so heavy-rain cloud hangs lower and reads as a dark wall.
    float lo = mix(0.15, 0.04, precipOut);
    float prof = smoothstep(0.0, lo, hf) * (1.0 - smoothstep(0.55, 1.0, hf));
    if (prof <= 0.0) return 0.0;

    vec3 wp = p + vec3(u_wind, 0.0);
    vec4 sh = texture(u_shape, wp * u_shape_scale);
    float base = sh.r * prof;                       // perlin-worley bulk

    // Coverage→threshold: a higher bar at low coverage leaves real blue gaps;
    // overcast lowers it so the sky nearly fills.  (Tuned via the debug view.)
    float thresh = mix(0.95, 0.55, coverage);
    float d = clamp(remap(base, thresh, min(thresh + 0.25, 1.0), 0.0, 1.0),
                    0.0, 1.0);
    if (d > 0.0) {
        // Erode edges into wisps with the Worley octaves + detail volume
        // (multiplied by (1-d) so cores stay solid, only rims erode).
        float fbm = sh.g * 0.5 + sh.b * 0.3 + sh.a * 0.2;
        float det = texture(u_detail, wp * u_detail_scale).r;
        d = clamp(d - (fbm + det) * u_detail_strength * (1.0 - d), 0.0, 1.0);
    }
    return d * density;
}

// Transmittance from p toward the sun (self-shadowing) via a short cone march.
float lightMarch(vec3 p) {
    float dsum = 0.0;
    float precipDummy;
    for (int j = 0; j < u_light_steps; ++j) {
        float ls = (float(j) + 0.5) * u_light_step_m;
        dsum += sampleDensity(p + u_sun_dir * ls, precipDummy);
    }
    return exp(-dsum * u_light_step_m * u_sigma);
}

void main() {
    vec3 rd = normalize(v_dir);
    vec3 ro = u_cam_pos;
    // Extend the marched slab DOWNWARD when virga is on so the rain shafts
    // hanging below the cloud base (sampleDensity's hf<=0 branch) are covered.
    // With virga off A == u_altitude exactly → identical bounds to pre-M4.
    float virgaDepth = (u_virga_enabled != 0 && u_weather_map_enabled != 0)
                       ? 0.7 * u_thickness : 0.0;
    float A = u_altitude - virgaDepth;
    float B = u_altitude + u_thickness;

    // Ray vs horizontal slab.
    float t0, t1;
    if (abs(rd.z) < 1e-4) {
        if (ro.z < A || ro.z > B) { frag_color = vec4(0.0, 0.0, 0.0, 1.0); return; }
        t0 = 0.0; t1 = u_max_dist;
    } else {
        float ta = (A - ro.z) / rd.z;
        float tb = (B - ro.z) / rd.z;
        t0 = max(min(ta, tb), 0.0);
        t1 = max(ta, tb);
        if (t1 <= 0.0) { frag_color = vec4(0.0, 0.0, 0.0, 1.0); return; }
        t1 = min(t1, u_max_dist);
    }
    if (t1 <= t0) { frag_color = vec4(0.0, 0.0, 0.0, 1.0); return; }

    // Scale the sample count with the (virga-)extended slab depth so dt — and
    // thus cloud-body sampling density — stays ~constant when virga widens the
    // marched range.  Capped at 1.7× u_steps to bound the extra cost; with
    // virga off thickFrac == 1 → exactly u_steps (pre-M4).
    float thickFrac = (u_thickness + virgaDepth) / u_thickness;     // 1 .. 1.7
    int nSteps = int(min(float(u_steps) * thickFrac, float(u_steps) * 1.7));
    float steps = float(nSteps);
    float dt = (t1 - t0) / steps;
    float jitter = hash12(gl_FragCoord.xy + u_time);
    float t = t0 + dt * jitter;

    float mu = dot(rd, u_sun_dir);
    float phase = hgPhase(mu, u_hg) + 0.25 * hgPhase(mu, -0.15);   // fwd + ambient lobe

    vec3 scattered = vec3(0.0);
    float T = 1.0;
    // Distance fade so the slab edge at u_max_dist doesn't pop.
    for (int i = 0; i < nSteps; ++i) {
        vec3 p = ro + rd * t;
        float precip;
        float dens = sampleDensity(p, precip);
        if (dens > 0.001) {
            float Tl = lightMarch(p);
            float powder = 1.0 - exp(-dens * 4.0);          // dark cores
            // Direct sun = phase·(self-shadow) + a multi-scatter wrap term so
            // clouds read as bright WHITE masses (high albedo), not just a lit
            // sun-facing edge.  The 2.0× compensates SkyState.sun_radiance
            // already being dimmed by cloud cover at the ground — the cloud
            // TOPS see the full, undimmed sun.  Ambient is a MODEST blue fill —
            // keep it below the sun term or clouds tint to the sky and vanish.
            vec3 sun = u_sun_radiance * 2.0 * (phase * Tl + 0.45 * Tl) *
                       mix(0.55, 1.0, powder);
            // Height vs the CLOUD body (u_altitude), not the virga-extended A —
            // keeps the body's ambient gradient identical to pre-M4.  Below the
            // base (virga) hf clamps to 0 (darkest fill).
            float hf = clamp((p.z - u_altitude) / u_thickness, 0.0, 1.0);
            vec3 amb = u_sky_ambient * (0.4 + 0.3 * hf);    // modest sky fill
            // Precip DARKENS + GRAYS the lit colour: storm bases read as a dark
            // wall and virga as a desaturated gray shaft.  Strongest low in the
            // cloud / in the shaft (1-hf), so tops stay bright.  Gated by the
            // map (precip is 0 when u_weather_map_enabled==0 → no change).
            float wet = precip * (1.0 - 0.6 * hf);
            vec3 lit = sun + amb;
            float gray = dot(lit, vec3(0.299, 0.587, 0.114));
            lit = mix(lit, vec3(gray) * 0.55, wet * 0.7);   // desaturate + dim
            float Tstep = exp(-dens * u_sigma * dt);
            scattered += T * lit * (1.0 - Tstep);
            T *= Tstep;
            if (T < 0.02) break;
        }
        t += dt;
    }

    // Fade out near the far cutoff and toward the horizon grazing angle.
    float edge = 1.0 - smoothstep(u_max_dist * 0.75, u_max_dist, t0);
    float alpha = (1.0 - T) * edge;
    scattered *= edge;
    T = 1.0 - alpha;

    if (u_hdr_output > 0.5) {
        // Premultiplied linear HDR; composite blend adds dst*T.
        frag_color = vec4(scattered, T);
    } else {
        vec3 ldr = pow(acesTonemap(scattered * u_exposure), vec3(1.0 / 2.2));
        frag_color = vec4(ldr, T);
    }
}
