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

// Cloud density at world point p (0 = clear, →1 = dense).
float sampleDensity(vec3 p) {
    float hf = clamp((p.z - u_altitude) / u_thickness, 0.0, 1.0);
    // Rounded height profile: soft bottoms, tapered tops (cumulus-ish).
    float prof = smoothstep(0.0, 0.15, hf) * (1.0 - smoothstep(0.55, 1.0, hf));
    if (prof <= 0.0) return 0.0;

    vec3 wp = p + vec3(u_wind, 0.0);
    vec4 sh = texture(u_shape, wp * u_shape_scale);
    float base = sh.r * prof;                       // perlin-worley bulk

    // Coverage→threshold: a higher bar at low coverage leaves real blue gaps;
    // overcast lowers it so the sky nearly fills.  (Tuned via the debug view.)
    float thresh = mix(0.95, 0.55, u_coverage);
    float d = clamp(remap(base, thresh, min(thresh + 0.25, 1.0), 0.0, 1.0),
                    0.0, 1.0);
    if (d > 0.0) {
        // Erode edges into wisps with the Worley octaves + detail volume
        // (multiplied by (1-d) so cores stay solid, only rims erode).
        float fbm = sh.g * 0.5 + sh.b * 0.3 + sh.a * 0.2;
        float det = texture(u_detail, wp * u_detail_scale).r;
        d = clamp(d - (fbm + det) * u_detail_strength * (1.0 - d), 0.0, 1.0);
    }
    return d * u_cloud_density;
}

// Transmittance from p toward the sun (self-shadowing) via a short cone march.
float lightMarch(vec3 p) {
    float dsum = 0.0;
    for (int j = 0; j < u_light_steps; ++j) {
        float ls = (float(j) + 0.5) * u_light_step_m;
        dsum += sampleDensity(p + u_sun_dir * ls);
    }
    return exp(-dsum * u_light_step_m * u_sigma);
}

void main() {
    vec3 rd = normalize(v_dir);
    vec3 ro = u_cam_pos;
    float A = u_altitude;
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

    float steps = float(u_steps);
    float dt = (t1 - t0) / steps;
    float jitter = hash12(gl_FragCoord.xy + u_time);
    float t = t0 + dt * jitter;

    float mu = dot(rd, u_sun_dir);
    float phase = hgPhase(mu, u_hg) + 0.25 * hgPhase(mu, -0.15);   // fwd + ambient lobe

    vec3 scattered = vec3(0.0);
    float T = 1.0;
    // Distance fade so the slab edge at u_max_dist doesn't pop.
    for (int i = 0; i < u_steps; ++i) {
        vec3 p = ro + rd * t;
        float dens = sampleDensity(p);
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
            float hf = clamp((p.z - A) / u_thickness, 0.0, 1.0);
            vec3 amb = u_sky_ambient * (0.4 + 0.3 * hf);    // modest sky fill
            float Tstep = exp(-dens * u_sigma * dt);
            scattered += T * (sun + amb) * (1.0 - Tstep);
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
