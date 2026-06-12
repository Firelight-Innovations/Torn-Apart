
#version 330 core
uniform samplerCube u_star_cube;    // night_sky_cube faces (alpha = luminance)
uniform sampler2D u_moon_tex;       // procedural "moon_surface" lunar disc
uniform vec3  u_celestial_axis;     // unit, toward the celestial north pole

uniform vec3  u_sun_dir;
uniform vec3  u_sun_color;          // legacy tint (kept for stubs); disc uses transmittance
uniform float u_sun_intensity;
uniform vec3  u_moon_dir;
uniform float u_moon_phase;
uniform float u_moon_glow;          // 0-1 illuminated fraction (moonlit-sky gain)
uniform vec3  u_zenith_color;       // legacy/weather-graded (below-horizon fill)
uniform vec3  u_horizon_color;
uniform float u_star_visibility;
uniform float u_star_rotation;
uniform float u_time;
uniform float u_daylight;           // SkyState.daylight (night-floor blend)
uniform float u_weather_gray;       // 0-1 overcast desaturation weight
uniform float u_exposure;           // tonemap exposure (match terrain shader)
uniform float u_hdr_output;         // 1.0 → linear HDR out (post tonemaps); 0.0 → tonemap here
// Config-exposed sky/sun tuning (gfx_* in [graphics]; see core/config.py).
uniform float u_sun_disc_intensity; // HDR gain on the sun disc (bloom blob)
uniform float u_sun_halo_intensity; // HDR gain on the forward-Mie sun halo
uniform float u_sun_min_brightness; // floor on disc/halo transmittance (low sun)
uniform float u_sky_inscatter_scale;// scattered-sky radiance multiplier
uniform vec3  u_fog_color;
uniform float u_fog_blend;          // legacy horizon fog (CPU backend; 0 on GPU)
uniform sampler3D u_fog_integrated; // froxel fog (GPU backend)
uniform float u_fog_enabled;
uniform vec2  u_viewport;
uniform float u_ss_active;
uniform vec3  u_ss_start;
uniform vec3  u_ss_travel;
uniform float u_ss_progress;

in vec3 v_dir;
out vec4 frag_color;

const float PI = 3.14159265358979;

// --- Physical atmosphere constants (mirror sky/atmosphere.py exactly) -----
const vec3  BETA_R  = vec3(5.8e-6, 13.5e-6, 33.1e-6);
const float BETA_M  = 3.9e-6;
const float MIE_EXT = 1.1;
const float HR      = 8500.0;
const float HM      = 1200.0;
const float RP      = 6371000.0;
const float RT      = 6431000.0;          // RP + 60 km
const float R0      = 6371002.0;          // RP + observer 2 m
const float SUN_I   = 22.0;               // SUN_TOA_RADIANCE
const float MIE_G   = 0.76;
const int   STEPS   = 16;     // view-ray scatter samples (twilight smoothness)
const int   LSTEPS  = 4;       // sun-ray optical-depth samples per view sample

// Disc angular radii â€” ~2.5x their realistic sizes, per art direction.
const float SUN_ANG_R  = 0.0218;          // ~1.25 deg
const float MOON_ANG_R = 0.0175;          // ~1.00 deg

// Stable 3D -> 1D hash (no trig, no texture) for star twinkle.
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float exitDist(float r, float cb) {
    float disc = cb * cb - (r * r - RT * RT);
    return -cb + sqrt(max(disc, 0.0));
}

// Transmittance from the observer toward dir (sun/moon disc tint).
vec3 viewTransmittance(vec3 d) {
    float cb = R0 * max(d.z, 0.0);
    float tExit = exitDist(R0, cb);
    float odR = 0.0;
    float odM = 0.0;
    float dt = tExit / 8.0;
    for (int i = 0; i < 8; i++) {
        float t = (float(i) + 0.5) * dt;
        float r = sqrt(R0 * R0 + t * t + 2.0 * t * cb);
        float h = max(r - RP, 0.0);
        odR += exp(-h / HR) * dt;
        odM += exp(-h / HM) * dt;
    }
    return exp(-(BETA_R * odR + BETA_M * MIE_EXT * odM));
}

// Single-scattered radiance along view ray d for a celestial light at s â€”
// the same integral as sky/atmosphere.py sky_radiance(), fewer steps.
vec3 scatterLight(vec3 d, vec3 s, float lightI) {
    float cb = R0 * d.z;
    float tExit = exitDist(R0, cb);
    float discP = cb * cb - (R0 * R0 - RP * RP);
    if (cb < 0.0 && discP > 0.0) tExit = min(tExit, -cb - sqrt(discP));
    float mu = dot(d, s);
    float phR = (3.0 / (16.0 * PI)) * (1.0 + mu * mu);
    float g2 = MIE_G * MIE_G;
    float phM = (3.0 / (8.0 * PI)) * (1.0 - g2) * (1.0 + mu * mu)
              / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * MIE_G * mu, 1.5));
    float odRv = 0.0;
    float odMv = 0.0;
    vec3 L = vec3(0.0);
    for (int i = 0; i < STEPS; i++) {
        // Quadratic spacing (mirrors sky/atmosphere.py): dense sampling near
        // the observer so grazing rays see the low, dense atmosphere.
        float u = (float(i) + 0.5) / float(STEPS);
        float t = tExit * u * u;
        float dt = tExit * (2.0 * u / float(STEPS));
        float r = sqrt(R0 * R0 + t * t + 2.0 * t * cb);
        float h = max(r - RP, 0.0);
        float dR = exp(-h / HR);
        float dM = exp(-h / HM);
        odRv += dR * dt;
        odMv += dM * dt;
        // Sun-ray from the sample: planet-shadow test gives the earth-shadow
        // twilight arch after sunset.
        float cbl = R0 * s.z + t * mu;
        float discL = cbl * cbl - (r * r - RP * RP);
        if (cbl < 0.0 && discL > 0.0) continue;
        float tl = exitDist(r, cbl);
        float dl = tl / float(LSTEPS);
        float odRl = 0.0;
        float odMl = 0.0;
        for (int j = 0; j < LSTEPS; j++) {
            float t2 = (float(j) + 0.5) * dl;
            float rl = sqrt(r * r + t2 * t2 + 2.0 * t2 * cbl);
            float hl = max(rl - RP, 0.0);
            odRl += exp(-hl / HR) * dl;
            odMl += exp(-hl / HM) * dl;
        }
        vec3 tau = BETA_R * (odRv + odRl) + BETA_M * MIE_EXT * (odMv + odMl);
        L += exp(-tau) * (BETA_R * (dR * phR) + BETA_M * (dM * phM)) * dt;
    }
    return lightI * L;
}

// Disc-local coordinates for a celestial body; returns the disc mask.
float discFactor(vec3 d, vec3 dirTo, float angR, out vec2 local, out float rr) {
    vec3 ref = (abs(dirTo.z) > 0.97) ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 t = normalize(cross(ref, dirTo));
    vec3 b = cross(dirTo, t);
    local = vec2(dot(d, t), dot(d, b)) / angR;
    rr = length(local);
    return 1.0 - smoothstep(0.92, 1.0, rr);
}

vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

void main() {
    vec3 d = normalize(v_dir);

    // --- Physically-scattered sky (Rayleigh + Mie single scattering) -------
    // Below the geometric horizon (the dome shows past the finite world's
    // edge) the scatter ray is clamped to graze the horizon so the band
    // continues the horizon hue instead of going black at the ground hit.
    vec3 ds = (d.z < 0.015) ? normalize(vec3(d.xy, 0.015)) : d;
    // Scattered-sky radiance, scaled by the config knob (lower = less low-sun
    // wash-out).  The sun disc/halo are added AFTER and are NOT scaled here.
    vec3 col = scatterLight(ds, u_sun_dir, SUN_I) * u_sky_inscatter_scale;
    // Moonlit sky: same physics, tiny intensity, scaled by the phase.
    if (u_moon_dir.z > -0.05 && u_moon_glow > 0.01) {
        col += scatterLight(ds, u_moon_dir, SUN_I * 0.0045 * u_moon_glow);
    }
    // Overcast: desaturate toward gray (luminance-preserving).
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, vec3(luma), 0.75 * u_weather_gray);
    // Artistic night floor (matches SkySystem's _NIGHT_* constants).
    float grad = pow(clamp(d.z, 0.0, 1.0), 0.42);
    col += mix(vec3(0.020, 0.026, 0.046), vec3(0.012, 0.016, 0.035), grad)
           * (1.0 - u_daylight);
    // Below the horizon: gently dim the continued horizon band downward.
    if (d.z < 0.0) col *= clamp(1.0 + d.z * 1.2, 0.45, 1.0);

    vec3 sunT = viewTransmittance(u_sun_dir);
    // Keep a low (sunrise/sunset) sun reading bright: lift its transmittance to
    // a floor on the PEAK channel, preserving the warm red/orange hue so the
    // sun no longer "fades out" near the horizon.
    float sunTpk = max(sunT.r, max(sunT.g, sunT.b));
    sunT *= max(1.0, u_sun_min_brightness / max(sunTpk, 1e-4));

    // Disc coverage of the sun/moon at this pixel — opaque bodies that must
    // occlude the star skybox (added post-tonemap below) so they never look
    // transparent.  Accumulated as the max of the sun & moon disc masks.
    float bodyMask = 0.0;

    // --- Volumetric fog over the BACKGROUND sky (GPU froxel far slice) ------
    // CRITICAL ORDER: fog the BACKGROUND sky here, then add the sun/moon discs
    // AFTER, attenuated by the fog transmittance ONLY (not buried under the
    // grey inscatter ``fog.rgb``).  The sky is at infinity so it is correctly
    // seen through the whole fog column; the discs are bright HDR spikes, so
    // even multiplied by a low transmittance they survive tonemap + bloom and
    // PUNCH THROUGH the fog (dimmed, not erased) — fixing the "sun hidden
    // behind a grey fog layer" look.  (Old order added the disc to ``col``
    // first, so fog then both attenuated it AND swamped it with inscatter.)
    float fogA = 1.0;
    vec3  fogRGB = vec3(0.0);
    if (u_fog_enabled > 0.5) {
        vec4 fog = texture(u_fog_integrated,
                           vec3(gl_FragCoord.xy / u_viewport, 1.0));
        fogA = fog.a;
        fogRGB = fog.rgb;
    }
    col = col * fogA + fogRGB;

    // --- Sun: large limb-darkened disc tinted by its own transmittance -----
    // In HDR the disc is pushed FAR above 1.0 so the bloom pass bleeds it into
    // a soft, edgeless blob (how a real bright sun reads) — clamping the disc
    // to ~white (legacy path) instead gives the hard-edged disc.  The halo is
    // the forward-Mie glow that haloes the disc and also feeds the bloom.
    float discGain = (u_hdr_output > 0.5) ? u_sun_disc_intensity : 14.0;
    float haloGain = (u_hdr_output > 0.5) ? u_sun_halo_intensity : 0.55;
    float sc = dot(d, u_sun_dir);
    if (sc > 0.999) {
        vec2 sl;
        float sr;
        float sdisc = discFactor(d, u_sun_dir, SUN_ANG_R, sl, sr);
        if (sdisc > 0.0) {
            float limb = sqrt(max(1.0 - sr * sr, 0.0));
            float ld = 0.42 + 0.58 * limb;               // limb darkening
            col += sunT * (u_sun_intensity * sdisc * ld * discGain) * fogA;
            bodyMask = max(bodyMask, sdisc);             // sun occludes stars
        }
    }
    // Forward-Mie halo around the sun (also fog-attenuated).
    col += sunT * (u_sun_intensity * pow(max(sc, 0.0), 350.0) * haloGain) * fogA;

    // --- Moon: large textured disc with dynamic phase terminator -----------
    float mc = dot(d, u_moon_dir);
    if (mc > 0.999) {
        vec2 ml;
        float mr;
        float mdisc = discFactor(d, u_moon_dir, MOON_ANG_R, ml, mr);
        if (mdisc > 0.0) {
            vec3 mtex = texture(u_moon_tex, ml * 0.5 + 0.5).rgb;
            float mz = sqrt(max(1.0 - clamp(mr * mr, 0.0, 1.0), 0.0));
            float ph = (u_moon_phase - 0.5) * 2.0 * PI;   // 0 at full
            vec3 mlight = vec3(sin(ph), 0.0, cos(ph));
            float lit = smoothstep(-0.08, 0.28, dot(vec3(ml, mz), mlight));
            vec3 moonCol = mtex * viewTransmittance(u_moon_dir)
                         * (0.05 + 1.10 * lit) * 1.5;
            col = mix(col, moonCol, mdisc * fogA);        // dims in thick fog
            bodyMask = max(bodyMask, mdisc);              // and occludes stars
        }
    }
    // Faint moon halo, night only.
    col += vec3(0.75, 0.78, 0.85)
         * (pow(max(mc, 0.0), 1600.0) * 0.10 * (1.0 - u_daylight) * u_moon_glow)
         * fogA;

    // Legacy horizon fog band (CPU backend; u_fog_blend forced 0 on GPU).
    float fog_band = 1.0 - smoothstep(0.0, 0.38, d.z);
    col = mix(col, u_fog_color, u_fog_blend * fog_band);

    // --- Night sky art (stars/galaxy/twinkle) ------------------------------
    // The whole celestial sphere rotates about the TILTED celestial axis
    // (Polaris elevation = the world's latitude), not world +Z â€” so the
    // stars rise in the east and set in the west instead of pinwheeling
    // overhead.  Rodrigues rotation of the view dir into the star frame.
    vec3 ax = u_celestial_axis;
    float cr = cos(u_star_rotation);
    float sr = sin(u_star_rotation);
    vec3 sd = d * cr + cross(ax, d) * sr + ax * (dot(ax, d) * (1.0 - cr));
    vec4 night = texture(u_star_cube, sd);
    // Per-star flicker: a stable hash over the star-frame direction picks
    // each star's phase + speed; a smooth sine gives a lively shimmer
    // (galaxy band barely flickers â€” the alpha mask gates it to stars).
    float h = hash13(floor(sd * 300.0));
    float tw01 = 0.5 + 0.5 * sin(u_time * (3.0 + 6.0 * h) + h * 6.2831853);
    float twinkle = mix(1.0, 0.30 + 1.40 * tw01,
                        smoothstep(0.40, 0.85, night.a));
    vec3 stars = night.rgb * twinkle * u_star_visibility;
    stars *= smoothstep(-0.06, 0.18, d.z);              // sink into horizon haze
    stars *= 1.0 - clamp(bodyMask, 0.0, 1.0);           // sun/moon in front of sky

    // --- Shooting star: bright fading streak along a great circle ----------
    vec3 ss = vec3(0.0);
    if (u_ss_active > 0.5) {
        vec3 s = u_ss_start;
        vec3 tv = u_ss_travel;
        vec3 n = cross(s, tv);
        float dist_plane = dot(d, n);
        float along = atan(dot(d, tv), dot(d, s));
        float arc = 0.55;
        float head = u_ss_progress * arc;
        float behind = head - along;
        float tail_lum = (behind >= 0.0) ? exp(-behind * 30.0) : 0.0;
        float width = exp(-(dist_plane * dist_plane) / (2.0 * 0.003 * 0.003));
        float fade = sin(PI * clamp(u_ss_progress, 0.0, 1.0));
        ss = vec3(1.0, 0.97, 0.88) * tail_lum * width * fade * 1.8
           * u_star_visibility;
    }

    // --- Output ------------------------------------------------------------
    // HDR path: the sky+disc go out linear (exposure applied) for the post
    // chain to tonemap, so the bright sun survives into bloom.  Stars + the
    // shooting star are display-referred emissive detail added on top; they
    // ride along into the tonemap (they bloom nicely once that phase lands).
    // Legacy path: tonemap here, then add the LDR night art exactly as before.
    if (u_hdr_output > 0.5) {
        frag_color = vec4(col * u_exposure + stars + ss, 1.0);
    } else {
        vec3 ldr = pow(acesTonemap(col * u_exposure), vec3(1.0 / 2.2));
        frag_color = vec4(ldr + stars + ss, 1.0);
    }
}
