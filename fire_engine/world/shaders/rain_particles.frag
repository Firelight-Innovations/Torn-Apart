#version 330 core
// rain_particles.frag — volumetric rain-streak fragment stage (M6).
//
// A soft vertical streak: bright core fading to the edges, additively blended
// so streaks brighten the scene subtly (like the cylinder rain) rather than
// smearing gray.  Distance-dimmed by the froxel fog (inherited from
// terrain_root) so far streaks fade into the murk instead of punching through.

in vec2  v_uv;                    // [0,1]² over the streak quad
in float v_fade;                  // precip × edge fade × alive (vertex)

out vec4 frag_color;

uniform float u_exposure;         // shared lighting exposure (inherited)
uniform vec3  u_rain_tint;        // streak colour (cool gray-blue)

// --- froxel fog (inherited from terrain_root) ---------------------------
uniform sampler3D u_fog_integrated;
uniform float u_fog_enabled;
uniform vec2  u_viewport;

void main() {
    if (v_fade < 0.004) discard;

    // Streak shape: a thin vertical line that fades to the sides and tapers at
    // the ends.  |u-0.5| → side falloff; v taper softens head/tail.
    float side = 1.0 - smoothstep(0.0, 0.5, abs(v_uv.x - 0.5));
    float ends = smoothstep(0.0, 0.25, v_uv.y) * smoothstep(0.0, 0.25, 1.0 - v_uv.y);
    float a = side * ends * v_fade;
    if (a < 0.004) discard;

    // Additive: streaks add a cool highlight scaled by exposure.
    vec3 add = u_rain_tint * a * (0.35 * clamp(u_exposure, 0.2, 4.0));

    if (u_fog_enabled > 0.5) {
        vec2 suv = gl_FragCoord.xy / u_viewport;
        float w = clamp(gl_FragCoord.z, 0.0, 1.0);
        vec4 fog = texture(u_fog_integrated, vec3(suv, w));
        add *= fog.a;                  // fog.a = scene transmittance
    }

    frag_color = vec4(add, a);
}
