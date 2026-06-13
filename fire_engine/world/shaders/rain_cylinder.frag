#version 330 core
// rain_cylinder.frag — cheap "cylinders" rain mode fragment stage (M6).
//
// Samples the scrolled rain-streak texture (additive, like the old sky
// renderer), then applies the SAME two gates as the particle mode at the
// fragment's WORLD XY:
//   * heightmap cull — discard fragments below the rain-cover height (so the
//     cheap mode also stops raining under a roof/overhang);
//   * precip gate    — fade the streaks with the weather-map precip channel
//     (or the scalar u_rain_intensity when the weather map is off).

in vec2 v_uv;
in vec3 v_world;

out vec4 frag_color;

uniform sampler2D u_rain_tex;     // rain_streak texture (RGBA, additive)
uniform float u_rain_alpha;       // overall layer opacity (intensity-driven)
uniform float u_rain_intensity;   // scalar fallback when the weather map is off
uniform float u_rain_occlusion;   // 1.0 = apply the heightmap cull

// --- rain-cover heightmap -----------------------------------------------
uniform sampler2D u_rain_height_tex;
uniform vec2  u_rain_height_origin;
uniform float u_rain_height_cell_m;
uniform float u_rain_height_cells;

// --- weather map (inherited from render) --------------------------------
uniform sampler2D u_weather_map;
uniform vec2  u_wmap_origin;
uniform float u_wmap_cell_m;
uniform float u_wmap_cells;
uniform int   u_weather_map_enabled;

void main() {
    // Precip gate at the fragment XY.
    float precip;
    if (u_weather_map_enabled != 0) {
        vec2 wuv = (v_world.xy - u_wmap_origin) / (u_wmap_cell_m * u_wmap_cells);
        precip = texture(u_weather_map, wuv).b;
    } else {
        precip = u_rain_intensity;
    }
    if (precip < 0.02) discard;

    // Heightmap cull: discard under cover.
    if (u_rain_occlusion > 0.5) {
        vec2 huv = (v_world.xy - u_rain_height_origin)
                   / (u_rain_height_cell_m * u_rain_height_cells);
        if (huv.x >= 0.0 && huv.x <= 1.0 && huv.y >= 0.0 && huv.y <= 1.0) {
            float cover_z = texture(u_rain_height_tex, huv).r;
            if (v_world.z < cover_z) discard;
        }
    }

    vec4 streak = texture(u_rain_tex, v_uv);
    float a = streak.a * u_rain_alpha * clamp(precip, 0.0, 1.0);
    if (a < 0.004) discard;
    frag_color = vec4(streak.rgb * a, a);   // additive (premultiplied)
}
