#version 330 core
// lightning.frag — HDR emissive bolt ribbon (M7).
//
// A lightning channel is a thin, hot white-blue core with a softer glow falloff
// across the ribbon.  v_cross is -1..1 across the ribbon width; the core is the
// center, the glow the edges.  Output is premultiplied HDR, additively blended,
// so the bolt blooms in the post chain (no exposure pulse — the flash brightness
// rides in v_bright, set per phase by the renderer).

uniform float u_hdr_output;      // 1.0 → linear HDR out; 0.0 → tonemap-ish clamp
uniform vec3  u_core_color;      // hot core tint (near white, faint blue)
uniform vec3  u_glow_color;      // outer glow tint (cooler blue)

in float v_bright;               // flash × segment brightness
in float v_cross;                // -1..1 across the ribbon

out vec4 frag_color;

void main() {
    // Across-ribbon profile: a tight bright core + a wide soft glow.
    float a = abs(v_cross);
    float core = exp(-a * a * 9.0);          // tight hot center
    float glow = exp(-a * a * 1.6) * 0.45;   // broad halo

    vec3 col = (u_core_color * core + u_glow_color * glow) * v_bright;
    float alpha = clamp(core + glow, 0.0, 1.0) * clamp(v_bright, 0.0, 1.0);

    if (u_hdr_output > 0.5) {
        frag_color = vec4(col, alpha);       // premultiplied HDR, additive
    } else {
        // Legacy path: clamp so the LDR target doesn't blow out hard.
        frag_color = vec4(min(col, vec3(1.0)), alpha);
    }
}
