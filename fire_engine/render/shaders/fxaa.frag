#version 330 core
// FXAA 3.11 (console-quality subset) — cheap post anti-aliasing on the final
// tonemapped LDR image.  Runs as the very last pass (the HDR scene buffer can
// lose hardware MSAA, so this restores smooth edges without an MS render
// target).  Operates in gamma/LDR space, using luma-edge detection.
uniform sampler2D u_tex;       // tonemapped LDR (composite output)
in vec2 v_uv;
out vec4 frag_color;

const float EDGE_MIN = 1.0 / 24.0;
const float EDGE_MAX = 1.0 / 8.0;
const float SPAN_MAX = 8.0;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    vec2 px = 1.0 / vec2(textureSize(u_tex, 0));
    vec3 cM  = texture(u_tex, v_uv).rgb;
    float lM  = luma(cM);
    float lNW = luma(texture(u_tex, v_uv + vec2(-1.0, -1.0) * px).rgb);
    float lNE = luma(texture(u_tex, v_uv + vec2( 1.0, -1.0) * px).rgb);
    float lSW = luma(texture(u_tex, v_uv + vec2(-1.0,  1.0) * px).rgb);
    float lSE = luma(texture(u_tex, v_uv + vec2( 1.0,  1.0) * px).rgb);

    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
    if (lMax - lMin < max(EDGE_MIN, lMax * EDGE_MAX)) {
        frag_color = vec4(cM, 1.0);                 // no edge → leave it
        return;
    }

    vec2 dir;
    dir.x = -((lNW + lNE) - (lSW + lSE));
    dir.y =  ((lNW + lSW) - (lNE + lSE));
    float reduce = max((lNW + lNE + lSW + lSE) * 0.25 * (1.0 / 8.0), 1e-5);
    float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp, -SPAN_MAX, SPAN_MAX) * px;

    vec3 a = 0.5 * (texture(u_tex, v_uv + dir * (1.0 / 3.0 - 0.5)).rgb +
                    texture(u_tex, v_uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 b = a * 0.5 + 0.25 * (texture(u_tex, v_uv + dir * -0.5).rgb +
                               texture(u_tex, v_uv + dir *  0.5).rgb);
    float lB = luma(b);
    frag_color = vec4((lB < lMin || lB > lMax) ? a : b, 1.0);
}
