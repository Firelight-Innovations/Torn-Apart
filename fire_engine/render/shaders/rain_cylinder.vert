#version 330 core
// rain_cylinder.vert — cheap "cylinders" rain mode (M6 low preset).
//
// The old sky_renderer cylinder rain, moved into the rain module so it can
// honour the SAME rain-cover heightmap cull + weather-map precip gate as the
// particle mode (so even the cheap mode stops raining under roofs).  The
// geometry is the nested open cylinders built CPU-side; this shader passes the
// fragment its WORLD position + scrolled UV so the fragment can sample the
// cover heightmap and precip at its own world XY.

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;     // model → world (the cylinder follows the cam)

uniform vec2  u_uv_scroll;        // per-layer UV scroll offset (set per frame)

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

out vec2 v_uv;
out vec3 v_world;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_uv = p3d_MultiTexCoord0 + u_uv_scroll;
    v_world = (p3d_ModelMatrix * p3d_Vertex).xyz;
}
