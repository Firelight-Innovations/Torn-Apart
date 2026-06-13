#version 330 core
// building.vert — free-form building geometry.
//
// The building mesh is emitted in BUILDING-LOCAL space (buildings/meshing.py);
// the renderer puts the building's position + rotation on the node, so
// p3d_ModelMatrix carries that transform.  v_world MUST come from
// p3d_ModelMatrix * p3d_Vertex (NOT the raw vertex, unlike terrain chunks
// which sit at the world origin) or lit_surface.glsl samples the cascades at
// the wrong world cell.  Moving/rotating a building is a transform write here,
// never a remesh.
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;
in vec4 p3d_Color;

out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_color;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_world  = (p3d_ModelMatrix * p3d_Vertex).xyz;
    v_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);  // rotated normal
    v_uv     = p3d_MultiTexCoord0;
    v_color  = p3d_Color;     // flat white from the mesher (unused; reserved)
}
