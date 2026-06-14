#version 330 core
// lightning.vert — camera-facing ribbon expansion for procedural bolts (M7).
//
// Each bolt SEGMENT (a→b in world space) is uploaded as a 4-vertex quad.  Every
// vertex carries this endpoint (p3d_Vertex), the segment's OTHER endpoint
// (u_other / a_other) and a (side, alongT, width, brightness) tuple, so the
// vertex shader can billboard the segment into a flat ribbon that always faces
// the camera: it offsets each vertex sideways along (segment_dir × view_dir),
// scaled by the segment width.  Width grows a touch with distance so a far bolt
// stays visible (a minimum screen-ish thickness).
//
// Two-phase reveal/flash envelope is driven by uniforms (set per frame by the
// LightningRendererComponent):
//   u_reveal   0..1  — how far DOWN the channel is currently lit (top-down
//                      leader growth): a vertex with alongT > u_reveal is hidden.
//   u_flash    HDR   — global brightness multiplier for the current phase
//                      (flickering leader → bright return stroke → afterglow).
// The bolt is parented under terrain_root, so u_cam_pos is inherited.

uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ProjectionMatrix;
uniform vec3 u_cam_pos;          // camera world pos (inherited from terrain_root)

uniform float u_reveal;          // 0..1 top-down reveal front
uniform float u_flash;           // HDR brightness multiplier
uniform float u_width_scale;     // global ribbon width multiplier (m)

in vec4 p3d_Vertex;              // THIS endpoint world XYZ
in vec3 a_other;                 // the segment's OTHER endpoint world XYZ
in vec4 a_ribbon;                // x=side(-1/+1) y=alongT(0..1) z=width w=brightness

out float v_bright;              // per-fragment brightness (flash × seg brightness)
out float v_cross;               // -1..1 across the ribbon (for the soft core)

void main() {
    vec3 here  = p3d_Vertex.xyz;
    vec3 other = a_other;
    float side   = a_ribbon.x;
    float alongT = a_ribbon.y;
    float width  = a_ribbon.z;
    float seg_br = a_ribbon.w;

    // Hide segments below the reveal front (top-down leader growth).  A small
    // soft band keeps the leading tip from popping.
    float revealed = step(alongT, u_reveal + 0.02);

    // Ribbon axis = the segment direction; offset axis = axis × view_dir so the
    // flat ribbon faces the camera.
    vec3 seg_dir = normalize(other - here + vec3(1e-5));
    vec3 view_dir = normalize(here - u_cam_pos);
    vec3 offset_axis = normalize(cross(seg_dir, view_dir) + vec3(1e-5));

    // Distance-aware width: keep far bolts from vanishing to a sub-pixel line.
    float dist = length(here - u_cam_pos);
    float w = width * u_width_scale * (1.0 + dist * 0.0015);

    vec3 world = here + offset_axis * (side * w) * revealed;

    gl_Position = p3d_ProjectionMatrix * (p3d_ModelViewMatrix * vec4(world, 1.0));

    v_bright = seg_br * u_flash * revealed;
    v_cross  = side;
}
