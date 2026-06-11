
#version 330 core
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
out vec3 v_world;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_world = (p3d_ModelMatrix * p3d_Vertex).xyz;
}
