"""
world/terrain_shader.py — GLSL surface shader for GPU-volumetric-lit terrain.

Replaces the fixed-function texture × baked-vertex-colour pipeline when
``config.lighting_backend == "gpu"``.  Per fragment it:

1. samples **direct sun/moon** light through the cascade visibility volume
   (voxel-marched shadows, computed in `lighting/glsl.py` INJECT) with
   Lambert shading on the (optionally normal-mapped) surface normal,
2. samples **indirect GI** from the flood-fill radiance cascades, with the
   sample position quantised to ``light_quant_m`` (0.25 m) — the pixelated
   "light pixels" look (2×2×2 per terrain voxel),
3. applies **voxel AO** from the occupancy volume,
4. adds **emission** (own-cell volume emission + per-texel emission map),
5. composites **volumetric fog** by one tap into the integrated froxel
   texture at this fragment's screen position + depth (god rays included),
6. tonemaps (ACES approximation) and gamma-encodes.

Texture stages per material Geom (built by `world/geometry_bridge.py`):
``p3d_Texture0`` albedo, ``p3d_Texture1`` tangent-space normal map,
``p3d_Texture2`` emission map.  The TBN basis is analytic from the dominant
normal axis — exactly the axis pair the mesher uses for planar UVs.

All lighting inputs (samplers + uniforms) are bound by
``GpuLightingPipeline.bind_surface_inputs`` / ``update_surface_inputs``.

Example
-------
    from fire_engine.world.terrain_shader import apply_terrain_shader
    apply_terrain_shader(app.terrain_root, pipeline)   # once at boot
"""

from __future__ import annotations

from panda3d.core import NodePath, Shader  # type: ignore[import]

__all__ = ["apply_terrain_shader", "TERRAIN_VERTEX", "TERRAIN_FRAGMENT"]


TERRAIN_VERTEX = """#version 330 core
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
    v_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    v_uv     = p3d_MultiTexCoord0;
    v_color  = p3d_Color;     // facet accent (light is NOT baked on gpu path)
}
"""


TERRAIN_FRAGMENT = """#version 330 core
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

uniform sampler2D p3d_Texture0;   // albedo
uniform sampler2D p3d_Texture1;   // tangent-space normal map
uniform sampler2D p3d_Texture2;   // emission map (linear HDR/8-bit)

// --- radiance cascades (lighting/gpu.py contract) -----------------------
uniform sampler3D u_c0_radiance;
uniform sampler3D u_c0_vis;       // r sun, g moon, b sky visibility
uniform sampler3D u_c0_geom;      // rgb albedo, a occupancy
uniform sampler3D u_c0_emis;
uniform vec3  u_c0_origin_m;
uniform float u_c0_cell_m;
uniform float u_c0_cells;
uniform sampler3D u_c1_radiance;
uniform sampler3D u_c1_vis;
uniform sampler3D u_c1_geom;
uniform vec3  u_c1_origin_m;
uniform float u_c1_cell_m;
uniform float u_c1_cells;

uniform vec3  u_sun_dir;
uniform vec3  u_sun_radiance;
uniform vec3  u_moon_dir;
uniform vec3  u_moon_radiance;
uniform vec3  u_sky_ambient;
uniform float u_quant_m;          // light-pixel size (0.25 m)
uniform float u_ao_strength;
uniform float u_exposure;
uniform float u_emission_scale;

// --- froxel fog ----------------------------------------------------------
uniform sampler3D u_fog_integrated;  // rgb accumulated light, a transmittance
uniform float u_fog_near;
uniform float u_fog_far;
uniform float u_fog_enabled;
uniform vec2  u_viewport;
uniform vec3  u_cam_pos;

vec3 c_uv(vec3 wp, vec3 origin, float cell_m, float cells) {
    return (wp - origin) / (cell_m * cells);
}

bool inBox(vec3 uv, float pad) {
    return all(greaterThan(uv, vec3(pad))) && all(lessThan(uv, vec3(1.0 - pad)));
}

// Sample a cascade triple (radiance, vis, occupancy) with cascade-0 priority.
void sampleCascades(vec3 wp, out vec3 radiance, out vec3 vis, out float occ) {
    vec3 uv0 = c_uv(wp, u_c0_origin_m, u_c0_cell_m, u_c0_cells);
    if (inBox(uv0, 0.02)) {
        radiance = texture(u_c0_radiance, uv0).rgb;
        vis      = texture(u_c0_vis, uv0).rgb;
        occ      = texture(u_c0_geom, uv0).a;
        return;
    }
    vec3 uv1 = c_uv(wp, u_c1_origin_m, u_c1_cell_m, u_c1_cells);
    if (inBox(uv1, 0.01)) {
        radiance = texture(u_c1_radiance, uv1).rgb;
        vis      = texture(u_c1_vis, uv1).rgb;
        occ      = texture(u_c1_geom, uv1).a;
        return;
    }
    radiance = u_sky_ambient * 0.6;       // beyond all cascades: open sky guess
    vis      = vec3(1.0);
    occ      = 0.0;
}

vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

void main() {
    // ------------------------------------------------------------------
    // Surface basis + normal map (TBN from the dominant axis, matching the
    // mesher's planar UV projection: X-facing→(Y,Z), Y→(X,Z), Z→(X,Y)).
    // ------------------------------------------------------------------
    vec3 n  = normalize(v_normal);
    vec3 an = abs(n);
    vec3 t, b;
    if (an.x >= an.y && an.x >= an.z)      { t = vec3(0, 1, 0); b = vec3(0, 0, 1); }
    else if (an.y >= an.z)                 { t = vec3(1, 0, 0); b = vec3(0, 0, 1); }
    else                                   { t = vec3(1, 0, 0); b = vec3(0, 1, 0); }
    t = normalize(t - n * dot(n, t));
    b = normalize(b - n * dot(n, b) - t * dot(t, b));
    vec3 nm = texture(p3d_Texture1, v_uv).xyz * 2.0 - 1.0;
    vec3 N  = normalize(t * nm.x + b * nm.y + n * max(nm.z, 0.3));

    // ------------------------------------------------------------------
    // Light sampling — positions quantised to the light-pixel grid so the
    // lighting itself is visibly pixelated (2x2x2 light pixels per voxel).
    // ------------------------------------------------------------------
    vec3 wq = (floor(v_world / u_quant_m) + 0.5) * u_quant_m;
    // Shadow/GI probes hop off the surface along the *face* normal.
    vec3 probe   = wq + n * (u_c0_cell_m * 0.75);
    vec3 radiance, vis;
    float occ;
    sampleCascades(probe, radiance, vis, occ);

    // Voxel AO: occupancy a little farther out along the normal + above.
    vec3 aoR, aoV; float occFar;
    sampleCascades(wq + n * (u_c0_cell_m * 1.6), aoR, aoV, occFar);
    float ao = 1.0 - u_ao_strength * clamp(0.5 * occ + 0.7 * occFar, 0.0, 1.0);

    // ------------------------------------------------------------------
    // Compose: direct celestial + flood-fill GI + emission.
    // ------------------------------------------------------------------
    vec3 base = pow(texture(p3d_Texture0, v_uv).rgb, vec3(2.2))
              * v_color.rgb;

    vec3 direct =
        u_sun_radiance  * (vis.r * max(dot(N, u_sun_dir),  0.0)) +
        u_moon_radiance * (vis.g * max(dot(N, u_moon_dir), 0.0));

    vec3 ownEmis = vec3(0.0);
    vec3 uv0 = c_uv(v_world - n * (u_c0_cell_m * 0.25),
                    u_c0_origin_m, u_c0_cell_m, u_c0_cells);
    if (inBox(uv0, 0.0))
        ownEmis = texture(u_c0_emis, uv0).rgb * u_emission_scale;
    vec3 mapEmis = pow(texture(p3d_Texture2, v_uv).rgb, vec3(2.2)) * 4.0;

    vec3 hdr = base * (direct + radiance * ao) + ownEmis + mapEmis;

    // ------------------------------------------------------------------
    // Volumetric fog composite (one tap into the integrated froxels).
    // ------------------------------------------------------------------
    if (u_fog_enabled > 0.5) {
        float dist = length(v_world - u_cam_pos);
        float w = log(max(dist, u_fog_near) / u_fog_near)
                / log(u_fog_far / u_fog_near);
        vec2 suv = gl_FragCoord.xy / u_viewport;
        vec4 fog = texture(u_fog_integrated, vec3(suv, clamp(w, 0.0, 1.0)));
        hdr = hdr * fog.a + fog.rgb;
    }

    vec3 ldr = acesTonemap(hdr * u_exposure);
    frag_color = vec4(pow(ldr, vec3(1.0 / 2.2)), 1.0);
}
"""


def apply_terrain_shader(terrain_root: NodePath, pipeline) -> None:
    """
    Compile and apply the volumetric terrain shader to ``terrain_root``.

    Binds the pipeline's static lighting inputs immediately; the App's frame
    loop must call ``pipeline.update_surface_inputs(terrain_root, sky_state)``
    each frame (window origins, radiance ping-pong, sun/moon uniforms).

    Parameters
    ----------
    terrain_root : NodePath
        Parent of every chunk Geom (``App.terrain_root``).
    pipeline : GpuLightingPipeline
        The active lighting pipeline (`lighting/gpu.py`).
    """
    shader = Shader.make(Shader.SL_GLSL,
                         vertex=TERRAIN_VERTEX,
                         fragment=TERRAIN_FRAGMENT)
    terrain_root.set_shader(shader)
    pipeline.bind_surface_inputs(terrain_root)
