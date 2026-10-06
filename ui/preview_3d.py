"""Real-time material preview: a sphere/cube/plane rendered with a
simplified Cook-Torrance PBR shader, fed by whatever Albedo/Normal/
Roughness textures the project currently has.

HDRI handling is a middle ground, not full image-based lighting (no
cubemap convolution / specular pre-filtering): the selected equirect image
is rendered as an actual visible backdrop behind the object (each pixel's
view ray is reconstructed from the camera basis + FOV and used to sample
the equirect texture directly, so it orbits correctly with the camera),
and its average color separately tints the ambient lighting term. This
keeps the widget fully independent of material-generation parameters; it
only consumes final texture data and the values from PreviewPropertiesPanel.
"""
from __future__ import annotations

import ctypes
import math
from typing import Optional

import numpy as np
from OpenGL import GL as gl
from PIL import Image
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from ui import trackpad

# Default environment when no HDRI is selected: a procedural sky in the style
# of Unity's default skybox (blue zenith, bright horizon, grey ground and a
# sun disk aligned with the key light). It is drawn as the viewport backdrop
# and also lights the object (hemispherical ambient + sky reflections), so
# materials read well out of the box instead of sitting in near-black ambient.
SKY_GLSL = """
const vec3 SKY_ZENITH = vec3(0.32, 0.47, 0.72);
const vec3 SKY_HORIZON = vec3(0.80, 0.82, 0.86);
const vec3 SKY_GROUND = vec3(0.37, 0.35, 0.34);

// Sky radiance in display (sRGB-ish) space. `sunDir` points towards the sun.
vec3 proceduralSky(vec3 d, vec3 sunDir, bool withSunDisk) {
    float y = d.y;
    vec3 c;
    if (y >= 0.0) {
        c = mix(SKY_HORIZON, SKY_ZENITH, pow(clamp(y, 0.0, 1.0), 0.5));
    } else {
        c = mix(SKY_HORIZON * 0.85, SKY_GROUND, smoothstep(0.0, 0.12, -y));
    }
    float s = max(dot(d, sunDir), 0.0);
    c += vec3(1.0, 0.92, 0.78) * (pow(s, 48.0) * 0.35);
    if (withSunDisk && y > -0.02) {
        c += vec3(1.0, 0.96, 0.88) * smoothstep(0.9993, 0.9997, s) * 1.5;
    }
    return c;
}
"""

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inTangent;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform float uUVScale;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;
out vec3 vTangent;

void main() {
    vec4 world = uModel * vec4(inPosition, 1.0);
    vWorldPos = world.xyz;
    vNormal = mat3(uModel) * inNormal;
    vTangent = mat3(uModel) * inTangent;
    vUV = inUV * uUVScale;
    gl_Position = uProj * uView * world;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in vec3 vTangent;
out vec4 outColor;

uniform vec3 uCamPos;
uniform vec3 uLightDir;
uniform float uLightIntensity;
uniform float uEnvIntensity;
uniform float uExposure;
uniform vec3 uAmbientColor;

uniform sampler2D uAlbedoTex;
uniform bool uHasAlbedo;
uniform sampler2D uNormalTex;
uniform bool uHasNormal;
uniform sampler2D uRoughnessTex;
uniform bool uHasRoughnessTex;
uniform float uRoughnessValue;
uniform sampler2D uHeightTex;
uniform bool uHasHeight;
uniform sampler2D uAOTex;
uniform bool uHasAO;
uniform bool uPOMEnabled;
uniform float uPOMHeightScale;
uniform int uPOMMaxLayers;
uniform bool uSkyAmbient;

const float PI = 3.14159265359;
""" + SKY_GLSL + """
// Diffuse irradiance of the procedural sky for a surface facing `n`
// (linear space): ground below, sky above, sun glow towards the light.
vec3 skyIrradiance(vec3 n, vec3 sunDir) {
    vec3 skyAvg = pow(mix(SKY_HORIZON, SKY_ZENITH, 0.55), vec3(2.2));
    vec3 ground = pow(SKY_GROUND, vec3(2.2));
    vec3 c = mix(ground, skyAvg, clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    return c + vec3(1.0, 0.92, 0.78) * 0.08 * max(dot(n, sunDir), 0.0);
}

vec2 parallaxOcclusionMapping(vec2 texCoords, vec3 viewDirTS) {
    float maxLayers = float(max(uPOMMaxLayers, 4));
    float minLayers = max(4.0, maxLayers * 0.25);
    // At grazing angles the parallax error is most visible, so spend more
    // samples there.  The cap keeps the viewport predictable on integrated GPUs.
    float layerCount = mix(maxLayers, minLayers, abs(viewDirTS.z));
    float layerDepth = 1.0 / layerCount;
    vec2 deltaUV = (viewDirTS.xy / max(viewDirTS.z, 0.15))
        * (uPOMHeightScale / layerCount);

    float currentLayerDepth = 0.0;
    vec2 currentUV = texCoords;
    // PBRCELAIN height maps use white = high and black = low.  Ray marching
    // proceeds from the viewer into the surface, so it needs the inverse
    // representation: black = deepest point to traverse.
    float sampledDepth = 1.0 - texture(uHeightTex, currentUV).r;
    if (sampledDepth <= 0.0) {
        return texCoords;
    }
    for (int i = 0; i < 64; ++i) {
        if (float(i) >= layerCount || currentLayerDepth >= sampledDepth) {
            break;
        }
        currentUV -= deltaUV;
        sampledDepth = 1.0 - texture(uHeightTex, currentUV).r;
        currentLayerDepth += layerDepth;
    }

    // Interpolate across the final ray-march interval to avoid visible layer bands.
    vec2 previousUV = currentUV + deltaUV;
    float afterDepth = sampledDepth - currentLayerDepth;
    float beforeDepth = (1.0 - texture(uHeightTex, previousUV).r) - currentLayerDepth + layerDepth;
    float denominator = afterDepth - beforeDepth;
    float weight = abs(denominator) > 1e-5 ? afterDepth / denominator : 0.0;
    weight = clamp(weight, 0.0, 1.0);
    return previousUV * weight + currentUV * (1.0 - weight);
}

void main() {
    vec3 N = normalize(vNormal);
    // The UV sphere's tangent degenerates at the poles (vTangent ~ 0 there);
    // fall back to an arbitrary tangent orthogonal to N instead of
    // normalizing a near-zero vector (which produces NaNs).
    vec3 rawT = vTangent - N * dot(vTangent, N);
    // Pick a fallback axis guaranteed not to be parallel to N (a fixed axis
    // like (0,0,1) can itself coincide with N somewhere on a full sphere,
    // which would make this cross product degenerate too and normalize()
    // would return NaN — exactly the kind of bright/broken pixel this
    // fallback exists to avoid).
    vec3 fallbackAxis = abs(N.z) > 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 T = length(rawT) > 1e-4 ? normalize(rawT) : normalize(cross(fallbackAxis, N));
    vec3 B = cross(N, T);

    vec2 uv = vUV;
    vec3 V = normalize(uCamPos - vWorldPos);
    if (uPOMEnabled && uHasHeight && uPOMHeightScale > 0.0) {
        vec3 viewDirTS = vec3(dot(V, T), dot(V, B), dot(V, N));
        uv = parallaxOcclusionMapping(uv, viewDirTS);
    }

    vec3 albedo = uHasAlbedo ? texture(uAlbedoTex, uv).rgb : vec3(0.75);
    albedo = pow(albedo, vec3(2.2));
    float roughness = uHasRoughnessTex ? texture(uRoughnessTex, uv).r : uRoughnessValue;
    roughness = clamp(roughness, 0.05, 1.0);

    if (uHasNormal) {
        vec3 tex = texture(uNormalTex, uv).rgb * 2.0 - 1.0;
        mat3 TBN = mat3(T, B, N);
        N = normalize(TBN * tex);
    }

    vec3 L = normalize(-uLightDir);
    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0);
    float D = alpha2 / (PI * denom * denom + 1e-6);

    float k = (roughness + 1.0);
    k = (k * k) / 8.0;
    float G1V = NdotV / (NdotV * (1.0 - k) + k);
    float G1L = NdotL / (NdotL * (1.0 - k) + k);
    float G = G1V * G1L;

    vec3 F0 = vec3(0.04);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4);
    vec3 kd = (1.0 - F);
    vec3 diffuse = kd * albedo / PI;

    float ao = uHasAO ? texture(uAOTex, uv).r : 1.0;
    vec3 color = (diffuse + specular) * uLightIntensity * NdotL;
    if (uSkyAmbient) {
        // Image-based lighting from the procedural sky: diffuse irradiance
        // plus a reflection of the sky that blurs towards the irradiance as
        // roughness rises, weighted by a roughness-aware Fresnel term.
        vec3 Fenv = F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - NdotV, 5.0);
        vec3 R = reflect(-V, N);
        vec3 reflected = pow(proceduralSky(R, L, false), vec3(2.2));
        vec3 envSpec = mix(reflected, skyIrradiance(R, L), roughness);
        vec3 envDiffuse = (1.0 - Fenv) * albedo * skyIrradiance(N, L);
        color += (envDiffuse + envSpec * Fenv) * uEnvIntensity * ao;
    } else {
        color += albedo * uAmbientColor * uEnvIntensity * ao;
    }
    color *= uExposure;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
"""

GRID_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 inPosition;
uniform mat4 uView;
uniform mat4 uProj;
void main() { gl_Position = uProj * uView * vec4(inPosition, 1.0); }
"""

GRID_FRAGMENT_SHADER = """
#version 330 core
out vec4 outColor;
uniform vec3 uColor;
void main() { outColor = vec4(uColor, 1.0); }
"""

BACKGROUND_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 inClipPos;
out vec2 vClip;
void main() {
    vClip = inClipPos;
    gl_Position = vec4(inClipPos, 0.9999, 1.0);
}
"""

BACKGROUND_FRAGMENT_SHADER = """
#version 330 core
in vec2 vClip;
out vec4 outColor;

uniform vec3 uCamForward;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform float uTanHalfFov;
uniform float uAspect;
uniform sampler2D uHdriTex;
uniform bool uHasHdri;
uniform vec3 uFlatColor;
uniform float uBrightness;
uniform vec3 uSunDir;

const float PI = 3.14159265359;
""" + SKY_GLSL + """

void main() {
    vec3 color;
    if (uHasHdri) {
        vec3 dir = normalize(
            uCamForward
            + vClip.x * uAspect * uTanHalfFov * uCamRight
            + vClip.y * uTanHalfFov * uCamUp
        );
        float u = 0.5 + atan(dir.z, dir.x) / (2.0 * PI);
        float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
        color = texture(uHdriTex, vec2(u, v)).rgb;
    } else {
        vec3 dir = normalize(
            uCamForward
            + vClip.x * uAspect * uTanHalfFov * uCamRight
            + vClip.y * uTanHalfFov * uCamUp
        );
        color = proceduralSky(dir, uSunDir, true);
    }
    outColor = vec4(color * uBrightness, 1.0);
}
"""


def configure_surface_format() -> None:
    """Request an OpenGL 3.3 core profile. Call before QApplication is created."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


def _build_uv_sphere(rings: int = 40, segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Returns (vertices, indices). Each vertex: pos(3) normal(3) uv(2) tangent(3) = 11 floats."""
    verts = []
    for r in range(rings + 1):
        theta = r * math.pi / rings
        for s in range(segments + 1):
            phi = s * 2.0 * math.pi / segments
            x = math.sin(theta) * math.cos(phi)
            y = math.cos(theta)
            z = math.sin(theta) * math.sin(phi)
            u = s / segments
            v = r / rings
            tx = -math.sin(phi)
            ty = 0.0
            tz = math.cos(phi)
            verts.extend([x, y, z, x, y, z, u, v, tx, ty, tz])

    indices = []
    stride = segments + 1
    for r in range(rings):
        for s in range(segments):
            i0 = r * stride + s
            i1 = i0 + stride
            indices.extend([i0, i1, i0 + 1, i0 + 1, i1, i1 + 1])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)


def _build_cube(half_extent: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (vertices, indices) for a cube, 4 unshared verts per face so
    each face keeps its own flat normal/tangent/UV (0..1 per face)."""
    s = half_extent
    faces = [
        ((1, 0, 0), (0, 0, -1), [(s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)]),
        ((-1, 0, 0), (0, 0, 1), [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)]),
        ((0, 1, 0), (1, 0, 0), [(-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s)]),
        ((0, -1, 0), (1, 0, 0), [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)]),
        ((0, 0, 1), (1, 0, 0), [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]),
        ((0, 0, -1), (-1, 0, 0), [(s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s)]),
    ]
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    verts: list[float] = []
    indices: list[int] = []
    base = 0
    for normal, tangent, corners in faces:
        for corner, uv in zip(corners, uvs):
            verts.extend([*corner, *normal, *uv, *tangent])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        base += 4

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)


def _build_plane(half_extent: float = 1.4) -> tuple[np.ndarray, np.ndarray]:
    """Returns (vertices, indices) for a flat horizontal square (normal +Y)."""
    s = half_extent
    normal = (0.0, 1.0, 0.0)
    tangent = (1.0, 0.0, 0.0)
    corners = [(-s, 0.0, s), (s, 0.0, s), (s, 0.0, -s), (-s, 0.0, -s)]
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    verts: list[float] = []
    for corner, uv in zip(corners, uvs):
        verts.extend([*corner, *normal, *uv, *tangent])
    indices = [0, 1, 2, 0, 2, 3]

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)


def _build_grid(size: float = 4.0, divisions: int = 8) -> np.ndarray:
    y = -0.001
    lines = []
    step = (2 * size) / divisions
    for i in range(divisions + 1):
        c = -size + i * step
        lines.extend([c, y, -size, c, y, size])
        lines.extend([-size, y, c, size, y, c])
    return np.array(lines, dtype=np.float32)


def _build_fullscreen_triangle() -> np.ndarray:
    """One oversized triangle covering the whole clip-space viewport — the
    standard fullscreen-pass trick, cheaper than a quad (no shared edge)."""
    return np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype=np.float32)


class Preview3DWidget(QOpenGLWidget):
    distanceChanged = pyqtSignal(float)
    lightChanged = pyqtSignal(float, float)
    autoRotateToggled = pyqtSignal(bool)
    pivotChanged = pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFormat(QSurfaceFormat.defaultFormat())
        self.setMinimumHeight(260)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._azimuth = 35.0
        self._elevation = 20.0
        self._last_mouse_pos = None
        self._is_dragging = False

        self._settings = {
            "mesh": "sphere",
            "fov": 75.0,
            "distance": 3.5,
            "rotation_speed": 0.5,
            "pivot_x": 0.0,
            "pivot_y": 0.0,
            "pivot_z": 0.0,
            "light_intensity": 1.2,
            "env_intensity": 1.0,
            "exposure": 1.0,
            "show_grid": False,
            "auto_rotate": True,
            "wireframe": False,
            "pom_enabled": True,
            "pom_height_scale": 0.04,
            "pom_max_layers": 32,
            "background_brightness": 1.0,
            "light_azimuth": 45.0,
            "light_elevation": 50.0,
            "uv_tiling": 1.0,
        }
        self._ambient_color = (0.05, 0.05, 0.06)
        self._background_color = (0.05, 0.05, 0.06)

        self._albedo_image: Optional[Image.Image] = None
        self._normal_image: Optional[Image.Image] = None
        self._roughness_image: Optional[Image.Image] = None
        self._height_image: Optional[Image.Image] = None
        self._ao_image: Optional[Image.Image] = None
        self._albedo_source: Optional[Image.Image] = None
        self._normal_source: Optional[Image.Image] = None
        self._roughness_source: Optional[Image.Image] = None
        self._height_source: Optional[Image.Image] = None
        self._ao_source: Optional[Image.Image] = None
        self._texture_upload_needed: set[str] = set()
        self._roughness_value = 0.5
        self._hdri_image: Optional[Image.Image] = None
        self._hdri_pending_upload = False

        self._gl_ready = False
        self._pending_texture_upload = False

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def set_textures(
        self,
        albedo: Optional[Image.Image],
        normal: Optional[Image.Image],
        roughness: Optional[Image.Image] = None,
        height: Optional[Image.Image] = None,
        ao: Optional[Image.Image] = None,
    ) -> None:
        changed: set[str] = set()
        if albedo is not self._albedo_source:
            self._albedo_source = albedo
            self._albedo_image = albedo.convert("RGB") if albedo is not None else None
            changed.add("albedo")
        if normal is not self._normal_source:
            self._normal_source = normal
            self._normal_image = normal.convert("RGB") if normal is not None else None
            changed.add("normal")
        if roughness is not self._roughness_source:
            self._roughness_source = roughness
            self._roughness_image = roughness.convert("L") if roughness is not None else None
            changed.add("roughness")
        if height is not self._height_source:
            self._height_source = height
            if height is not None:
                if height.mode in ("I;16", "I"):
                    arr = (np.asarray(height, dtype=np.float32) / 256.0).clip(0, 255).astype(np.uint8)
                    self._height_image = Image.fromarray(arr, mode="L")
                elif height.mode == "F":
                    arr = (np.asarray(height, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
                    self._height_image = Image.fromarray(arr, mode="L")
                else:
                    self._height_image = height.convert("L")
            else:
                self._height_image = None
            changed.add("height")
        if ao is not self._ao_source:
            self._ao_source = ao
            self._ao_image = ao.convert("L") if ao is not None else None
            changed.add("ao")

        self._texture_upload_needed.update(changed)
        if self._gl_ready and changed:
            self._upload_textures()
        elif changed:
            self._pending_texture_upload = True
        self.update()

    def set_preview_settings(self, data: dict) -> None:
        self._settings.update(data)
        hdri_path = data.get("hdri_path")
        if hdri_path != getattr(self, "_hdri_path_loaded", None):
            self._hdri_path_loaded = hdri_path
            self._update_environment_color(hdri_path)
        self.update()

    def _update_environment_color(self, hdri_path: Optional[str]) -> None:
        if not hdri_path:
            self._ambient_color = (0.05, 0.05, 0.06)
            self._background_color = (0.05, 0.05, 0.06)
            self._hdri_image = None
            self._queue_hdri_upload()
            return
        try:
            with Image.open(hdri_path) as raw:
                rgb = raw.convert("RGB")
                rgb.thumbnail((2048, 1024))
                self._hdri_image = rgb.copy()
                img_small = rgb.resize((32, 16))
            arr = np.asarray(img_small, dtype=np.float32) / 255.0
            avg = arr.reshape(-1, 3).mean(axis=0)
            self._ambient_color = tuple(float(c) for c in avg)
            self._background_color = tuple(float(c) for c in avg)
        except Exception:
            self._ambient_color = (0.05, 0.05, 0.06)
            self._background_color = (0.05, 0.05, 0.06)
            self._hdri_image = None
        self._queue_hdri_upload()

    def _queue_hdri_upload(self) -> None:
        if self._gl_ready:
            self._upload_hdri_texture()
        else:
            self._hdri_pending_upload = True

    def initializeGL(self) -> None:
        # macOS: Qt's context setup can leave a stale GL_INVALID_ENUM in the
        # error queue, which PyOpenGL would attribute to our first call.
        while gl.glGetError() != gl.GL_NO_ERROR:
            pass
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        self._program = self._build_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self._grid_program = self._build_program(GRID_VERTEX_SHADER, GRID_FRAGMENT_SHADER)
        self._uniforms = self._cache_uniform_locations(self._program, [
            "uModel", "uView", "uProj", "uCamPos", "uLightDir", "uLightIntensity",
            "uEnvIntensity", "uExposure", "uAmbientColor", "uRoughnessValue",
            "uHasAlbedo", "uHasNormal", "uHasRoughnessTex",
            "uHasHeight", "uPOMEnabled", "uPOMHeightScale", "uPOMMaxLayers",
            "uAlbedoTex", "uNormalTex", "uRoughnessTex", "uHeightTex", "uUVScale",
            "uAOTex", "uHasAO", "uSkyAmbient",
        ])
        self._grid_uniforms = self._cache_uniform_locations(self._grid_program, ["uView", "uProj", "uColor"])

        self._bg_program = self._build_program(BACKGROUND_VERTEX_SHADER, BACKGROUND_FRAGMENT_SHADER)
        self._bg_uniforms = self._cache_uniform_locations(self._bg_program, [
            "uCamForward", "uCamRight", "uCamUp", "uTanHalfFov", "uAspect",
            "uHdriTex", "uHasHdri", "uFlatColor", "uBrightness", "uSunDir",
        ])
        bg_verts = _build_fullscreen_triangle()
        self._bg_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._bg_vao)
        bg_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, bg_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, bg_verts.nbytes, bg_verts, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 2 * 4, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)

        self._meshes = {
            "sphere": self._upload_mesh(*_build_uv_sphere()),
            "cube": self._upload_mesh(*_build_cube()),
            "plane": self._upload_mesh(*_build_plane()),
        }

        grid_verts = _build_grid()
        self._grid_vertex_count = len(grid_verts) // 3
        self._grid_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._grid_vao)
        grid_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, grid_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, grid_verts.nbytes, grid_verts, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)

        self._albedo_tex = self._make_placeholder_texture((191, 191, 191))
        self._normal_tex = self._make_placeholder_texture((128, 128, 255))
        self._roughness_tex = self._make_placeholder_texture((128, 128, 128))
        self._height_tex = self._make_placeholder_texture((0, 0, 0))
        self._ao_tex = self._make_placeholder_texture((255, 255, 255))
        self._hdri_tex = self._make_placeholder_texture((13, 13, 15))

        self._gl_ready = True
        if self._pending_texture_upload:
            self._upload_textures()
            self._pending_texture_upload = False
        if self._hdri_pending_upload:
            self._upload_hdri_texture()
            self._hdri_pending_upload = False

    def _build_program(self, vs_src: str, fs_src: str) -> int:
        vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vs, vs_src)
        gl.glCompileShader(vs)
        if not gl.glGetShaderiv(vs, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(vs).decode())

        fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fs, fs_src)
        gl.glCompileShader(fs)
        if not gl.glGetShaderiv(fs, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(fs).decode())

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vs)
        gl.glAttachShader(program, fs)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(program).decode())
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return program

    @staticmethod
    def _cache_uniform_locations(program: int, names: list[str]) -> dict[str, int]:
        return {name: gl.glGetUniformLocation(program, name) for name in names}

    @staticmethod
    def _upload_mesh(verts: np.ndarray, indices: np.ndarray) -> tuple[int, int]:
        """Uploads a pos(3)/normal(3)/uv(2)/tangent(3) mesh, returns (vao, index_count)."""
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        stride = 11 * 4
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(6 * 4))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8 * 4))
        gl.glEnableVertexAttribArray(3)
        gl.glBindVertexArray(0)
        return vao, len(indices)

    def _make_placeholder_texture(self, rgb: tuple[int, int, int]) -> int:
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        data = bytes(rgb) * 4
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, 2, 2, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        return tex

    def _upload_image_texture(self, tex: int, image: Image.Image) -> None:
        arr = np.asarray(image)
        h, w = arr.shape[0], arr.shape[1]
        fmt = gl.GL_RGB if arr.ndim == 3 and arr.shape[2] == 3 else gl.GL_RED
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB if fmt == gl.GL_RGB else gl.GL_RED,
                         w, h, 0, fmt, gl.GL_UNSIGNED_BYTE, np.ascontiguousarray(arr))
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

    def _upload_textures(self) -> None:
        pending = self._texture_upload_needed.copy()
        if not pending:
            return
        self.makeCurrent()
        if "albedo" in pending and self._albedo_image is not None:
            self._upload_image_texture(self._albedo_tex, self._albedo_image)
        if "normal" in pending and self._normal_image is not None:
            self._upload_image_texture(self._normal_tex, self._normal_image)
        if "roughness" in pending and self._roughness_image is not None:
            self._upload_image_texture(self._roughness_tex, self._roughness_image)
        if "height" in pending and self._height_image is not None:
            self._upload_image_texture(self._height_tex, self._height_image)
        if "ao" in pending and self._ao_image is not None:
            self._upload_image_texture(self._ao_tex, self._ao_image)
        self.doneCurrent()
        self._texture_upload_needed.difference_update(pending)

    def _upload_hdri_texture(self) -> None:
        self.makeCurrent()
        if self._hdri_image is not None:
            self._upload_image_texture(self._hdri_tex, self._hdri_image)
        self.doneCurrent()

    def resizeGL(self, w: int, h: int) -> None:
        dpr = self.devicePixelRatio()
        gl.glViewport(0, 0, max(1, int(round(w * dpr))), max(1, int(round(h * dpr))))

    def _on_tick(self) -> None:
        if self._settings.get("auto_rotate") and not self._is_dragging:
            self._azimuth = (self._azimuth + self._settings.get("rotation_speed", 0.5)) % 360.0
            self.update()

    def paintGL(self) -> None:
        dpr = self.devicePixelRatio()
        gl.glViewport(0, 0, max(1, int(round(self.width() * dpr))), max(1, int(round(self.height() * dpr))))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        aspect = max(self.width(), 1) / max(self.height(), 1)
        fov = self._settings.get("fov", 45.0)
        proj = _perspective(fov, aspect, 0.01, 100.0)

        distance = self._settings.get("distance", 3.5)
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)
        orbit_offset = np.array([
            distance * math.cos(el) * math.sin(az),
            distance * math.sin(el),
            distance * math.cos(el) * math.cos(az),
        ], dtype=np.float32)

        pivot = np.array([
            self._settings.get("pivot_x", 0.0),
            self._settings.get("pivot_y", 0.0),
            self._settings.get("pivot_z", 0.0),
        ], dtype=np.float32)

        cam_pos = pivot + orbit_offset
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        view = _look_at(cam_pos, pivot, world_up)
        model = np.identity(4, dtype=np.float32)

        cam_forward = -orbit_offset / max(np.linalg.norm(orbit_offset), 1e-6)
        cam_right = np.cross(cam_forward, world_up)
        cam_right = cam_right / max(np.linalg.norm(cam_right), 1e-6)
        cam_up = np.cross(cam_right, cam_forward)

        light_az = math.radians(self._settings.get("light_azimuth", 45.0))
        light_el = math.radians(self._settings.get("light_elevation", 50.0))
        light_source_dir = (
            math.cos(light_el) * math.sin(light_az),
            math.sin(light_el),
            math.cos(light_el) * math.cos(light_az),
        )
        light_dir = tuple(-c for c in light_source_dir)

        bg_brightness = self._settings.get("background_brightness", 1.0)
        bu = self._bg_uniforms
        gl.glDepthMask(gl.GL_FALSE)
        gl.glUseProgram(self._bg_program)
        gl.glUniform3f(bu["uCamForward"], *cam_forward.tolist())
        gl.glUniform3f(bu["uCamRight"], *cam_right.tolist())
        gl.glUniform3f(bu["uCamUp"], *cam_up.tolist())
        gl.glUniform1f(bu["uTanHalfFov"], math.tan(math.radians(fov) / 2.0))
        gl.glUniform1f(bu["uAspect"], aspect)
        gl.glUniform1i(bu["uHasHdri"], 1 if self._hdri_image is not None else 0)
        gl.glUniform3f(bu["uFlatColor"], *self._background_color)
        gl.glUniform1f(bu["uBrightness"], bg_brightness)
        gl.glUniform3f(bu["uSunDir"], *light_source_dir)
        gl.glActiveTexture(gl.GL_TEXTURE4)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._hdri_tex)
        gl.glUniform1i(bu["uHdriTex"], 4)
        gl.glBindVertexArray(self._bg_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)
        gl.glDepthMask(gl.GL_TRUE)

        u = self._uniforms
        gl.glUseProgram(self._program)
        self._set_mat4(u["uModel"], model)
        self._set_mat4(u["uView"], view)
        self._set_mat4(u["uProj"], proj)
        gl.glUniform3f(u["uCamPos"], *cam_pos.tolist())
        gl.glUniform3f(u["uLightDir"], *light_dir)
        gl.glUniform1f(u["uLightIntensity"], self._settings.get("light_intensity", 1.2))
        gl.glUniform1f(u["uEnvIntensity"], self._settings.get("env_intensity", 1.0))
        gl.glUniform1f(u["uExposure"], self._settings.get("exposure", 1.0))
        gl.glUniform3f(u["uAmbientColor"], *self._ambient_color)
        gl.glUniform1i(u["uSkyAmbient"], 1 if self._hdri_image is None else 0)
        gl.glUniform1f(u["uRoughnessValue"], self._roughness_value)
        gl.glUniform1i(u["uHasAlbedo"], 1 if self._albedo_image is not None else 0)
        gl.glUniform1i(u["uHasNormal"], 1 if self._normal_image is not None else 0)
        gl.glUniform1i(u["uHasRoughnessTex"], 1 if self._roughness_image is not None else 0)
        gl.glUniform1i(u["uHasHeight"], 1 if self._height_image is not None else 0)
        gl.glUniform1i(u["uHasAO"], 1 if self._ao_image is not None else 0)
        gl.glUniform1i(u["uPOMEnabled"], 1 if self._settings.get("pom_enabled", True) else 0)
        gl.glUniform1f(u["uPOMHeightScale"], self._settings.get("pom_height_scale", 0.04))
        gl.glUniform1i(u["uPOMMaxLayers"], self._settings.get("pom_max_layers", 32))
        gl.glUniform1f(u["uUVScale"], float(self._settings.get("uv_tiling", 1.0)))

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._albedo_tex)
        gl.glUniform1i(u["uAlbedoTex"], 0)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._normal_tex)
        gl.glUniform1i(u["uNormalTex"], 1)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._roughness_tex)
        gl.glUniform1i(u["uRoughnessTex"], 2)
        gl.glActiveTexture(gl.GL_TEXTURE3)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._height_tex)
        gl.glUniform1i(u["uHeightTex"], 3)
        gl.glActiveTexture(gl.GL_TEXTURE5)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._ao_tex)
        gl.glUniform1i(u["uAOTex"], 5)

        mesh_vao, mesh_index_count = self._meshes.get(self._settings.get("mesh", "sphere"), self._meshes["sphere"])
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if self._settings.get("wireframe") else gl.GL_FILL)
        gl.glBindVertexArray(mesh_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, mesh_index_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        if self._settings.get("show_grid"):
            gu = self._grid_uniforms
            gl.glUseProgram(self._grid_program)
            self._set_mat4(gu["uView"], view)
            self._set_mat4(gu["uProj"], proj)
            gl.glUniform3f(gu["uColor"], 0.35, 0.35, 0.38)
            gl.glBindVertexArray(self._grid_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, self._grid_vertex_count)
            gl.glBindVertexArray(0)

    @staticmethod
    def _set_mat4(location: int, mat: np.ndarray) -> None:
        gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE, mat)

    def mousePressEvent(self, event) -> None:
        self._last_mouse_pos = event.position()
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._is_dragging = True

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse_pos is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        pos = event.position()
        dx = pos.x() - self._last_mouse_pos.x()
        dy = pos.y() - self._last_mouse_pos.y()
        self._last_mouse_pos = pos
        self._apply_drag(dx, dy, event.modifiers())

    def _apply_drag(self, dx: float, dy: float, modifiers) -> None:
        """Orbit (plain), move the light (Shift) or pan the pivot (Ctrl/Cmd)
        by a drag of (dx, dy) pixels — from the mouse or a trackpad swipe."""
        is_pan = bool(modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))

        if is_pan:
            distance = self._settings.get("distance", 3.5)
            az = math.radians(self._azimuth)
            el = math.radians(self._elevation)
            cam_forward = np.array([
                -math.cos(el) * math.sin(az),
                -math.sin(el),
                -math.cos(el) * math.cos(az),
            ], dtype=np.float32)
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            cam_right = np.cross(cam_forward, world_up)
            norm_r = np.linalg.norm(cam_right)
            if norm_r > 1e-6:
                cam_right /= norm_r
            cam_up = np.cross(cam_right, cam_forward)
            norm_u = np.linalg.norm(cam_up)
            if norm_u > 1e-6:
                cam_up /= norm_u

            pan_speed = max(0.001, distance * 0.002)
            dx_pan = -dx * pan_speed
            dy_pan = dy * pan_speed

            delta_pivot = cam_right * dx_pan + cam_up * dy_pan
            new_px = round(float(self._settings.get("pivot_x", 0.0) + delta_pivot[0]), 3)
            new_py = round(float(self._settings.get("pivot_y", 0.0) + delta_pivot[1]), 3)
            new_pz = round(float(self._settings.get("pivot_z", 0.0) + delta_pivot[2]), 3)

            self._settings["pivot_x"] = new_px
            self._settings["pivot_y"] = new_py
            self._settings["pivot_z"] = new_pz
            self.pivotChanged.emit(new_px, new_py, new_pz)
            self.update()
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            new_az = (self._settings.get("light_azimuth", 45.0) + dx * 0.5) % 360.0
            new_el = max(-90.0, min(90.0, self._settings.get("light_elevation", 50.0) - dy * 0.5))
            self._settings["light_azimuth"] = new_az
            self._settings["light_elevation"] = new_el
            self.lightChanged.emit(new_az, new_el)
            self.update()
        else:
            self._azimuth = (self._azimuth - dx * 0.4) % 360.0
            self._elevation = max(-85.0, min(85.0, self._elevation + dy * 0.4))
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._last_mouse_pos = None
        self._is_dragging = False

    def wheelEvent(self, event) -> None:
        if trackpad.is_trackpad_scroll(event):
            # Two-finger swipe orbits (Shift: light, Cmd: pan), like a drag.
            d = trackpad.scroll_delta(event)
            self._apply_drag(d.x(), d.y(), event.modifiers())
            return
        delta = event.angleDelta().y() / 120.0
        current_distance = self._settings.get("distance", 3.5)
        step = 0.05 if current_distance <= 1.0 else 0.2
        self._set_distance(current_distance - delta * step)

    def _set_distance(self, distance: float) -> None:
        new_distance = max(0.01, min(20.0, distance))
        self._settings["distance"] = new_distance
        self.distanceChanged.emit(new_distance)
        self.update()

    def event(self, event) -> bool:
        factor = trackpad.pinch_factor(event)
        if factor is not None:
            self._set_distance(self._settings.get("distance", 3.5) / factor)
            return True
        if trackpad.is_smart_zoom(event):
            self.reset_camera()
            return True
        return super().event(event)

    def reset_camera(self) -> None:
        self._azimuth = 35.0
        self._elevation = 20.0
        self._settings["distance"] = 3.5
        self._settings["pivot_x"] = 0.0
        self._settings["pivot_y"] = 0.0
        self._settings["pivot_z"] = 0.0
        self.distanceChanged.emit(3.5)
        self.pivotChanged.emit(0.0, 0.0, 0.0)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F:
            self.reset_camera()
        elif event.key() == Qt.Key.Key_Space:
            auto = not self._settings.get("auto_rotate", True)
            self._settings["auto_rotate"] = auto
            self.autoRotateToggled.emit(auto)
            self.update()
        else:
            super().keyPressEvent(event)


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m.T.copy()


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m.T.copy()
