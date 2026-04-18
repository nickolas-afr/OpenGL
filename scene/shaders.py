_SHADOW_PCF = """\
uniform sampler2D uShadowMap;
// n_dot_l: dot(normal, sunDir), clamped to [0,1] — drives slope-based bias.
// Low bias when surface faces the sun (shadow starts at caster base);
// higher bias on grazing surfaces to suppress self-shadowing acne.
float shadow_pcf(vec4 lp, float n_dot_l) {
    vec3 p = lp.xyz / lp.w * 0.5 + 0.5;
    if (p.z > 1.0) return 0.0;
    float bias = max(0.001 * (1.0 - n_dot_l), 0.00005);
    float s    = 0.0;
    vec2  sz   = 1.0 / vec2(textureSize(uShadowMap, 0));
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            s += float(p.z - bias > texture(uShadowMap, p.xy + vec2(x, y) * sz).r);
    return s / 9.0;
}
"""

# Shared spot-light helper for all 14 poles.
# Each lamp head points straight down; the cone is defined by dir.y (fraction of
# the light-to-fragment vector that is vertical). Attenuation uses a quadratic
# falloff so pools of light are clearly visible without being overwhelming.
_POLE_LIGHTS = """\
#define MAX_POLES 20
uniform vec3  uPoleLights[MAX_POLES];
uniform int   uNumPoles;
uniform vec3  uPoleColor;
uniform float uPoleIntensity;
vec3 pole_light(vec3 worldPos, vec3 normal) {
    vec3 acc = vec3(0.0);
    for (int i = 0; i < uNumPoles; i++) {
        vec3  toL  = uPoleLights[i] - worldPos;
        float dist = length(toL);
        if (dist < 0.001) continue;
        vec3  dir  = toL / dist;
        // dir.y > 0 means lamp is above fragment (normal case).
        // smoothstep fades the cone out sideways so only surfaces
        // below and near the lamp receive significant light.
        float spot = smoothstep(0.0, 0.55, dir.y);
        float att  = spot / (1.0 + 0.1 * dist + 0.02 * dist * dist);
        acc += max(dot(normal, dir), 0.0) * att;
    }
    return acc * uPoleIntensity * uPoleColor;
}
"""

VERT_SHADOW = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uLightMVP;
void main() {
    gl_Position = uLightMVP * vec4(aPos, 1.0);
}
"""

FRAG_SHADOW = """
#version 330 core
void main() {}
"""

VERT_SKY = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vUV = aUV;
}
"""

FRAG_SKY = """
#version 330 core
in  vec2 vUV;
out vec4 fColor;
uniform sampler2D uTex;
void main() {
    fColor = texture(uTex, vUV);
}
"""

VERT_TERRAIN = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec3 aNorm;
out vec2 vUV;
out vec3 vNorm;
out vec3 vWorldPos;
out vec4 vLightPos;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat4 uLightMat;
void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    gl_Position = uMVP * vec4(aPos, 1.0);
    vUV       = aUV;
    vNorm     = normalize(mat3(transpose(inverse(uModel))) * aNorm);
    vWorldPos = worldPos.xyz;
    vLightPos = uLightMat * vec4(aPos, 1.0);
}
"""

FRAG_TERRAIN = """
#version 330 core
in  vec2 vUV;
in  vec3 vNorm;
in  vec3 vWorldPos;
in  vec4 vLightPos;
out vec4 fColor;
uniform sampler2D uTex;
uniform vec3  uSunDir;
uniform vec3  uSunColor;
uniform float uAmbient;
""" + _SHADOW_PCF + _POLE_LIGHTS + """
void main() {
    vec3  n      = normalize(vNorm);
    float diff   = max(dot(n, normalize(uSunDir)), 0.0);
    vec3  albedo = texture(uTex, vUV).rgb;
    float shad   = shadow_pcf(vLightPos, diff);
    vec3  poles  = pole_light(vWorldPos, n);
    fColor = vec4((uAmbient + diff * (1.0 - shad) * uSunColor + poles) * albedo, 1.0);
}
"""

VERT_PYRAMID = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNorm;
out vec3 vNorm;
out vec3 vWorldPos;
out vec4 vLightPos;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat4 uLightMat;
void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    gl_Position = uMVP * vec4(aPos, 1.0);
    vNorm     = normalize(mat3(transpose(inverse(uModel))) * aNorm);
    vWorldPos = worldPos.xyz;
    vLightPos = uLightMat * vec4(aPos, 1.0);
}
"""

FRAG_PYRAMID = """
#version 330 core
in  vec3 vNorm;
in  vec3 vWorldPos;
in  vec4 vLightPos;
out vec4 fColor;
uniform vec3  uColor;
uniform vec3  uSunDir;
uniform vec3  uSunColor;
uniform float uAmbient;
""" + _SHADOW_PCF + _POLE_LIGHTS + """
void main() {
    vec3  n    = normalize(vNorm);
    float diff = max(dot(n, normalize(uSunDir)), 0.0);
    float shad = shadow_pcf(vLightPos, diff);
    vec3  poles = pole_light(vWorldPos, n);
    fColor = vec4((uAmbient + diff * (1.0 - shad) * uSunColor + poles) * uColor, 1.0);
}
"""

