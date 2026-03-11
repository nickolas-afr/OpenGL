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
uniform mat4 uMVP;
uniform mat4 uModel;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vUV   = aUV;
    vNorm = normalize(mat3(transpose(inverse(uModel))) * aNorm);
}
"""

FRAG_TERRAIN = """
#version 330 core
in  vec2 vUV;
in  vec3 vNorm;
out vec4 fColor;
uniform sampler2D uTex;
uniform vec3  uSunDir;
uniform vec3  uSunColor;
uniform float uAmbient;
void main() {
    float diff   = max(dot(normalize(vNorm), normalize(uSunDir)), 0.0);
    vec3  albedo = texture(uTex, vUV).rgb;
    fColor = vec4((uAmbient + diff * uSunColor) * albedo, 1.0);
}
"""

VERT_PYRAMID = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNorm;
out vec3 vNorm;
uniform mat4 uMVP;
uniform mat4 uModel;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vNorm = normalize(mat3(transpose(inverse(uModel))) * aNorm);
}
"""

FRAG_PYRAMID = """
#version 330 core
in  vec3 vNorm;
out vec4 fColor;
uniform vec3  uColor;
uniform vec3  uSunDir;
uniform vec3  uSunColor;
uniform float uAmbient;
void main() {
    float diff = max(dot(normalize(vNorm), normalize(uSunDir)), 0.0);
    fColor = vec4((uAmbient + diff * uSunColor) * uColor, 1.0);
}
"""
