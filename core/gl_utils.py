import ctypes

import numpy as np
from OpenGL.GL import *


def compile_shader(src, kind):
    sh = glCreateShader(kind)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sh).decode())
    return sh


def link_program(vs_src, fs_src):
    vs   = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs   = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


def upload_texture(pil_img, repeat=True):
    """Upload a PIL image to an OpenGL texture (Y-flipped to match GL convention)."""
    rgba = np.array(pil_img.convert("RGBA"), dtype=np.uint8)
    data = np.ascontiguousarray(rgba[::-1])   # flip Y: Pillow top-row → GL bottom-row
    h, w = data.shape[:2]
    tid  = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tid)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glGenerateMipmap(GL_TEXTURE_2D)
    wrap = GL_REPEAT if repeat else GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tid


def create_mesh(vertices: np.ndarray, indices: np.ndarray, attribs):
    """
    Build and return (vao_id, index_count) from flat float32 / uint32 arrays.
    attribs: list of (location, n_components, stride_bytes, offset_bytes)
    """
    vao = int(glGenVertexArrays(1))
    vbo = int(glGenBuffers(1))
    ebo = int(glGenBuffers(1))

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    for loc, n, stride, offset in attribs:
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, n, GL_FLOAT, GL_FALSE, stride,
                              ctypes.c_void_p(offset))
    glBindVertexArray(0)
    return vao, len(indices)


def set_mat4(prog, name, mat):
    """Upload a row-major float32 numpy matrix (GL_TRUE = transpose to column-major)."""
    glUniformMatrix4fv(glGetUniformLocation(prog, name),
                       1, GL_TRUE, np.ascontiguousarray(mat, dtype=np.float32))
