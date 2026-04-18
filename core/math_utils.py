import math

import numpy as np


def mat_perspective(fov_deg, aspect, near, far):
    f  = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    nf = 1.0 / (near - far)
    return np.array([
        [f / aspect, 0,  0,                      0                    ],
        [0,          f,  0,                      0                    ],
        [0,          0,  (far + near) * nf,      2.0 * far * near * nf],
        [0,          0, -1,                      0                    ],
    ], dtype=np.float32)


def mat_translate(x, y, z):
    """Row-major 4×4 translation matrix."""
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x;  m[1, 3] = y;  m[2, 3] = z
    return m


def mat_ortho(left, right, bottom, top, near, far):
    """Row-major 4×4 orthographic projection matrix (same convention as mat_perspective)."""
    rl = right - left
    tb = top - bottom
    fn = far - near
    return np.array([
        [2.0/rl, 0,       0,        -(right+left)/rl],
        [0,      2.0/tb,  0,        -(top+bottom)/tb],
        [0,      0,      -2.0/fn,   -(far+near)/fn  ],
        [0,      0,       0,         1.0             ],
    ], dtype=np.float32)


def mat_scale(sx, sy, sz):
    """Row-major 4×4 scale matrix."""
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx;  m[1, 1] = sy;  m[2, 2] = sz
    return m


def mat_look_at(eye, target):
    e  = np.asarray(eye,    dtype=np.float64)
    t  = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0])
    f  = t - e;  f /= np.linalg.norm(f)
    r  = np.cross(f, up)
    rn = np.linalg.norm(r)
    r  = r / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
    u  = np.cross(r, f)
    return np.array([
        [ r[0],  r[1],  r[2], -np.dot(r, e)],
        [ u[0],  u[1],  u[2], -np.dot(u, e)],
        [-f[0], -f[1], -f[2],  np.dot(f, e)],
        [ 0,     0,     0,     1            ],
    ], dtype=np.float32)
