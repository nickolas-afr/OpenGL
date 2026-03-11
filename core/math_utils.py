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
