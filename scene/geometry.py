import math

import numpy as np

from core.gl_utils import create_mesh

_QUAD_IDX    = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
_SKY_ATTRIBS = [(0, 3, 20, 0), (1, 2, 20, 12)]   # pos(3) + uv(2), stride=20 B


def _sky_face(p0, p1, p2, p3, u0=0., u1=1., v0=0., v1=1.):
    """Quad vertices: BL p0, BR p1, TR p2, TL p3 (rendered with culling OFF)."""
    return np.array([
        *p0, u0, v0,
        *p1, u1, v0,
        *p2, u1, v1,
        *p3, u0, v1,
    ], dtype=np.float32)


def build_skybox(W, H, D, floor_tile=25.0):
    """
    Returns [(vao, 6), …] for each of the 6 faces of the world cube.
    Face order: [0]=bottom(grass), [1]=top(sky),
                [2]=front(+Z),     [3]=back(−Z),
                [4]=right(+X),     [5]=left(−X)   ← sides get horizon texture
    The cube spans x∈[±W], y∈[0,H], z∈[±D].
    """
    ft = floor_tile
    faces = [
        # Bottom y=0 – tiled grass (UV spans 0→ft on both axes)
        _sky_face((-W, 0, -D), ( W, 0, -D), ( W, 0,  D), (-W, 0,  D),
                  0, ft, 0, ft),
        # Top y=H – sky (single image)
        _sky_face((-W, H,  D), ( W, H,  D), ( W, H, -D), (-W, H, -D)),
        # Front  z=+D
        _sky_face((-W, 0,  D), ( W, 0,  D), ( W, H,  D), (-W, H,  D)),
        # Back   z=−D
        _sky_face(( W, 0, -D), (-W, 0, -D), (-W, H, -D), ( W, H, -D)),
        # Right  x=+W
        _sky_face(( W, 0,  D), ( W, 0, -D), ( W, H, -D), ( W, H,  D)),
        # Left   x=−W
        _sky_face((-W, 0, -D), (-W, 0,  D), (-W, H,  D), (-W, H, -D)),
    ]
    return [create_mesh(f, _QUAD_IDX, _SKY_ATTRIBS) for f in faces]


def build_terrain(half, divs, h_max, tile):
    """
    Generate a terrain grid mesh.
    Vertex layout: pos(3) + uv(2) + normal(3) → stride = 32 B
    Returns (vao, index_count).
    """
    n    = divs + 1
    lin  = np.linspace(-half, half, n, dtype=np.float32)
    X, Z = np.meshgrid(lin, lin)
    fx   = X / half
    fz   = Z / half

    if h_max > 0:
        H = (h_max * (
             0.30 * np.sin(fx * math.pi * 3.0)        * np.cos(fz * math.pi * 2.5)
           + 0.22 * np.sin(fx * math.pi * 6.5 + 1.5)  * np.sin(fz * math.pi * 5.5)
           + 0.18 * np.cos(fx * math.pi * 11.0)        * np.cos(fz * math.pi * 9.0 + 0.7)
           + 0.12 * np.sin(fx * math.pi * 19.0 + 0.4)  * np.sin(fz * math.pi * 17.0)
           + 0.08 * np.cos(fx * math.pi * 30.0)        * np.cos(fz * math.pi * 28.0 + 1.2)
           + 0.05 * np.sin(fx * math.pi * 48.0 + 0.8)  * np.sin(fz * math.pi * 45.0)
        ))
        H = (H - H.min()) / (H.max() - H.min() + 1e-9) * h_max

        # Smooth fade-to-flat near edges (avoids gaps with the skybox floor)
        fade_start = 0.78
        dist  = np.maximum(np.abs(fx), np.abs(fz))
        fade  = np.clip(1.0 - (dist - fade_start) / (1.0 - fade_start), 0.0, 1.0)
        H    *= fade ** 2
    else:
        H = np.zeros_like(X)

    # Surface normals via central differences (flat terrain → all (0,1,0))
    cell = 2.0 * half / (n - 1)
    Hp   = np.pad(H, 1, mode="edge")
    dHdx = (Hp[1:-1, 2:] - Hp[1:-1, :-2]) / (2.0 * cell)
    dHdz = (Hp[2:,  1:-1] - Hp[:-2, 1:-1]) / (2.0 * cell)
    Nx, Ny, Nz = -dHdx, np.ones_like(dHdx), -dHdz
    L = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    Nx /= L;  Ny /= L;  Nz /= L

    U = (fx + 1.0) * 0.5 * tile
    V = (fz + 1.0) * 0.5 * tile

    verts = np.stack([X, H, Z, U, V, Nx, Ny, Nz], axis=-1).astype(np.float32).ravel()

    I, J  = np.meshgrid(np.arange(divs), np.arange(divs), indexing="ij")
    I, J  = I.ravel(), J.ravel()
    tl = I * n + J
    tri1 = np.stack([tl,     tl + n,     tl + 1    ], axis=1)
    tri2 = np.stack([tl + 1, tl + n,     tl + n + 1], axis=1)
    idxs = np.concatenate([tri1, tri2], axis=1).ravel().astype(np.uint32)

    # stride=32 B: pos@0 B, uv@12 B, normal@20 B
    attribs = [(0, 3, 32, 0), (1, 2, 32, 12), (2, 3, 32, 20)]
    return create_mesh(verts, idxs, attribs)


def build_pyramid(base_half, height):
    """
    Build a square pyramid centered at the origin, base on y=0, apex at y=height.
    Vertex layout: pos(3) + normal(3) → stride = 24 B
    Returns (vao, index_count).
    Each of the 4 triangular faces has its own flat normal (no shared vertices).
    """
    s = base_half
    h = height

    apex = np.array([0.0, h, 0.0])
    corners = [
        np.array([-s, 0.0, -s]),  # back-left
        np.array([ s, 0.0, -s]),  # back-right
        np.array([ s, 0.0,  s]),  # front-right
        np.array([-s, 0.0,  s]),  # front-left
    ]

    def face_normal(a, b, c):
        n = np.cross(b - a, c - a).astype(np.float32)
        return n / np.linalg.norm(n)

    # 4 side faces, each as a triangle: left-base, right-base, apex
    side_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    verts = []
    for li, ri in side_pairs:
        a, b, c = corners[ri], corners[li], apex
        n = face_normal(a, b, c)
        for pt in (a, b, c):
            verts.extend([*pt, *n])

    verts = np.array(verts, dtype=np.float32)
    idxs  = np.arange(12, dtype=np.uint32)   # 4 faces × 3 vertices, no sharing

    # stride=24 B: pos@0 B, normal@12 B
    attribs = [(0, 3, 24, 0), (1, 3, 24, 12)]
    return create_mesh(verts, idxs, attribs)
    """
    Generate a terrain grid mesh.
    Vertex layout: pos(3) + uv(2) + normal(3)  → stride = 32 B
    Returns (vao, index_count).
    """
    import math
    n    = divs + 1
    lin  = np.linspace(-half, half, n, dtype=np.float32)
    X, Z = np.meshgrid(lin, lin)
    fx   = X / half
    fz   = Z / half

    H = (h_max * (
         0.30 * np.sin(fx * math.pi * 3.0)        * np.cos(fz * math.pi * 2.5)
       + 0.22 * np.sin(fx * math.pi * 6.5 + 1.5)  * np.sin(fz * math.pi * 5.5)
       + 0.18 * np.cos(fx * math.pi * 11.0)        * np.cos(fz * math.pi * 9.0 + 0.7)
       + 0.12 * np.sin(fx * math.pi * 19.0 + 0.4)  * np.sin(fz * math.pi * 17.0)
       + 0.08 * np.cos(fx * math.pi * 30.0)        * np.cos(fz * math.pi * 28.0 + 1.2)
       + 0.05 * np.sin(fx * math.pi * 48.0 + 0.8)  * np.sin(fz * math.pi * 45.0)
    ))
    H = (H - H.min()) / (H.max() - H.min() + 1e-9) * h_max

    # Smooth fade-to-flat near edges (avoids gaps with the skybox floor)
    fade_start = 0.78
    dist  = np.maximum(np.abs(fx), np.abs(fz))
    fade  = np.clip(1.0 - (dist - fade_start) / (1.0 - fade_start), 0.0, 1.0)
    fade  = fade ** 2
    H    *= fade

    # Surface normals (central differences)
    cell = 2.0 * half / (n - 1)
    Hp   = np.pad(H, 1, mode="edge")
    dHdx = (Hp[1:-1, 2:] - Hp[1:-1, :-2]) / (2.0 * cell)
    dHdz = (Hp[2:,  1:-1] - Hp[:-2, 1:-1]) / (2.0 * cell)
    Nx, Ny, Nz = -dHdx, np.ones_like(dHdx), -dHdz
    L         = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    Nx /= L;  Ny /= L;  Nz /= L

    U = (fx + 1.0) * 0.5 * tile
    V = (fz + 1.0) * 0.5 * tile

    verts = np.stack([X, H, Z, U, V, Nx, Ny, Nz], axis=-1).astype(np.float32).ravel()

    I, J  = np.meshgrid(np.arange(divs), np.arange(divs), indexing="ij")
    I, J  = I.ravel(), J.ravel()
    tl = I * n + J
    tri1 = np.stack([tl,     tl + n,     tl + 1    ], axis=1)
    tri2 = np.stack([tl + 1, tl + n,     tl + n + 1], axis=1)
    idxs = np.concatenate([tri1, tri2], axis=1).ravel().astype(np.uint32)

    # stride=32 B: pos@0 B, uv@12 B, normal@20 B
    attribs = [(0, 3, 32, 0), (1, 2, 32, 12), (2, 3, 32, 20)]
    return create_mesh(verts, idxs, attribs)
