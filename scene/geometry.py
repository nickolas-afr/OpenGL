import math

import numpy as np

from core.gl_utils import create_mesh

_QUAD_IDX    = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
_SKY_ATTRIBS = [(0, 3, 20, 0), (1, 2, 20, 12)]   # pos(3) + uv(2), stride=20 B


def _catmull_rom(pts, subdivisions=8):
    """
    Centripetal Catmull-Rom subdivision (α=0.5) using the Barry-Goldman algorithm.
    Unlike uniform parameterisation, centripetal CR is mathematically guaranteed
    to produce no cusps or self-intersections at sharp or unequally-spaced corners.
    Returns a (N*subdivisions, 2) float32 array passing through every waypoint.
    """
    N   = len(pts)
    out = []

    def _knot(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1]) ** 0.5   # α = 0.5

    def _lerp(ta, tb, t, a, b):
        if abs(tb - ta) < 1e-12:
            return (a + b) * 0.5
        return (tb - t) / (tb - ta) * a + (t - ta) / (tb - ta) * b

    for i in range(N):
        p0 = pts[(i - 1) % N].astype(np.float64)
        p1 = pts[i].astype(np.float64)
        p2 = pts[(i + 1) % N].astype(np.float64)
        p3 = pts[(i + 2) % N].astype(np.float64)

        t0 = 0.0
        t1 = t0 + _knot(p0, p1)
        t2 = t1 + _knot(p1, p2)
        t3 = t2 + _knot(p2, p3)

        if t2 - t1 < 1e-9:          # degenerate zero-length segment
            out.extend([p1] * subdivisions)
            continue

        for k in range(subdivisions):
            t  = t1 + (t2 - t1) * k / subdivisions
            # Barry-Goldman recursive evaluation
            A1 = _lerp(t0, t1, t, p0, p1)
            A2 = _lerp(t1, t2, t, p1, p2)
            A3 = _lerp(t2, t3, t, p2, p3)
            B1 = _lerp(t0, t2, t, A1, A2)
            B2 = _lerp(t1, t3, t, A2, A3)
            out.append(_lerp(t1, t2, t, B1, B2))

    return np.array(out, dtype=np.float32)


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


def make_height_grid(half, divs, h_max):
    """
    Compute the terrain height grid used by build_terrain.
    Returns a (divs+1, divs+1) float32 array H where H[row, col] is the
    elevation at world position (x=lin[col], z=lin[row]).
    Also returns the 1-D coordinate array lin = linspace(-half, half, divs+1).
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
        fade_start = 0.78
        dist  = np.maximum(np.abs(fx), np.abs(fz))
        fade  = np.clip(1.0 - (dist - fade_start) / (1.0 - fade_start), 0.0, 1.0)
        H    *= fade ** 2
    else:
        H = np.zeros_like(X)

    return H.astype(np.float32), lin


class TerrainSampler:
    """
    Bilinear interpolator for the terrain height grid.
    Call sampler(x, z) to get the terrain elevation at any world position.
    Points outside the grid are clamped to the grid edge.
    """
    def __init__(self, H, half):
        self.H    = H
        self.half = float(half)
        self.n    = H.shape[0]

    def __call__(self, x, z):
        n    = self.n
        half = self.half
        # Map world coords to fractional grid indices
        # H[row, col] → row = z axis, col = x axis
        col_f = (float(x) + half) / (2.0 * half) * (n - 1)
        row_f = (float(z) + half) / (2.0 * half) * (n - 1)
        col0  = int(max(0, min(n - 2, col_f)))
        row0  = int(max(0, min(n - 2, row_f)))
        tc    = col_f - col0
        tr    = row_f - row0
        H     = self.H
        return float(
            H[row0,   col0  ] * (1 - tr) * (1 - tc)
          + H[row0,   col0+1] * (1 - tr) * tc
          + H[row0+1, col0  ] * tr       * (1 - tc)
          + H[row0+1, col0+1] * tr       * tc
        )


def build_terrain(half, divs, h_max, tile):
    """
    Generate a terrain grid mesh.
    Vertex layout: pos(3) + uv(2) + normal(3) → stride = 32 B
    Returns (vao, index_count).
    """
    H, lin = make_height_grid(half, divs, h_max)
    n      = len(lin)
    X, Z   = np.meshgrid(lin, lin)
    fx     = X / half
    fz     = Z / half

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

    I, J  = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
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


def build_circuit(waypoints, road_half_w, y_offset=0.05, v_tile=0.04,
                  subdivisions=8, height_sampler=None):
    """
    Closed road strip from (x, z) centre-line waypoints.
    Each segment is subdivided with a Catmull-Rom spline for smooth corners.
    If height_sampler is provided (a TerrainSampler), each vertex y is set to
    sampler(x, z) + y_offset so the road conforms to the terrain.
    Vertex layout: pos(3)+uv(2)+normal(3), stride=32 B.
    Returns (vao, index_count).
    """
    pts = np.array(waypoints, dtype=np.float32)   # (N, 2)
    pts = _catmull_rom(pts, subdivisions)          # smooth subdivision
    N   = len(pts)

    # Smooth tangent at each vertex
    tangents = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        d_in  = (pts[i] - pts[(i - 1) % N]).astype(np.float64)
        d_out = (pts[(i + 1) % N] - pts[i]).astype(np.float64)
        for d in (d_in, d_out):
            ln = math.hypot(d[0], d[1])
            if ln > 1e-6:
                d /= ln
        t  = d_in + d_out
        ln = math.hypot(t[0], t[1])
        tangents[i] = (t / ln if ln > 1e-6 else d_out).astype(np.float32)

    # Perpendicular (right side) = clockwise rotation: (tz, -tx)
    perps = np.stack([ tangents[:, 1], -tangents[:, 0]], axis=1)
    left  = pts - perps * road_half_w
    right = pts + perps * road_half_w

    verts, idxs, v_coord, vc = [], [], 0.0, 0

    _SKIRT_DEPTH = 1.2   # how far below the sampled terrain the skirt extends

    def _ht(xy):
        return height_sampler(float(xy[0]), float(xy[1])) if height_sampler else 0.0

    for i in range(N):
        j       = (i + 1) % N
        seg_len = math.hypot(*(pts[j] - pts[i]).tolist())
        v_next  = v_coord + seg_len * v_tile

        h_li = _ht(left[i]);  h_lj = _ht(left[j])
        h_ri = _ht(right[i]); h_rj = _ht(right[j])

        # ── TOP SURFACE ─────────────────────────────────────────────────────
        # vertex order: left_i, left_j, right_j, right_i  → normal UP (0,1,0)
        for (x, z), h, u, v in [
            (left[i],  h_li, 0.0, v_coord),
            (left[j],  h_lj, 0.0, v_next),
            (right[j], h_rj, 1.0, v_next),
            (right[i], h_ri, 1.0, v_coord),
        ]:
            verts.extend([x, h + y_offset, z, u, v, 0.0, 1.0, 0.0])
        idxs.extend([vc, vc+1, vc+2,  vc, vc+2, vc+3]);  vc += 4

        # Outward normals for skirts (horizontal, perpendicular to road direction)
        px, pz = float(perps[i][0]), float(perps[i][1])
        plen   = math.hypot(px, pz)
        if plen > 1e-6: px, pz = px / plen, pz / plen

        # ── LEFT SKIRT ──────────────────────────────────────────────────────
        # Normal points LEFT (−perp).  CCW from outside: top_i, top_j, bot_j, bot_i
        nx_l, nz_l = -px, -pz
        for (x, z), y_val, u, v in [
            (left[i], h_li + y_offset,    0.0, v_coord),   # top_i
            (left[j], h_lj + y_offset,    0.0, v_next),    # top_j
            (left[j], h_lj - _SKIRT_DEPTH, 1.0, v_next),   # bot_j
            (left[i], h_li - _SKIRT_DEPTH, 1.0, v_coord),  # bot_i
        ]:
            verts.extend([x, y_val, z, u, v, nx_l, 0.0, nz_l])
        idxs.extend([vc, vc+1, vc+2,  vc, vc+2, vc+3]);  vc += 4

        # ── RIGHT SKIRT ─────────────────────────────────────────────────────
        # Normal points RIGHT (+perp).  CCW from outside: top_i, bot_i, bot_j, top_j
        nx_r, nz_r = px, pz
        for (x, z), y_val, u, v in [
            (right[i], h_ri + y_offset,    0.0, v_coord),   # top_i
            (right[i], h_ri - _SKIRT_DEPTH, 1.0, v_coord),  # bot_i
            (right[j], h_rj - _SKIRT_DEPTH, 1.0, v_next),   # bot_j
            (right[j], h_rj + y_offset,    0.0, v_next),    # top_j
        ]:
            verts.extend([x, y_val, z, u, v, nx_r, 0.0, nz_r])
        idxs.extend([vc, vc+1, vc+2,  vc, vc+2, vc+3]);  vc += 4

        v_coord = v_next

    verts = np.array(verts, dtype=np.float32)
    idxs  = np.array(idxs,  dtype=np.uint32)
    return create_mesh(verts, idxs, [(0, 3, 32, 0), (1, 2, 32, 12), (2, 3, 32, 20)])


def build_box(w, h, d):
    """
    Axis-aligned textured box: width w (X), height h (Y), depth d (Z).
    Base at y=0, centred in X and Z.
    Vertex layout: pos(3)+uv(2)+normal(3), stride=32 B.
    Returns (vao, index_count).
    """
    hw, hd = w * 0.5, d * 0.5
    faces = [
        ((0, 1, 0), (-hw,h,+hd), (+hw,h,+hd), (+hw,h,-hd), (-hw,h,-hd)),  # top    +Y
        ((0,-1, 0), (-hw,0,-hd), (+hw,0,-hd), (+hw,0,+hd), (-hw,0,+hd)),  # bottom -Y
        ((1, 0, 0), (+hw,0,+hd), (+hw,0,-hd), (+hw,h,-hd), (+hw,h,+hd)),  # right  +X
        ((-1,0, 0), (-hw,0,-hd), (-hw,0,+hd), (-hw,h,+hd), (-hw,h,-hd)),  # left   -X
        ((0, 0, 1), (-hw,0,+hd), (+hw,0,+hd), (+hw,h,+hd), (-hw,h,+hd)),  # front  +Z
        ((0, 0,-1), (+hw,0,-hd), (-hw,0,-hd), (-hw,h,-hd), (+hw,h,-hd)),  # back   -Z
    ]
    uv_quad = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    quad    = [0, 1, 2,  0, 2, 3]
    verts, idxs, base = [], [], 0
    for (nx, ny, nz), *corners in faces:
        for (px, py, pz), (u, v) in zip(corners, uv_quad):
            verts.extend([px, py, pz, u, v, nx, ny, nz])
        idxs.extend([base + q for q in quad])
        base += 4
    verts = np.array(verts, dtype=np.float32)
    idxs  = np.array(idxs,  dtype=np.uint32)
    return create_mesh(verts, idxs, [(0, 3, 32, 0), (1, 2, 32, 12), (2, 3, 32, 20)])


def build_box_solid(w, h, d):
    """
    Same as build_box but without UVs: pos(3)+normal(3), stride=24 B.
    For solid-colour rendering with pyramid_prog.
    Returns (vao, index_count).
    """
    hw, hd = w * 0.5, d * 0.5
    faces = [
        ((0, 1, 0), (-hw,h,+hd), (+hw,h,+hd), (+hw,h,-hd), (-hw,h,-hd)),
        ((0,-1, 0), (-hw,0,-hd), (+hw,0,-hd), (+hw,0,+hd), (-hw,0,+hd)),
        ((1, 0, 0), (+hw,0,+hd), (+hw,0,-hd), (+hw,h,-hd), (+hw,h,+hd)),
        ((-1,0, 0), (-hw,0,-hd), (-hw,0,+hd), (-hw,h,+hd), (-hw,h,-hd)),
        ((0, 0, 1), (-hw,0,+hd), (+hw,0,+hd), (+hw,h,+hd), (-hw,h,+hd)),
        ((0, 0,-1), (+hw,0,-hd), (-hw,0,-hd), (-hw,h,-hd), (+hw,h,-hd)),
    ]
    quad = [0, 1, 2,  0, 2, 3]
    verts, idxs, base = [], [], 0
    for (nx, ny, nz), *corners in faces:
        for (px, py, pz) in corners:
            verts.extend([px, py, pz, nx, ny, nz])
        idxs.extend([base + q for q in quad])
        base += 4
    verts = np.array(verts, dtype=np.float32)
    idxs  = np.array(idxs,  dtype=np.uint32)
    return create_mesh(verts, idxs, [(0, 3, 24, 0), (1, 3, 24, 12)])


def build_cone(radius, height, segments=12):
    """
    Flat-shaded cone: base at y=0, apex at y=height.
    Vertex layout: pos(3)+normal(3), stride=24 B.
    Winding: apex → p1 → p0 gives outward-facing normals.
    Returns (vao, index_count).
    """
    apex  = np.array([0.0, float(height), 0.0])
    verts = []
    for i in range(segments):
        a0 = 2.0 * math.pi * i       / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        p0 = np.array([radius * math.cos(a0), 0.0, radius * math.sin(a0)])
        p1 = np.array([radius * math.cos(a1), 0.0, radius * math.sin(a1)])
        raw = np.cross(p1 - apex, p0 - apex)
        ln  = np.linalg.norm(raw)
        n   = (raw / ln).astype(np.float32) if ln > 1e-9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for pt in (apex, p1, p0):
            verts.extend([*(pt.astype(np.float32)), *n])
    verts = np.array(verts, dtype=np.float32)
    idxs  = np.arange(segments * 3, dtype=np.uint32)
    return create_mesh(verts, idxs, [(0, 3, 24, 0), (1, 3, 24, 12)])
