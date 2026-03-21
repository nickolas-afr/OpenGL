import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _noise2d(size, frequencies, amplitudes, phase_seed=0):
    """Sum of sine-based 2-D noise octaves; returns (size,size) float array in [−1,1]."""
    xs = np.linspace(0, 64, size, dtype=np.float32)
    X, Y = np.meshgrid(xs, xs)
    n = np.zeros((size, size), dtype=np.float32)
    rng = np.random.default_rng(phase_seed)
    for freq, amp in zip(frequencies, amplitudes):
        px, py = rng.uniform(0, 6.28, 2)
        n += amp * np.sin(X * freq + px) * np.cos(Y * freq + py)
    mx = sum(amplitudes)
    return np.clip(n / mx, -1.0, 1.0)


def make_grass_tex(size=512):
    """Tiling grass texture with sine-noise colour variation."""
    n = _noise2d(size,
                 [0.40, 0.93, 1.80, 3.50, 7.00, 14.0],
                 [0.30, 0.25, 0.20, 0.15, 0.07, 0.03],
                 phase_seed=42)
    t = (n + 1.0) * 0.5
    r = np.clip(28  + t * 40, 0, 255).astype(np.uint8)
    g = np.clip(88  + t * 75, 0, 255).astype(np.uint8)
    b = np.clip(18  + t * 22, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")
    return img.filter(ImageFilter.GaussianBlur(0.8))


def make_sky_top_tex(size=512):
    """Overhead sky: deep blue at centre, lighter near edges."""
    import math
    xs = np.linspace(-1, 1, size, dtype=np.float32)
    X, Y = np.meshgrid(xs, xs)
    d = np.sqrt(X ** 2 + Y ** 2) / math.sqrt(2.0)
    r = np.clip( 75 + d * 85,  0, 255).astype(np.uint8)
    g = np.clip(125 + d * 75,  0, 255).astype(np.uint8)
    b = np.clip(205 + d * 40,  0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def make_horizon_tex(size=512):
    """Side-wall texture: sky gradient → mountain silhouette → ground."""
    img  = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    sky_row = int(size * 0.72)

    for py in range(sky_row):
        t = py / sky_row
        draw.line([(0, py), (size - 1, py)],
                  fill=(int(78  + t * 92),
                        int(128 + t * 82),
                        int(198 + t * 32)))

    for py in range(sky_row, size):
        t = (py - sky_row) / (size - sky_row)
        draw.line([(0, py), (size - 1, py)],
                  fill=(int(72 - t * 22),
                        int(102 - t * 12),
                        int(36  - t * 8)))

    rng = np.random.default_rng(seed=77)
    pts = [(0, sky_row + 14)]
    x = 0
    while x < size:
        x += int(rng.integers(10, 52))
        y  = sky_row - int(rng.integers(4, 72))
        pts.append((min(x, size), max(y, 4)))
    pts += [(size, sky_row + 14), (size, sky_row + 42), (0, sky_row + 42)]
    draw.polygon(pts, fill=(58, 66, 80))

    inner = pts[1:-2]
    for i in range(len(inner) - 1):
        if inner[i][1] < sky_row - 38:
            draw.line([(inner[i][0], inner[i][1] - 5),
                       (inner[i + 1][0], inner[i + 1][1] - 5)],
                      fill=(208, 214, 220), width=3)

    return img.filter(ImageFilter.GaussianBlur(2.4))


def make_terrain_tex(size=512):
    """Detailed tiling grass/dirt texture for the terrain mesh."""
    n = _noise2d(size,
                 [0.31, 0.79, 1.70, 3.20, 6.50, 12.0, 22.0],
                 [0.28, 0.22, 0.18, 0.12, 0.10, 0.06, 0.04],
                 phase_seed=99)
    t = (n + 1.0) * 0.5
    r = np.clip(32  + t * 50, 0, 255).astype(np.uint8)
    g = np.clip(82  + t * 80, 0, 255).astype(np.uint8)
    b = np.clip(18  + t * 25, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")
    return img.filter(ImageFilter.GaussianBlur(0.6))


def make_road_tex(size=512):
    """Dark asphalt with sine-noise grain and a dashed centre-line."""
    n = _noise2d(size, [2.0, 5.0, 12.0], [0.55, 0.30, 0.15], phase_seed=17)
    t = (n + 1.0) * 0.5
    r = np.clip(38 + t * 18, 0, 255).astype(np.uint8)
    g = np.clip(38 + t * 18, 0, 255).astype(np.uint8)
    b = np.clip(44 + t * 18, 0, 255).astype(np.uint8)
    img  = Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")
    draw = ImageDraw.Draw(img)
    # Dashed centre line
    cx = size // 2
    dash, gap = size // 8, size // 16
    y = 0
    while y < size:
        draw.rectangle([(cx - 4, y), (cx + 4, y + dash)], fill=(210, 210, 210))
        y += dash + gap
    # Edge lines
    for ex in [size // 9, size - size // 9]:
        draw.rectangle([(ex - 3, 0), (ex + 3, size - 1)], fill=(220, 215, 160))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def make_building_tex(size=512):
    """Concrete facade with a grid of windows."""
    n = _noise2d(size, [3.0, 8.0], [0.70, 0.30], phase_seed=55)
    t = (n + 1.0) * 0.5
    r = np.clip(155 + t * 35, 0, 255).astype(np.uint8)
    g = np.clip(153 + t * 32, 0, 255).astype(np.uint8)
    b = np.clip(148 + t * 28, 0, 255).astype(np.uint8)
    img  = Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")
    draw = ImageDraw.Draw(img)
    cols, rows = 5, 4
    mx, my = size // 10, size // 8
    cw = (size - 2 * mx) // cols
    ch = (size - 2 * my) // rows
    pad = max(cw // 5, 4)
    for row in range(rows):
        for col in range(cols):
            x0 = mx + col * cw + pad
            y0 = my + row * ch + pad
            draw.rectangle([(x0, y0), (x0 + cw - 2*pad, y0 + ch - 2*pad)],
                           fill=(75, 98, 128))
    return img.filter(ImageFilter.GaussianBlur(0.5))
