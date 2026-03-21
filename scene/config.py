import numpy as np

# Window
WIN_W, WIN_H = 1280, 720
TITLE        = "Phase 1 – Scene & Environment"

# World-container (skybox) cube
SKY_W = 300.0   # x half-extent  (±300)
SKY_H = 200.0   # total height   (y: 0 → 200)
SKY_D = 300.0   # z half-extent  (±300)

# Terrain
TERRAIN_HALF   = 240.0  # half-extent of the terrain grid (< SKY_W / SKY_D)
TERRAIN_DIVS   = 100    # grid subdivisions (100×100 quads)
TERRAIN_HEIGHT = 2.0    # max hill elevation; set to 0 for a flat ground plane
TERRAIN_TILE   = 32.0   # texture tile repetitions across terrain surface

# Pyramid
PYRAMID_BASE_HALF = 20.0   # half-width of the square base
PYRAMID_HEIGHT    = 40.0   # height of the apex above y=0
PYRAMID_COLOR     = np.array([0.82, 0.70, 0.50], dtype=np.float32)  # sandy/tan

# Camera — elevated rear view so the full circuit is visible on startup
CAM_START  = (0.0, 50.0, 200.0)
CAM_YAW    = -90.0    # initial yaw  (looks toward –Z)
CAM_PITCH  = -20.0    # initial pitch (looking down at circuit)
CAM_SPEED  = 20.0     # units / second
MOUSE_SENS = 0.12     # degrees / pixel
FOV_DEG    = 60.0
NEAR, FAR  = 0.1, 800.0

# Directional light (sun) – direction pointing FROM surface TOWARD sun
SUN_DIR   = np.array([0.45, 0.82, 0.35], dtype=np.float32)
SUN_COLOR = np.array([1.00, 0.95, 0.82], dtype=np.float32)
AMBIENT   = 0.32

# Circuit  ─────────────────────────────────────────────────────────────────
CIRCUIT_ROAD_HALF_W = 7.0    # metres either side of centreline
CIRCUIT_Y           = 0.80  # must exceed worst-case bilinear-vs-triangle mismatch

# 21 centreline waypoints (x, z); closure is handled in build_circuit()
CIRCUIT_WAYPOINTS = [
    ( 65, -100), ( 65,  -60), ( 65,  -20), ( 65,   20), ( 65,   60),
    ( 65,   85), ( 45,  108), ( 15,  118), (  0,  118),
    (-20,  112), (-55,   95), (-68,   70),
    (-68,   40), (-68,    0), (-68,  -40), (-68,  -70),
    (-60, -100), (-30, -115), (  0, -120),
    ( 30, -115), ( 55, -105),
]

# Static objects  ──────────────────────────────────────────────────────────
# 10 trees: (x, z) — placed outside the circuit perimeter
TREE_POSITIONS = [
    ( 90, -80), ( 95, -20), ( 92,  40),
    ( 30, 135), (-30, 135),
    (-92,  50), (-95, -10), (-92, -60),
    (-10, -135), ( 40, -130),
]
TRUNK_H      = 3.5
CANOPY_R     = 4.5
CANOPY_H     = 7.0
TRUNK_COLOR  = np.array([0.38, 0.22, 0.09], dtype=np.float32)
CANOPY_COLOR = np.array([0.13, 0.50, 0.10], dtype=np.float32)

# 4 buildings: (cx, cz, width, height, depth) — box centred at (cx, 0, cz)
BUILDINGS = [
    ( 92, -100, 36,  7, 13),   # pit building at start/finish
    ( 92,  -55,  8, 24,  8),   # control tower
    (106,   10, 46, 10,  8),   # grandstand A (right side, clear of road edge at x=72)
    (-106,  10, 46, 10,  8),   # grandstand B (left side, clear of road edge at x=-75)
]
BUILDING_COLOR = np.array([0.75, 0.73, 0.70], dtype=np.float32)
