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
TERRAIN_HEIGHT = 0.0    # max hill elevation; set to 0 for a flat ground plane
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
CIRCUIT_Y           = 0.10  # must exceed worst-case bilinear-vs-triangle mismatch

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

# Shadow mapping
SHADOW_MAP_SIZE = 2048

# Lighting poles — (x, z) placed outside the circuit perimeter (road edge ≈ ±7 from centreline)
POLE_POSITIONS = [
    # Right straight (centreline x≈65, road edge x≈72)
    ( 80, -90), ( 80, -50), ( 80, -10), ( 80,  30), ( 80,  65),
    # Top hairpin / curve
    ( 55,  115), (  0,  132),
    # Left straight (centreline x≈-68, road edge x≈-75)
    (-83,  55), (-83,  10), (-83, -35), (-83, -75),
    # Bottom curve
    (-45, -125), ( 10, -133), ( 45, -120),
]
POLE_HEIGHT = 7.0
POLE_COLOR  = np.array([0.25, 0.25, 0.28], dtype=np.float32)  # dark steel
LAMP_COLOR  = np.array([0.95, 0.90, 0.72], dtype=np.float32)  # warm white

# Emitted light from each lamp head (additive, no per-pole shadow map)
POLE_LIGHT_COLOR     = np.array([1.0, 0.88, 0.55], dtype=np.float32)  # sodium yellow
POLE_LIGHT_INTENSITY = 2.5

# ── Dynamic entities  ──────────────────────────────────────────────────────
# Car — spawns at the start of the right straight, facing +Z (toward circuit)
CAR_START_X   =  65.0
CAR_START_Z   = -80.0
CAR_START_YAW =   0.0
CAR_COLOR_BODY = np.array([0.85, 0.10, 0.10], dtype=np.float32)  # red body
CAR_COLOR_CAB  = np.array([0.60, 0.06, 0.06], dtype=np.float32)  # dark-red cab

# Birds — number to spawn; all use a single colour
NUM_BIRDS  = 6
BIRD_COLOR = np.array([0.15, 0.15, 0.20], dtype=np.float32)

# Pedestrians — one colour per pedestrian + a shared skin tone for heads
NUM_PEDESTRIANS = 5
PEDESTRIAN_COLORS = [
    np.array([0.70, 0.50, 0.30], dtype=np.float32),  # tan jacket
    np.array([0.20, 0.40, 0.65], dtype=np.float32),  # blue jacket
    np.array([0.55, 0.25, 0.25], dtype=np.float32),  # red jacket
    np.array([0.30, 0.55, 0.30], dtype=np.float32),  # green jacket
    np.array([0.50, 0.50, 0.52], dtype=np.float32),  # grey jacket
]
PED_HEAD_COLOR   = np.array([0.85, 0.70, 0.55], dtype=np.float32)
PED_PANTS_COLOR  = np.array([0.20, 0.22, 0.35], dtype=np.float32)
CAR_WHEEL_COLOR  = np.array([0.14, 0.14, 0.17], dtype=np.float32)
