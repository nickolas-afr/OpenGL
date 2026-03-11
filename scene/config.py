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
TERRAIN_HEIGHT = 15.0   # max hill elevation; set to 0 for a flat ground plane
TERRAIN_TILE   = 32.0   # texture tile repetitions across terrain surface

# Pyramid
PYRAMID_BASE_HALF = 20.0   # half-width of the square base
PYRAMID_HEIGHT    = 40.0   # height of the apex above y=0
PYRAMID_COLOR     = np.array([0.82, 0.70, 0.50], dtype=np.float32)  # sandy/tan

# Camera
CAM_START  = (0.0, 15.0, 80.0)
CAM_YAW    = -90.0    # initial yaw  (looks toward –Z)
CAM_PITCH  = -10.0    # initial pitch (slightly downward)
CAM_SPEED  = 20.0     # units / second
MOUSE_SENS = 0.12     # degrees / pixel
FOV_DEG    = 60.0
NEAR, FAR  = 0.1, 800.0

# Directional light (sun) – direction pointing FROM surface TOWARD sun
SUN_DIR   = np.array([0.45, 0.82, 0.35], dtype=np.float32)
SUN_COLOR = np.array([1.00, 0.95, 0.82], dtype=np.float32)
AMBIENT   = 0.32
