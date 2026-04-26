import math

import numpy as np
import glfw

from core.math_utils import mat_translate, mat_rotate_y, mat_rotate_x


class Car:
    """Player-controlled car on the terrain surface.

    Controls:  UP / DOWN = throttle / brake+reverse
               LEFT / RIGHT = steer
               C (handled in renderer) = toggle follow camera
    """

    # Mesh dimensions passed to build_box_solid(w, h, d).
    # The car faces local +Z; w = side-to-side (X), d = front-to-back (Z).
    BODY_W, BODY_H, BODY_D = 2.0, 0.8, 4.0
    CAB_W,  CAB_H,  CAB_D  = 1.8, 0.7, 2.5

    # Wheel geometry (build_wheel_solid)
    WHEEL_RADIUS = 0.38
    WHEEL_T      = 0.30   # axle thickness
    WHEEL_AX_X   = 1.15   # ±X offset from car centre (just outside body half-width)
    WHEEL_AX_Z   = 1.30   # ±Z offset (front/rear of body)

    COLLISION_RADIUS = 2.4   # conservative circle radius enclosing the rotated body

    ACCEL        = 18.0  # units / s²
    FRICTION     = 10.0  # units / s²  (coasting deceleration)
    BRAKE_DECEL  = 28.0  # units / s²  (braking / reverse deceleration)
    MAX_SPEED    = 28.0  # units / s   forward
    MAX_REVERSE  =  9.0  # units / s   reverse
    TURN_RATE    = 70.0  # degrees / s at MAX_SPEED, scales with |speed|

    def __init__(self, x, z, yaw_deg, terrain_sampler):
        self.x     = float(x)
        self.z     = float(z)
        self.yaw   = float(yaw_deg)  # degrees; 0 = facing +Z
        self.speed = 0.0
        self._sampler = terrain_sampler

    @property
    def y(self):
        return self._sampler(self.x, self.z)

    @property
    def yaw_rad(self):
        return math.radians(self.yaw)

    def update(self, window, dt):
        fwd   = glfw.get_key(window, glfw.KEY_UP)    == glfw.PRESS
        back  = glfw.get_key(window, glfw.KEY_DOWN)  == glfw.PRESS
        left  = glfw.get_key(window, glfw.KEY_LEFT)  == glfw.PRESS
        right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS

        # Speed
        if fwd and not back:
            self.speed = min(self.speed + self.ACCEL * dt, self.MAX_SPEED)
        elif back and not fwd:
            if self.speed > 0.0:
                self.speed = max(0.0, self.speed - self.BRAKE_DECEL * dt)
            else:
                self.speed = max(self.speed - self.ACCEL * dt, -self.MAX_REVERSE)
        else:
            if abs(self.speed) < 0.3:
                self.speed = 0.0
            elif self.speed > 0.0:
                self.speed = max(0.0, self.speed - self.FRICTION * dt)
            else:
                self.speed = min(0.0, self.speed + self.FRICTION * dt)

        # Steering (turn rate proportional to |speed| so it feels natural)
        if abs(self.speed) > 0.5:
            sign = 1 if self.speed > 0 else -1
            turn = self.TURN_RATE * abs(self.speed) / self.MAX_SPEED * dt
            if left:  self.yaw += turn * sign
            if right: self.yaw -= turn * sign

        yr = self.yaw_rad
        self.x += math.sin(yr) * self.speed * dt
        self.z += math.cos(yr) * self.speed * dt

        limit = 220.0
        self.x = max(-limit, min(limit, self.x))
        self.z = max(-limit, min(limit, self.z))

    def models(self):
        """Returns (body_model, cab_model) — float32 4×4 row-major matrices."""
        cy   = self.y
        base = mat_translate(self.x, cy, self.z) @ mat_rotate_y(self.yaw_rad)
        cab  = base @ mat_translate(0.0, self.BODY_H, 0.0)
        return base, cab

    def wheel_models(self):
        """Returns [fl, fr, rl, rr] matrices — each a wheel centred at (±wx, wr, ±wz) in car space."""
        cy   = self.y
        base = mat_translate(self.x, cy, self.z) @ mat_rotate_y(self.yaw_rad)
        wx, wr, wz = self.WHEEL_AX_X, self.WHEEL_RADIUS, self.WHEEL_AX_Z
        return [
            base @ mat_translate(-wx, wr, +wz),  # front-left
            base @ mat_translate(+wx, wr, +wz),  # front-right
            base @ mat_translate(-wx, wr, -wz),  # rear-left
            base @ mat_translate(+wx, wr, -wz),  # rear-right
        ]

    def follow_eye_target(self):
        """Returns (eye, target) for a follow camera positioned behind and above the car."""
        yr  = self.yaw_rad
        cy  = self.y
        eye = np.array([
            self.x - math.sin(yr) * 12.0,
            cy + 5.0,
            self.z - math.cos(yr) * 12.0,
        ], dtype=np.float64)
        target = np.array([self.x, cy + 0.8, self.z], dtype=np.float64)
        return eye, target


class Bird:
    """Small object flying above the scene, changing direction randomly."""

    # Body: small elongated torso (faces direction of flight along local +Z)
    BODY_W, BODY_H, BODY_D = 0.20, 0.22, 0.85
    # Wings: very wide and flat, centred on body, offset slightly upward
    WING_W, WING_H, WING_D = 1.80, 0.05, 0.55
    WING_Y = 0.07   # how far above the body centre the wings are placed

    _ALT_MIN = 18.0
    _ALT_MAX = 55.0
    _BOUND   = 200.0
    _SPD_MIN =  8.0
    _SPD_MAX = 14.0

    def __init__(self, x, y, z, rng):
        self.pos   = np.array([x, y, z], dtype=np.float64)
        self._rng  = rng
        self.vel   = np.zeros(3, dtype=np.float64)
        self.timer = 0.0
        self._pick_direction()

    def _pick_direction(self):
        angle       = self._rng.uniform(0.0, 2.0 * math.pi)
        speed       = self._rng.uniform(self._SPD_MIN, self._SPD_MAX)
        self.vel[0] = math.sin(angle) * speed
        self.vel[2] = math.cos(angle) * speed
        self.vel[1] = self._rng.uniform(-1.5, 1.5)
        self.timer  = self._rng.uniform(3.0, 8.0)

    def update(self, dt):
        self.pos   += self.vel * dt
        self.timer -= dt
        if self.timer <= 0.0:
            self._pick_direction()

        x, y, z = self.pos

        # Turn back toward origin when approaching world boundary
        if abs(x) > self._BOUND or abs(z) > self._BOUND:
            dx, dz = -x, -z
            spd_hz = math.hypot(self.vel[0], self.vel[2])
            d_len  = math.hypot(dx, dz)
            if d_len > 1e-3:
                self.vel[0] = dx / d_len * spd_hz
                self.vel[2] = dz / d_len * spd_hz
            self.timer = self._rng.uniform(2.0, 5.0)
            self.pos[0] = max(-self._BOUND, min(self._BOUND, x))
            self.pos[2] = max(-self._BOUND, min(self._BOUND, z))

        # Keep altitude in range
        if y < self._ALT_MIN:
            self.vel[1] =  abs(self.vel[1]) + 0.5
        elif y > self._ALT_MAX:
            self.vel[1] = -abs(self.vel[1]) - 0.5

    @property
    def yaw_rad(self):
        """Angle so that local +Z faces the horizontal velocity direction."""
        return math.atan2(self.vel[0], self.vel[2])

    def models(self):
        """Returns (body_model, wing_model) — float32 4×4 row-major matrices."""
        x, y, z = self.pos
        base = mat_translate(float(x), float(y), float(z)) @ mat_rotate_y(self.yaw_rad)
        wing = base @ mat_translate(0.0, self.WING_Y, 0.0)
        return base, wing


class Pedestrian:
    """Walks along the circuit road at a fixed lateral offset from the centreline."""

    # 4-part body: legs → torso → arms → head (stacked upward)
    LEG_W,   LEG_H,   LEG_D   = 0.44, 0.55, 0.32
    TORSO_W, TORSO_H, TORSO_D = 0.50, 0.50, 0.30
    ARM_W,   ARM_H,   ARM_D   = 0.95, 0.13, 0.18
    HEAD_SIZE                  = 0.33

    def __init__(self, path_pts, start_t, speed, side_offset, terrain_sampler):
        """
        path_pts    (N, 2) float32 — smoothed circuit centreline, columns = (x, z)
        start_t     float 0…N     — starting position along path
        speed       float         — path-points per second (negative = opposite direction)
        side_offset float         — signed perpendicular offset from centreline
                                    (positive = right side, negative = left side)
        """
        self.path        = path_pts
        self.t           = float(start_t)
        self.speed       = float(speed)
        self.side_offset = float(side_offset)
        self._sampler    = terrain_sampler
        self.knocked_down = False
        self._flat_x   = 0.0
        self._flat_z   = 0.0
        self._flat_yaw = 0.0

    def knock_down(self):
        """Freeze the pedestrian in place and lay them flat (called on car collision)."""
        if not self.knocked_down:
            self._flat_x, self._flat_z, self._flat_yaw = self._world_pos_yaw()
            self.knocked_down = True

    @property
    def world_pos(self):
        """Current (x, z) world position regardless of knocked-down state."""
        if self.knocked_down:
            return self._flat_x, self._flat_z
        x, z, _ = self._world_pos_yaw()
        return x, z

    def update(self, dt):
        if self.knocked_down:
            return
        N     = len(self.path)
        self.t = (self.t + self.speed * dt) % N

    def _world_pos_yaw(self):
        N    = len(self.path)
        i    = int(self.t) % N
        j    = (i + 1) % N
        frac = self.t - int(self.t)

        pi = self.path[i]
        pj = self.path[j]

        cx  = float(pi[0] + (pj[0] - pi[0]) * frac)
        cz  = float(pi[1] + (pj[1] - pi[1]) * frac)
        dx  = float(pj[0] - pi[0])
        dz  = float(pj[1] - pi[1])
        mag = math.hypot(dx, dz)
        if mag > 1e-6:
            dx /= mag
            dz /= mag

        # Perpendicular right-side direction: rotate tangent 90° CW → (dz, -dx)
        px, pz_perp = dz, -dx
        x = cx + px       * self.side_offset
        z = cz + pz_perp  * self.side_offset

        yaw = math.atan2(dx, dz) if self.speed >= 0 else math.atan2(-dx, -dz)
        return x, z, yaw

    def models(self):
        """Returns (legs_m, torso_m, arms_m, head_m) — float32 4×4 row-major matrices."""
        if self.knocked_down:
            y   = self._sampler(self._flat_x, self._flat_z)
            # rot_x(+π/2) tips the pedestrian forward so their body lies flat on the ground;
            # y-offsets (LEG_H etc.) become z-extents along the facing direction.
            rot   = (mat_translate(self._flat_x, y + 0.15, self._flat_z)
                     @ mat_rotate_y(self._flat_yaw)
                     @ mat_rotate_x(math.pi / 2))
            torso = rot @ mat_translate(0.0, self.LEG_H, 0.0)
            arms  = rot @ mat_translate(0.0, self.LEG_H + self.TORSO_H * 0.4, 0.0)
            head  = rot @ mat_translate(0.0, self.LEG_H + self.TORSO_H, 0.0)
            return rot, torso, arms, head

        x, z, yaw = self._world_pos_yaw()
        y    = self._sampler(x, z)
        base  = mat_translate(x, y, z) @ mat_rotate_y(yaw)
        torso = base  @ mat_translate(0.0, self.LEG_H, 0.0)
        arms  = torso @ mat_translate(0.0, self.TORSO_H * 0.4, 0.0)
        head  = torso @ mat_translate(0.0, self.TORSO_H, 0.0)
        return base, torso, arms, head
