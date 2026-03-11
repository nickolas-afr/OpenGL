import math

import numpy as np
import glfw

from scene.config import CAM_SPEED, MOUSE_SENS
from core.math_utils import mat_look_at


class Camera:
    def __init__(self, pos, yaw, pitch):
        self.pos   = np.array(pos, dtype=np.float64)
        self.yaw   = float(yaw)
        self.pitch = float(pitch)
        self._refresh()

    def _refresh(self):
        yr = math.radians(self.yaw)
        pr = math.radians(self.pitch)
        f  = np.array([math.cos(pr) * math.cos(yr),
                        math.sin(pr),
                        math.cos(pr) * math.sin(yr)])
        self.front = f / np.linalg.norm(f)
        r = np.cross(self.front, (0.0, 1.0, 0.0))
        rn = np.linalg.norm(r)
        self.right = r / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])

    def on_mouse(self, dx, dy):
        self.yaw   = (self.yaw + dx * MOUSE_SENS) % 360.0
        self.pitch = float(np.clip(self.pitch - dy * MOUSE_SENS, -89.0, 89.0))
        self._refresh()

    def on_keys(self, window, dt):
        s = CAM_SPEED * dt
        if glfw.get_key(window, glfw.KEY_W)          == glfw.PRESS: self.pos += self.front * s
        if glfw.get_key(window, glfw.KEY_S)          == glfw.PRESS: self.pos -= self.front * s
        if glfw.get_key(window, glfw.KEY_A)          == glfw.PRESS: self.pos -= self.right * s
        if glfw.get_key(window, glfw.KEY_D)          == glfw.PRESS: self.pos += self.right * s
        if glfw.get_key(window, glfw.KEY_SPACE)      == glfw.PRESS: self.pos[1] += s
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS: self.pos[1] -= s

    def view(self):
        return mat_look_at(self.pos, self.pos + self.front)
