#!/usr/bin/env python3
"""
Phase 1 – Scene Definition & Environment
==========================================
OpenGL 3.3 Core Profile | macOS Apple Silicon / Intel compatible

Controls:
  W / S           – fly forward / backward
  A / D           – strafe left / right
  SPACE / L-SHIFT – fly up / down
  Mouse           – look around (yaw + pitch)
  ESC             – quit
"""

from scene.renderer import run


if __name__ == "__main__":
    run()
