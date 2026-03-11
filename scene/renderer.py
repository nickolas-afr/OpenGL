import sys

import numpy as np
import glfw
from OpenGL.GL import *

import scene.config as config
import scene.shaders as shaders
from core.gl_utils   import link_program, upload_texture, set_mat4
from core.math_utils import mat_perspective
from scene.textures  import make_grass_tex, make_sky_top_tex, make_horizon_tex, make_terrain_tex
from scene.geometry  import build_skybox, build_terrain, build_pyramid
from scene.camera    import Camera


def run():
    if not glfw.init():
        sys.exit("GLFW init failed")

    # macOS requires 3.3 Core Profile + forward-compatibility flag
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,  3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,  3)
    glfw.window_hint(glfw.OPENGL_PROFILE,         glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT,  GL_TRUE)

    win = glfw.create_window(config.WIN_W, config.WIN_H, config.TITLE, None, None)
    if not win:
        glfw.terminate()
        sys.exit("Window creation failed")

    glfw.make_context_current(win)
    glfw.swap_interval(1)
    glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)

    cam     = Camera(config.CAM_START, config.CAM_YAW, config.CAM_PITCH)
    mx, my  = [config.WIN_W / 2.0], [config.WIN_H / 2.0]
    first_m = [True]

    def on_mouse(_, xpos, ypos):
        if first_m[0]:
            mx[0], my[0] = xpos, ypos
            first_m[0]   = False
        cam.on_mouse(xpos - mx[0], ypos - my[0])
        mx[0], my[0] = xpos, ypos

    def on_key(window, key, _sc, action, _mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    glfw.set_cursor_pos_callback(win, on_mouse)
    glfw.set_key_callback(win, on_key)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    print("Compiling shaders…")
    sky_prog     = link_program(shaders.VERT_SKY,      shaders.FRAG_SKY)
    terrain_prog = link_program(shaders.VERT_TERRAIN,  shaders.FRAG_TERRAIN)
    pyramid_prog = link_program(shaders.VERT_PYRAMID,  shaders.FRAG_PYRAMID)

    print("Generating procedural textures…")
    tex_grass   = upload_texture(make_grass_tex(),    repeat=True)
    tex_sky     = upload_texture(make_sky_top_tex(),  repeat=False)
    tex_horizon = upload_texture(make_horizon_tex(),  repeat=False)
    tex_terrain = upload_texture(make_terrain_tex(),  repeat=True)

    print("Building geometry…")
    sky_textures     = [tex_grass, tex_sky,
                        tex_horizon, tex_horizon, tex_horizon, tex_horizon]
    sky_faces        = build_skybox(config.SKY_W, config.SKY_H, config.SKY_D)
    ter_vao, ter_cnt = build_terrain(config.TERRAIN_HALF, config.TERRAIN_DIVS,
                                     config.TERRAIN_HEIGHT, config.TERRAIN_TILE)
    pyr_vao, pyr_cnt = build_pyramid(config.PYRAMID_BASE_HALF, config.PYRAMID_HEIGHT)

    fw, fh = glfw.get_framebuffer_size(win)
    glViewport(0, 0, fw, fh)
    proj   = mat_perspective(config.FOV_DEG, fw / max(fh, 1), config.NEAR, config.FAR)
    ident4 = np.eye(4, dtype=np.float32)

    print(f"Entering render loop  |  OpenGL {glGetString(GL_VERSION).decode()}")
    print("WASD + Mouse to navigate  |  SPACE / SHIFT = up / down  |  ESC = quit")

    t_prev = glfw.get_time()
    frame  = 0

    while not glfw.window_should_close(win):
        t_now  = glfw.get_time()
        dt     = t_now - t_prev
        t_prev = t_now
        frame += 1

        if frame % 60 == 0 and dt > 0:
            glfw.set_window_title(win, f"{config.TITLE}  |  {1.0 / dt:.0f} FPS")

        cam.on_keys(win, dt)

        glClearColor(0.05, 0.08, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = cam.view()
        mvp  = proj @ view

        # Pass 1: Skybox (no depth write; culling off – camera is inside)
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)

        glUseProgram(sky_prog)
        set_mat4(sky_prog, "uMVP", mvp)
        glUniform1i(glGetUniformLocation(sky_prog, "uTex"), 0)
        glActiveTexture(GL_TEXTURE0)

        for (vao, cnt), tid in zip(sky_faces, sky_textures):
            glBindTexture(GL_TEXTURE_2D, tid)
            glBindVertexArray(vao)
            glDrawElements(GL_TRIANGLES, cnt, GL_UNSIGNED_INT, None)

        glDepthMask(GL_TRUE)
        glEnable(GL_CULL_FACE)

        # Pass 2: Terrain (Lambertian shading)
        glUseProgram(terrain_prog)
        set_mat4(terrain_prog, "uMVP",   mvp)
        set_mat4(terrain_prog, "uModel", ident4)
        glUniform1i (glGetUniformLocation(terrain_prog, "uTex"),      0)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uSunDir"),   1, config.SUN_DIR)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uSunColor"), 1, config.SUN_COLOR)
        glUniform1f (glGetUniformLocation(terrain_prog, "uAmbient"),  config.AMBIENT)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_terrain)
        glBindVertexArray(ter_vao)
        glDrawElements(GL_TRIANGLES, ter_cnt, GL_UNSIGNED_INT, None)

        # Pass 3: Pyramid (Lambertian shading, solid sandy colour)
        glUseProgram(pyramid_prog)
        set_mat4(pyramid_prog, "uMVP",   mvp)
        set_mat4(pyramid_prog, "uModel", ident4)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"),    1, config.PYRAMID_COLOR)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uSunDir"),   1, config.SUN_DIR)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uSunColor"), 1, config.SUN_COLOR)
        glUniform1f (glGetUniformLocation(pyramid_prog, "uAmbient"),  config.AMBIENT)

        glBindVertexArray(pyr_vao)
        glDrawArrays(GL_TRIANGLES, 0, pyr_cnt)

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()
