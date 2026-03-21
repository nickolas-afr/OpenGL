import sys

import numpy as np
import glfw
from OpenGL.GL import *

import scene.config as config
import scene.shaders as shaders
from core.gl_utils   import link_program, upload_texture, set_mat4
from core.math_utils import mat_perspective, mat_translate
from scene.textures  import make_grass_tex, make_sky_top_tex, make_horizon_tex, make_terrain_tex, make_road_tex, make_building_tex
from scene.geometry  import build_skybox, build_terrain, build_pyramid, build_circuit, build_box, build_box_solid, build_cone, make_height_grid, TerrainSampler
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
    tex_road     = upload_texture(make_road_tex(),     repeat=True)
    tex_building = upload_texture(make_building_tex(), repeat=True)

    print("Building geometry…")
    sky_textures     = [tex_grass, tex_sky,
                        tex_horizon, tex_horizon, tex_horizon, tex_horizon]
    sky_faces        = build_skybox(config.SKY_W, config.SKY_H, config.SKY_D)
    ter_vao, ter_cnt = build_terrain(config.TERRAIN_HALF, config.TERRAIN_DIVS,
                                     config.TERRAIN_HEIGHT, config.TERRAIN_TILE)
    H_grid, _        = make_height_grid(config.TERRAIN_HALF, config.TERRAIN_DIVS,
                                        config.TERRAIN_HEIGHT)
    terrain_sampler  = TerrainSampler(H_grid, config.TERRAIN_HALF)
    pyr_vao, pyr_cnt = build_pyramid(config.PYRAMID_BASE_HALF, config.PYRAMID_HEIGHT)
    cir_vao,  cir_cnt  = build_circuit(config.CIRCUIT_WAYPOINTS,
                                        config.CIRCUIT_ROAD_HALF_W,
                                        config.CIRCUIT_Y,
                                        height_sampler=terrain_sampler)
    building_vaos = [build_box(bw, bh, bd)
                     for _, _, bw, bh, bd in config.BUILDINGS]
    trunk_vao,  trunk_cnt  = build_box_solid(1.5, config.TRUNK_H,  1.5)
    canopy_vao, canopy_cnt = build_cone(config.CANOPY_R, config.CANOPY_H, 12)

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

        # Pass 3: Circuit road (terrain shader, tiled road texture)
        glUseProgram(terrain_prog)
        set_mat4(terrain_prog, "uMVP",   mvp)
        set_mat4(terrain_prog, "uModel", ident4)
        glUniform1i (glGetUniformLocation(terrain_prog, "uTex"),      0)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uSunDir"),   1, config.SUN_DIR)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uSunColor"), 1, config.SUN_COLOR)
        glUniform1f (glGetUniformLocation(terrain_prog, "uAmbient"),  config.AMBIENT)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_road)
        glBindVertexArray(cir_vao)
        glDrawElements(GL_TRIANGLES, cir_cnt, GL_UNSIGNED_INT, None)

        # Pass 4: Buildings (terrain shader, building texture)
        glBindTexture(GL_TEXTURE_2D, tex_building)
        for (bx, bz, bw, bh, bd), (bvao, bcnt) in zip(config.BUILDINGS, building_vaos):
            by      = terrain_sampler(bx, bz)
            model   = mat_translate(bx, by, bz)
            mvp_obj = proj @ view @ model
            set_mat4(terrain_prog, "uMVP",   mvp_obj)
            set_mat4(terrain_prog, "uModel", model)
            glBindVertexArray(bvao)
            glDrawElements(GL_TRIANGLES, bcnt, GL_UNSIGNED_INT, None)

        # Pass 5: Trees — trunk (box) + canopy (cone), solid colour
        glUseProgram(pyramid_prog)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uSunDir"),   1, config.SUN_DIR)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uSunColor"), 1, config.SUN_COLOR)
        glUniform1f (glGetUniformLocation(pyramid_prog, "uAmbient"),  config.AMBIENT)
        for tx, tz in config.TREE_POSITIONS:
            ty = terrain_sampler(tx, tz)
            # Trunk
            model_t = mat_translate(tx, ty, tz)
            set_mat4(pyramid_prog, "uMVP",   proj @ view @ model_t)
            set_mat4(pyramid_prog, "uModel", model_t)
            glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, config.TRUNK_COLOR)
            glBindVertexArray(trunk_vao)
            glDrawElements(GL_TRIANGLES, trunk_cnt, GL_UNSIGNED_INT, None)
            # Canopy sits on top of trunk
            model_c = mat_translate(tx, ty + config.TRUNK_H, tz)
            set_mat4(pyramid_prog, "uMVP",   proj @ view @ model_c)
            set_mat4(pyramid_prog, "uModel", model_c)
            glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, config.CANOPY_COLOR)
            glBindVertexArray(canopy_vao)
            glDrawElements(GL_TRIANGLES, canopy_cnt, GL_UNSIGNED_INT, None)

        # Pass 6: Pyramid (Lambertian shading, solid sandy colour)
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
