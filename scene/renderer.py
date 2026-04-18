import sys

import numpy as np
import glfw
from OpenGL.GL import *

import scene.config as config
import scene.shaders as shaders
from core.gl_utils   import link_program, upload_texture, set_mat4
from core.math_utils import mat_perspective, mat_translate, mat_ortho, mat_look_at
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
    shadow_prog  = link_program(shaders.VERT_SHADOW,   shaders.FRAG_SHADOW)

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
    pole_shaft_vao, pole_shaft_cnt = build_box_solid(0.4, config.POLE_HEIGHT, 0.4)
    pole_lamp_vao,  pole_lamp_cnt  = build_box_solid(1.2, 0.5, 0.5)

    fw, fh = glfw.get_framebuffer_size(win)
    glViewport(0, 0, fw, fh)
    proj   = mat_perspective(config.FOV_DEG, fw / max(fh, 1), config.NEAR, config.FAR)
    ident4 = np.eye(4, dtype=np.float32)

    # Shadow map FBO + depth texture
    shadow_tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, shadow_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24,
                 config.SHADOW_MAP_SIZE, config.SHADOW_MAP_SIZE,
                 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR,
                     np.ones(4, dtype=np.float32))   # outside map → no shadow
    shadow_fbo = int(glGenFramebuffers(1))
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, shadow_tex, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # Directional (sun) light-space matrix — orthographic from sun's position
    sun_norm   = config.SUN_DIR / np.linalg.norm(config.SUN_DIR)
    light_view = mat_look_at(sun_norm * 350.0, (0.0, 0.0, 0.0))
    light_proj = mat_ortho(-280.0, 280.0, -280.0, 280.0, 0.1, 800.0)
    light_mat  = light_proj @ light_view

    # Precompute lamp-head world positions for the pole-light shader uniforms.
    # Each lamp head sits at (px, terrain_y + POLE_HEIGHT, pz).
    pole_lamps = np.array(
        [[px, terrain_sampler(px, pz) + config.POLE_HEIGHT, pz]
         for px, pz in config.POLE_POSITIONS],
        dtype=np.float32,
    )
    num_poles = len(config.POLE_POSITIONS)

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

        # ── SHADOW PASS ──────────────────────────────────────────────────────
        glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo)
        glViewport(0, 0, config.SHADOW_MAP_SIZE, config.SHADOW_MAP_SIZE)
        glClear(GL_DEPTH_BUFFER_BIT)
        glDisable(GL_CULL_FACE)
        glUseProgram(shadow_prog)

        # terrain + circuit (model = identity)
        set_mat4(shadow_prog, "uLightMVP", light_mat)
        glBindVertexArray(ter_vao)
        glDrawElements(GL_TRIANGLES, ter_cnt, GL_UNSIGNED_INT, None)
        glBindVertexArray(cir_vao)
        glDrawElements(GL_TRIANGLES, cir_cnt, GL_UNSIGNED_INT, None)

        # buildings
        for (bx, bz, bw, bh, bd), (bvao, bcnt) in zip(config.BUILDINGS, building_vaos):
            by = terrain_sampler(bx, bz)
            set_mat4(shadow_prog, "uLightMVP", light_mat @ mat_translate(bx, by, bz))
            glBindVertexArray(bvao)
            glDrawElements(GL_TRIANGLES, bcnt, GL_UNSIGNED_INT, None)

        # trees
        for tx, tz in config.TREE_POSITIONS:
            ty = terrain_sampler(tx, tz)
            set_mat4(shadow_prog, "uLightMVP", light_mat @ mat_translate(tx, ty, tz))
            glBindVertexArray(trunk_vao)
            glDrawElements(GL_TRIANGLES, trunk_cnt, GL_UNSIGNED_INT, None)
            set_mat4(shadow_prog, "uLightMVP",
                     light_mat @ mat_translate(tx, ty + config.TRUNK_H, tz))
            glBindVertexArray(canopy_vao)
            glDrawElements(GL_TRIANGLES, canopy_cnt, GL_UNSIGNED_INT, None)

        # poles
        for px, pz in config.POLE_POSITIONS:
            py = terrain_sampler(px, pz)
            set_mat4(shadow_prog, "uLightMVP", light_mat @ mat_translate(px, py, pz))
            glBindVertexArray(pole_shaft_vao)
            glDrawElements(GL_TRIANGLES, pole_shaft_cnt, GL_UNSIGNED_INT, None)
            set_mat4(shadow_prog, "uLightMVP",
                     light_mat @ mat_translate(px, py + config.POLE_HEIGHT, pz))
            glBindVertexArray(pole_lamp_vao)
            glDrawElements(GL_TRIANGLES, pole_lamp_cnt, GL_UNSIGNED_INT, None)

        # pyramid
        set_mat4(shadow_prog, "uLightMVP", light_mat)
        glBindVertexArray(pyr_vao)
        glDrawElements(GL_TRIANGLES, pyr_cnt, GL_UNSIGNED_INT, None)

        glEnable(GL_CULL_FACE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, fw, fh)
        # ─────────────────────────────────────────────────────────────────────

        glClearColor(0.05, 0.08, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = cam.view()
        mvp  = proj @ view

        # Bind shadow map to unit 1 for all lit passes
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, shadow_tex)

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

        # Pass 2: Terrain (Lambertian shading + shadow + pole lights)
        glUseProgram(terrain_prog)
        set_mat4(terrain_prog, "uMVP",      mvp)
        set_mat4(terrain_prog, "uModel",    ident4)
        set_mat4(terrain_prog, "uLightMat", light_mat)
        glUniform1i (glGetUniformLocation(terrain_prog, "uTex"),           0)
        glUniform1i (glGetUniformLocation(terrain_prog, "uShadowMap"),     1)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uSunDir"),        1, config.SUN_DIR)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uSunColor"),      1, config.SUN_COLOR)
        glUniform1f (glGetUniformLocation(terrain_prog, "uAmbient"),       config.AMBIENT)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uPoleLights"),    num_poles, pole_lamps)
        glUniform1i (glGetUniformLocation(terrain_prog, "uNumPoles"),      num_poles)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uPoleColor"),     1, config.POLE_LIGHT_COLOR)
        glUniform1f (glGetUniformLocation(terrain_prog, "uPoleIntensity"), config.POLE_LIGHT_INTENSITY)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_terrain)
        glBindVertexArray(ter_vao)
        glDrawElements(GL_TRIANGLES, ter_cnt, GL_UNSIGNED_INT, None)

        # Pass 3: Circuit road (terrain shader, tiled road texture)
        glUseProgram(terrain_prog)
        set_mat4(terrain_prog, "uMVP",      mvp)
        set_mat4(terrain_prog, "uModel",    ident4)
        set_mat4(terrain_prog, "uLightMat", light_mat)
        glUniform1i (glGetUniformLocation(terrain_prog, "uTex"),           0)
        glUniform1i (glGetUniformLocation(terrain_prog, "uShadowMap"),     1)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uSunDir"),        1, config.SUN_DIR)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uSunColor"),      1, config.SUN_COLOR)
        glUniform1f (glGetUniformLocation(terrain_prog, "uAmbient"),       config.AMBIENT)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uPoleLights"),    num_poles, pole_lamps)
        glUniform1i (glGetUniformLocation(terrain_prog, "uNumPoles"),      num_poles)
        glUniform3fv(glGetUniformLocation(terrain_prog, "uPoleColor"),     1, config.POLE_LIGHT_COLOR)
        glUniform1f (glGetUniformLocation(terrain_prog, "uPoleIntensity"), config.POLE_LIGHT_INTENSITY)
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
            set_mat4(terrain_prog, "uMVP",      mvp_obj)
            set_mat4(terrain_prog, "uModel",    model)
            set_mat4(terrain_prog, "uLightMat", light_mat @ model)
            glBindVertexArray(bvao)
            glDrawElements(GL_TRIANGLES, bcnt, GL_UNSIGNED_INT, None)

        # Pass 5: Trees — trunk (box) + canopy (cone), solid colour
        glUseProgram(pyramid_prog)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uSunDir"),        1, config.SUN_DIR)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uSunColor"),      1, config.SUN_COLOR)
        glUniform1f (glGetUniformLocation(pyramid_prog, "uAmbient"),       config.AMBIENT)
        glUniform1i (glGetUniformLocation(pyramid_prog, "uShadowMap"),     1)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uPoleLights"),    num_poles, pole_lamps)
        glUniform1i (glGetUniformLocation(pyramid_prog, "uNumPoles"),      num_poles)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uPoleColor"),     1, config.POLE_LIGHT_COLOR)
        glUniform1f (glGetUniformLocation(pyramid_prog, "uPoleIntensity"), config.POLE_LIGHT_INTENSITY)
        for tx, tz in config.TREE_POSITIONS:
            ty = terrain_sampler(tx, tz)
            # Trunk
            model_t = mat_translate(tx, ty, tz)
            set_mat4(pyramid_prog, "uMVP",      proj @ view @ model_t)
            set_mat4(pyramid_prog, "uModel",    model_t)
            set_mat4(pyramid_prog, "uLightMat", light_mat @ model_t)
            glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, config.TRUNK_COLOR)
            glBindVertexArray(trunk_vao)
            glDrawElements(GL_TRIANGLES, trunk_cnt, GL_UNSIGNED_INT, None)
            # Canopy sits on top of trunk
            model_c = mat_translate(tx, ty + config.TRUNK_H, tz)
            set_mat4(pyramid_prog, "uMVP",      proj @ view @ model_c)
            set_mat4(pyramid_prog, "uModel",    model_c)
            set_mat4(pyramid_prog, "uLightMat", light_mat @ model_c)
            glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, config.CANOPY_COLOR)
            glBindVertexArray(canopy_vao)
            glDrawElements(GL_TRIANGLES, canopy_cnt, GL_UNSIGNED_INT, None)

        # Pass 6: Pyramid (Lambertian shading + shadow, solid sandy colour)
        glUseProgram(pyramid_prog)
        set_mat4(pyramid_prog, "uMVP",      mvp)
        set_mat4(pyramid_prog, "uModel",    ident4)
        set_mat4(pyramid_prog, "uLightMat", light_mat)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"),    1, config.PYRAMID_COLOR)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uSunDir"),   1, config.SUN_DIR)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uSunColor"), 1, config.SUN_COLOR)
        glUniform1f (glGetUniformLocation(pyramid_prog, "uAmbient"),  config.AMBIENT)
        glUniform1i (glGetUniformLocation(pyramid_prog, "uShadowMap"), 1)

        glBindVertexArray(pyr_vao)
        glDrawArrays(GL_TRIANGLES, 0, pyr_cnt)

        # Pass 7: Lighting poles — shaft + lamp head
        for px, pz in config.POLE_POSITIONS:
            py = terrain_sampler(px, pz)
            # Shaft
            model_s = mat_translate(px, py, pz)
            set_mat4(pyramid_prog, "uMVP",      proj @ view @ model_s)
            set_mat4(pyramid_prog, "uModel",    model_s)
            set_mat4(pyramid_prog, "uLightMat", light_mat @ model_s)
            glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, config.POLE_COLOR)
            glBindVertexArray(pole_shaft_vao)
            glDrawElements(GL_TRIANGLES, pole_shaft_cnt, GL_UNSIGNED_INT, None)
            # Lamp head on top of shaft
            model_l = mat_translate(px, py + config.POLE_HEIGHT, pz)
            set_mat4(pyramid_prog, "uMVP",      proj @ view @ model_l)
            set_mat4(pyramid_prog, "uModel",    model_l)
            set_mat4(pyramid_prog, "uLightMat", light_mat @ model_l)
            glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, config.LAMP_COLOR)
            glBindVertexArray(pole_lamp_vao)
            glDrawElements(GL_TRIANGLES, pole_lamp_cnt, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()
