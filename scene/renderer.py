import sys

import numpy as np
import glfw
from OpenGL.GL import *

import scene.config as config
import scene.shaders as shaders
from core.gl_utils   import link_program, upload_texture, set_mat4
from core.math_utils import mat_perspective, mat_translate, mat_ortho, mat_look_at
from scene.textures  import make_grass_tex, make_sky_top_tex, make_horizon_tex, make_terrain_tex, make_road_tex, make_building_tex
from scene.geometry  import build_skybox, build_terrain, build_pyramid, build_circuit, build_box, build_box_solid, build_cone, build_wheel_solid, make_height_grid, TerrainSampler, RoadAwareSampler, get_circuit_path
from scene.camera    import Camera
from scene.entities  import Car, Bird, Pedestrian


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

    cam        = Camera(config.CAM_START, config.CAM_YAW, config.CAM_PITCH)
    mx, my     = [config.WIN_W / 2.0], [config.WIN_H / 2.0]
    first_m    = [True]
    follow_cam = [False]   # True = lock camera behind car

    def on_mouse(_, xpos, ypos):
        if first_m[0]:
            mx[0], my[0] = xpos, ypos
            first_m[0]   = False
            return
        if not follow_cam[0]:
            cam.on_mouse(xpos - mx[0], ypos - my[0])
        mx[0], my[0] = xpos, ypos

    def on_key(window, key, _sc, action, _mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        if key == glfw.KEY_C and action == glfw.PRESS:
            follow_cam[0] = not follow_cam[0]

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
    car_body_vao,   car_body_cnt   = build_box_solid(Car.BODY_W, Car.BODY_H, Car.BODY_D)
    car_cab_vao,    car_cab_cnt    = build_box_solid(Car.CAB_W,  Car.CAB_H,  Car.CAB_D)
    wheel_vao,      wheel_cnt      = build_wheel_solid(Car.WHEEL_RADIUS, Car.WHEEL_T)
    bird_body_vao,  bird_body_cnt  = build_box_solid(Bird.BODY_W, Bird.BODY_H, Bird.BODY_D)
    bird_wing_vao,  bird_wing_cnt  = build_box_solid(Bird.WING_W, Bird.WING_H, Bird.WING_D)
    ped_leg_vao,    ped_leg_cnt    = build_box_solid(Pedestrian.LEG_W,   Pedestrian.LEG_H,   Pedestrian.LEG_D)
    ped_torso_vao,  ped_torso_cnt  = build_box_solid(Pedestrian.TORSO_W, Pedestrian.TORSO_H, Pedestrian.TORSO_D)
    ped_arm_vao,    ped_arm_cnt    = build_box_solid(Pedestrian.ARM_W,   Pedestrian.ARM_H,   Pedestrian.ARM_D)
    ped_head_vao,   ped_head_cnt   = build_box_solid(Pedestrian.HEAD_SIZE, Pedestrian.HEAD_SIZE, Pedestrian.HEAD_SIZE)

    # ── Entity initialisation ─────────────────────────────────────────────
    rng          = np.random.default_rng(42)
    circuit_path = get_circuit_path(config.CIRCUIT_WAYPOINTS)
    N_path       = len(circuit_path)
    road_sampler = RoadAwareSampler(terrain_sampler, circuit_path,
                                    config.CIRCUIT_ROAD_HALF_W, config.CIRCUIT_Y)

    car = Car(config.CAR_START_X, config.CAR_START_Z, config.CAR_START_YAW,
              road_sampler)
    birds = [
        Bird(float(rng.uniform(-180, 180)),
             float(rng.uniform(20, 50)),
             float(rng.uniform(-180, 180)),
             rng)
        for _ in range(config.NUM_BIRDS)
    ]
    pedestrians = [
        Pedestrian(circuit_path, 0 * N_path // 5,  0.60, -4.5, road_sampler),
        Pedestrian(circuit_path, 1 * N_path // 5,  0.50,  4.5, road_sampler),
        Pedestrian(circuit_path, 2 * N_path // 5,  0.65, -4.5, road_sampler),
        Pedestrian(circuit_path, 3 * N_path // 5, -0.45,  4.5, road_sampler),
        Pedestrian(circuit_path, 4 * N_path // 5,  0.55, -3.0, road_sampler),
    ]

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
    print("WASD + Mouse = free camera  |  SPACE / SHIFT = up / down  |  ESC = quit")
    print("Arrow keys = drive car  |  C = toggle follow-car camera")

    def _check_collision(cx, cz, cr):
        """Return True if a circle (cx, cz, cr) overlaps any static obstacle."""
        for bx, bz, bw, _bh, bd in config.BUILDINGS:
            hw, hd = bw * 0.5, bd * 0.5
            nx = max(bx - hw, min(bx + hw, cx))
            nz = max(bz - hd, min(bz + hd, cz))
            if (cx - nx) ** 2 + (cz - nz) ** 2 < cr * cr:
                return True
        tree_r = 1.5
        for tx, tz in config.TREE_POSITIONS:
            if (cx - tx) ** 2 + (cz - tz) ** 2 < (cr + tree_r) ** 2:
                return True
        pole_r = 0.5
        for px, pz in config.POLE_POSITIONS:
            if (cx - px) ** 2 + (cz - pz) ** 2 < (cr + pole_r) ** 2:
                return True
        return False

    t_prev = glfw.get_time()
    frame  = 0

    while not glfw.window_should_close(win):
        t_now  = glfw.get_time()
        dt     = t_now - t_prev
        t_prev = t_now
        frame += 1

        if frame % 60 == 0 and dt > 0:
            mode = "Follow Cam" if follow_cam[0] else "Free Cam"
            glfw.set_window_title(win, f"{config.TITLE}  |  {1.0 / dt:.0f} FPS  |  {mode}")

        # ── Entity updates ────────────────────────────────────────────────────
        prev_x, prev_z = car.x, car.z
        car.update(win, dt)
        if _check_collision(car.x, car.z, Car.COLLISION_RADIUS):
            car.x, car.z  = prev_x, prev_z
            car.speed    *= -0.2   # slight bounce on impact
        # Knock down any pedestrian the car drives over
        _PED_HIT_R2 = 2.0 ** 2
        for ped in pedestrians:
            if not ped.knocked_down:
                px, pz = ped.world_pos
                if (car.x - px) ** 2 + (car.z - pz) ** 2 < _PED_HIT_R2:
                    ped.knock_down()
        for bird in birds:
            bird.update(dt)
        for ped in pedestrians:
            ped.update(dt)

        # ── Camera ────────────────────────────────────────────────────────────
        if follow_cam[0]:
            _fol_eye, _fol_tgt = car.follow_eye_target()
            cam.pos[:] = _fol_eye   # sync so free-cam resumes from here
        else:
            _fol_eye, _fol_tgt = None, None
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

        # Car
        car_bm, car_cm = car.models()
        set_mat4(shadow_prog, "uLightMVP", light_mat @ car_bm)
        glBindVertexArray(car_body_vao)
        glDrawElements(GL_TRIANGLES, car_body_cnt, GL_UNSIGNED_INT, None)
        set_mat4(shadow_prog, "uLightMVP", light_mat @ car_cm)
        glBindVertexArray(car_cab_vao)
        glDrawElements(GL_TRIANGLES, car_cab_cnt, GL_UNSIGNED_INT, None)
        for wm in car.wheel_models():
            set_mat4(shadow_prog, "uLightMVP", light_mat @ wm)
            glBindVertexArray(wheel_vao)
            glDrawElements(GL_TRIANGLES, wheel_cnt, GL_UNSIGNED_INT, None)

        # Pedestrians
        for ped in pedestrians:
            pl, pt, pa, ph = ped.models()
            for pm, pv, pc in [(pl, ped_leg_vao, ped_leg_cnt),
                               (pt, ped_torso_vao, ped_torso_cnt),
                               (pa, ped_arm_vao,   ped_arm_cnt),
                               (ph, ped_head_vao,  ped_head_cnt)]:
                set_mat4(shadow_prog, "uLightMVP", light_mat @ pm)
                glBindVertexArray(pv)
                glDrawElements(GL_TRIANGLES, pc, GL_UNSIGNED_INT, None)

        glEnable(GL_CULL_FACE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, fw, fh)
        # ─────────────────────────────────────────────────────────────────────

        glClearColor(0.05, 0.08, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = (mat_look_at(_fol_eye, _fol_tgt)
                if follow_cam[0] else cam.view())
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

        # Pass 8: Car  (pyramid_prog; global uniforms still set from Pass 5)
        car_bm, car_cm = car.models()
        set_mat4(pyramid_prog, "uMVP",      proj @ view @ car_bm)
        set_mat4(pyramid_prog, "uModel",    car_bm)
        set_mat4(pyramid_prog, "uLightMat", light_mat @ car_bm)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, config.CAR_COLOR_BODY)
        glBindVertexArray(car_body_vao)
        glDrawElements(GL_TRIANGLES, car_body_cnt, GL_UNSIGNED_INT, None)

        set_mat4(pyramid_prog, "uMVP",      proj @ view @ car_cm)
        set_mat4(pyramid_prog, "uModel",    car_cm)
        set_mat4(pyramid_prog, "uLightMat", light_mat @ car_cm)
        glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, config.CAR_COLOR_CAB)
        glBindVertexArray(car_cab_vao)
        glDrawElements(GL_TRIANGLES, car_cab_cnt, GL_UNSIGNED_INT, None)

        glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, config.CAR_WHEEL_COLOR)
        glBindVertexArray(wheel_vao)
        for wm in car.wheel_models():
            set_mat4(pyramid_prog, "uMVP",      proj @ view @ wm)
            set_mat4(pyramid_prog, "uModel",    wm)
            set_mat4(pyramid_prog, "uLightMat", light_mat @ wm)
            glDrawElements(GL_TRIANGLES, wheel_cnt, GL_UNSIGNED_INT, None)

        # Pass 9: Birds
        for bird in birds:
            bb, bw = bird.models()
            set_mat4(pyramid_prog, "uMVP",      proj @ view @ bb)
            set_mat4(pyramid_prog, "uModel",    bb)
            set_mat4(pyramid_prog, "uLightMat", light_mat @ bb)
            glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, config.BIRD_COLOR)
            glBindVertexArray(bird_body_vao)
            glDrawElements(GL_TRIANGLES, bird_body_cnt, GL_UNSIGNED_INT, None)
            set_mat4(pyramid_prog, "uMVP",      proj @ view @ bw)
            set_mat4(pyramid_prog, "uModel",    bw)
            set_mat4(pyramid_prog, "uLightMat", light_mat @ bw)
            glBindVertexArray(bird_wing_vao)
            glDrawElements(GL_TRIANGLES, bird_wing_cnt, GL_UNSIGNED_INT, None)

        # Pass 10: Pedestrians
        for i, ped in enumerate(pedestrians):
            pl, pt, pa, ph = ped.models()
            jacket = config.PEDESTRIAN_COLORS[i % len(config.PEDESTRIAN_COLORS)]
            for pm, pv, pc, col in [
                (pl, ped_leg_vao,   ped_leg_cnt,   config.PED_PANTS_COLOR),
                (pt, ped_torso_vao, ped_torso_cnt, jacket),
                (pa, ped_arm_vao,   ped_arm_cnt,   jacket),
                (ph, ped_head_vao,  ped_head_cnt,  config.PED_HEAD_COLOR),
            ]:
                set_mat4(pyramid_prog, "uMVP",      proj @ view @ pm)
                set_mat4(pyramid_prog, "uModel",    pm)
                set_mat4(pyramid_prog, "uLightMat", light_mat @ pm)
                glUniform3fv(glGetUniformLocation(pyramid_prog, "uColor"), 1, col)
                glBindVertexArray(pv)
                glDrawElements(GL_TRIANGLES, pc, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()
