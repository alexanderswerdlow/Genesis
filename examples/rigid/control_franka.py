import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    self.jenga = self.scene.add_entity(
                gs.morphs.MJCF(file='examples/manipulation/jenga.xml'),
    )

    cam = scene.add_camera(
                res    = (1280, 960),
                pos    = (3.5, 0.0, 2.5),
                lookat = (0, 0, 0.5),
                fov    = 30,
                GUI    = False
            )
    scene.build()

    jnt_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    ]
    dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

    cam.start_recording()

    # Optional: set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        dofs_idx,
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        dofs_idx,
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        dofs_idx,
    )
    # Hard reset
    for i in range(200):
        cam.render()

        if i < 50:
            franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)
        elif i < 100:
            franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0.02, 0.02]), dofs_idx)
        elif i < 150:
            franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
        else:
            franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0.08, 0.08]), dofs_idx)

        scene.step()

        scene.step()

    cam.stop_recording(save_to_filename=f'video_test.mp4', fps=25)


if __name__ == "__main__":
    main()
