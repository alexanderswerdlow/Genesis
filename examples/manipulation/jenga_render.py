import argparse
import genesis as gs
import torch
import math
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()
    dt = 0.01

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=4
        ),
        rigid_options=gs.options.RigidOptions(
            dt=dt / 2,
            constraint_solver=gs.constraint_solver.Newton,
            use_contact_island=True,
            use_hibernation=True,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    num_layers = 3
    scale = 1.0
    base_pos = (0.4, 0.4, 0.05)
    block_height = (0.04 * scale) + 2e-3
    delta_x = (0.05 * scale) + 2e-3

    jenga = []
    for layer in range(num_layers):
        euler = (0, 0, (layer % 2) * 90)
        z_pos = base_pos[2] + layer * block_height        
        offsets = [
            (0, -delta_x, 0),  # Left block
            (0, 0, 0),      # Center block
            (0, delta_x, 0)    # Right block
        ] if layer % 2 == 0 else [
            (-delta_x, 0, 0),  # Left block
            (0, 0, 0),      # Center block
            (delta_x, 0, 0)    # Right block
         ]
        
        for dx, dy, dz in offsets:
            jenga.append(scene.add_entity(
                gs.morphs.URDF(
                    file="examples/manipulation/assets/jenga.urdf",
                    pos=(
                        base_pos[0] + dx,
                        base_pos[1] + dy,
                        z_pos
                    ),
                    euler=euler,
                    scale=scale,
                    fixed=False,
                )
            ))

    cam = scene.add_camera(
        res    = (1280, 960),
        pos    = (2.5, 0.0, 2.0),
        lookat = (0, 0, 0.2),
        fov    = 50,
        GUI    = False
    )
    
    scene.build(compile_kernels=True)

    # init_jenga_pos = jenga.get_links_pos()
    # init_pos = init_jenga_pos.reshape(-1, 3)
    # init_pos += torch.tensor([0.5, 0.5, 0.5], device=init_jenga_pos.device)[None, :]
    # jenga._solver.set_links_pos(init_pos, jenga._get_ls_idx(None), envs_idx=None)

    for _jenga in jenga:
        for i in range(len(_jenga.links)):
            q_idxs = _jenga.links[i].joint.q_idx_local
            cur_pos = _jenga.get_qpos(qs_idx_local=q_idxs)
            delta = torch.zeros_like(cur_pos)
            delta[:3] = torch.tensor([0.1, 0.1, 0.5], device=cur_pos.device)
            _jenga.set_qpos(cur_pos + delta, qs_idx_local=q_idxs)

    # # quat = gu.ti_xyz_to_quat(xyz)
    # for i in range(len(jenga.links)):
    #     q_idxs = jenga.links[i].joint.q_idx_local[:3]
    #     cur_pos = jenga.get_qpos(qs_idx_local=q_idxs)
    #     delta = torch.tensor([0.5, 0.5, 0.5], device=cur_pos.device)
    #     target = cur_pos + delta
    #     jenga.set_dofs_position(target, dofs_idx_local=q_idxs)
    #     print(i, jenga.get_qpos(qs_idx_local=q_idxs))

    # for i in range(len(jenga.links)):
    #     jenga.links[0].set_mass(4)

    from pathlib import Path
    video_path = Path('video_test.mp4')
    
    cam.start_recording()
    for i in range(120):
        scene.step()
        rgb_arr, _, _, _ = cam.render()

    if video_path.exists():
        video_path.unlink()
    cam.stop_recording(save_to_filename=f'video_test.mp4', fps=1 / dt)


if __name__ == "__main__":
    main()
