import argparse
import numpy as np
import genesis as gs
import torch

gs.init(backend=gs.gpu)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        substeps=100
    ),
    show_viewer=False
)
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    morph=gs.morphs.Box(
        size=(0.1, 0.1, 0.1),
        pos=(0.0, 1.0, 0.5),
        euler=(0, 0, 0),
    ),
)

cam = scene.add_camera(
    res    = (1280, 960),
    pos    = (3.5, 0.0, 2.0),
    lookat = (0, 0, 0.5),
    fov    = 50,
    GUI    = False
)

scene.build()
cam.start_recording()
for i in range(120):
    rgb_arr, _, _, _ = cam.render()
    scene.step()
    print(scene.cur_t, scene.gravity)

cam.stop_recording(save_to_filename=f'video_test.mp4', fps=25)