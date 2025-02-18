import argparse
import os
import pickle

import torch
from examples.manipulation.franka_grasp_env_v3 import FrankaGraspEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="franka-grasp-v1")
    parser.add_argument("--ckpt", type=int, default=10)
    args = parser.parse_args()

    gs.init()

    log_dir = os.path.join("logs", args.exp_name)
    # Load configurations; note that for grasping tasks, the pickle contains:
    # [env_cfg, obs_cfg, reward_cfg, train_cfg]
    cfgs = pickle.load(open(os.path.join(log_dir, "cfgs.pkl"), "rb"))
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfgs

    # Remove reward scaling during evaluation (if undesired)
    # reward_cfg["reward_scales"] = {}

    # Create the Franka grasp environment with a single instance.
    # Note: command_cfg is not necessary for a manipulation task.
    env = FrankaGraspEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
    )

    # If a camera was set up in the environment, start recording.
    if hasattr(env, "cam"):
        env.cam.start_recording()

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    idx = 0
    with torch.no_grad():
        while True:
            if hasattr(env, "cam"):
                env.cam.render()
            actions = policy(obs)
            obs, _, _, dones, infos = env.step(actions)
            idx += 1
            if idx > 100:
                break

    if hasattr(env, "cam"):
        env.cam.stop_recording(save_to_filename=f'video_{args.exp_name}.mp4', fps=60)


if __name__ == "__main__":
    main()
