import argparse
import os
import pickle

import torch
from examples.manipulation.franka_grasp_env_v7 import FrankaGraspEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="franka-grasp-v4")
    parser.add_argument("--ckpt", type=int, default=10)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = os.path.join("logs", args.exp_name)
    cfgs = pickle.load(open(os.path.join(log_dir, "cfgs.pkl"), "rb"))
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfgs

    print(f"Experiment name: {args.exp_name}")
    visible_devices = torch.cuda.device_count()
    print(f"Number of visible CUDA devices: {visible_devices}")
    for i in range(visible_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    selected_device = torch.cuda.current_device()
    udid = torch.cuda.get_device_properties(selected_device).uuid
    print(f"Selected device UDID: {udid}")
    device = torch.device(f"cuda:{selected_device}")

    env = FrankaGraspEnv(
        num_envs=5,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        train_cfg=train_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        device=device,
        save_video=False,
        add_camera=False
    )

    # If a camera was set up in the environment, start recording.
    if hasattr(env, "cam"):
        env.cam.start_recording()

    checkpoint_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
    if checkpoint_files:
        latest_ckpt = max(checkpoint_files, key=lambda f: int(f.split('_')[1].split('.')[0]))
        print(f"Using the most recent checkpoint: {latest_ckpt}")
        resume_path = os.path.join(log_dir, latest_ckpt)
    else:
        raise FileNotFoundError("No checkpoint files found in the log directory.")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)
    
    for _ in range(1):
        idx = 0
        obs = env.get_observations()
        all_rewards = None
        with torch.no_grad():
            while True:
                if hasattr(env, "cam"):
                    env.cam.render()
                
                # actions = runner.alg.act(obs, obs)
                actions = policy(obs)
                obs, _, rewards, dones, infos = env.step(actions)
                if all_rewards is None:
                    all_rewards = rewards
                else:
                    all_rewards += rewards
                print(infos['is_grasped'])
                idx += 1
                if idx > env_cfg['episode_length_s'] / env_cfg['dt']:
                    break

        print(all_rewards)
        env.reset()

    if hasattr(env, "cam"):
        env.cam.stop_recording(save_to_filename=f'video_{args.exp_name}.mp4', fps=1 / env_cfg['dt'])


if __name__ == "__main__":
    main()
