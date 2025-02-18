import argparse
import os
import pickle
import shutil

from franka_grasp_env_v8 import FrankaGraspEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
import torch
import json

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 0.5,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "policy_class_name": "ActorCritic",
            "runner_class_name": "OnPolicyRunner",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 100,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "save_interval": 100,
            "eval_interval": 10,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }
    return train_cfg_dict

def get_cfgs():
    env_cfg = {
        "franka_mjcf_path": "xml/franka_emika_panda/panda.xml",   # or URDF
        "num_actions": 8,  # 7 arm + 1 finger split
        "dof_names": [
            "joint1", "joint2", "joint3", "joint4", "joint5", 
            "joint6", "joint7", "finger_joint1", "finger_joint2"
        ],
        "default_joint_angles": {
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
            "finger_joint1": 0.0,
            "finger_joint2": 0.0,
        },
        "kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
        "kd": [450, 450, 350, 350, 200, 200, 200, 10, 10],
        "force_lower": [-87, -87, -87, -87, -12, -12, -12, -100, -100],
        "force_upper": [87, 87, 87, 87, 12, 12, 12, 100, 100],
        "clip_actions": 1.0,
        "clip_actions_fingers": 1.0,
        "action_scale": 0.12,
        "action_scale_fingers": 0.04,
        "dt": 0.02,
        "substeps": 4,
        "episode_length_s": 4,
        "cube_size": 0.05,
        "cube_init_x": 0.65,
        "cube_init_y": 0.65,
        "cube_init_z": 0.015,
        "cube_spawn_range_x": [0.60, 0.70],
        "cube_spawn_range_y": [-0.1, 0.1],
        "target_offset_range_x": [-0.1, 0.1],
        "target_offset_range_y": [-0.1, 0.1],
        "target_offset_range_z": [0.2, 0.6],
        "use_jenga": False,
    }
    obs_cfg = {
        # 9 dof pos, 9 dof vel, 2 dof force, 3 cube pos, 4 cube quat,
        # 3 finger_joint1 pos, 3 finger_joint2 pos, 8 last action, 1 is_grasped flag => total 46
        "num_obs": 45,
        "obs_scales": {
            "dof_vel": 1.0,
            "dof_force": 1.0,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "gripper_box": 4.0,
            "box_target": 8.0,
            "robot_target_qpos": 0.3,
            "no_floor_collision": 0.25,
            "is_target_reached": 2.0,
            "is_grasped": 2.0,
        },
    }
    command_cfg = None
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="franka-grasp-v4")
    parser.add_argument("-B", "--num_envs", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--save_video", action="store_true", help="Save video of the environment")
    parser.add_argument("--use_jenga", action="store_true", help="Use Jenga tower reward")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        args.exp_name += f"_{os.urandom(4).hex()}"
        log_dir = f"logs/{args.exp_name}"

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Experiment name: {args.exp_name}")
    visible_devices = torch.cuda.device_count()
    print(f"Number of visible CUDA devices: {visible_devices}")
    for i in range(visible_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    selected_device = torch.cuda.current_device()
    udid = torch.cuda.get_device_properties(selected_device).uuid
    print(f"Selected device UDID: {udid}")
    device = torch.device(f"cuda:{selected_device}")

    if args.use_jenga:
        reward_cfg["reward_scales"]["jenga_tower_pos"] = 1.0
        reward_cfg["reward_scales"]["jenga_tower_vel"] = 1.0
        env_cfg["use_jenga"] = True
        env_cfg["dt"] = 0.02
        env_cfg["substeps"] = 32
        env_cfg['action_scale'] /= (0.04 / env_cfg["dt"])
        env_cfg['action_scale_fingers'] /= (0.04 / env_cfg["dt"])

    # Create environment
    env = FrankaGraspEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        train_cfg=train_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        device=device,
        save_video=False,
    )

    eval_env = FrankaGraspEnv(
        num_envs=1 if args.save_video else args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        train_cfg=train_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        device="cuda",
        save_video=False,
        add_camera=args.save_video,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device, eval_env=eval_env)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    with open(f"{log_dir}/cfgs.json", "w") as f:
        json.dump({
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "train_cfg": train_cfg
        }, f, indent=4)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)

    print(f"Finished training")
    print(f"Experiment name: {args.exp_name}")

if __name__ == "__main__":
    main()
