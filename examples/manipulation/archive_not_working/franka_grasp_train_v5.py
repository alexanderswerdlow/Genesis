import argparse
import os
import pickle
import shutil

from franka_grasp_env_v5 import FrankaGraspEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

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
            "activation": "elu",
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
            "num_steps_per_env": 24,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "save_interval": 50,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }
    return train_cfg_dict

def get_cfgs():
    # Example environment config for Franka
    env_cfg = {
        "franka_mjcf_path": "xml/franka_emika_panda/panda.xml",   # or URDF
        "num_actions": 9,  # 7 arm + 2 fingers
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
        "clip_actions": 1000.0,
        "action_scale": 1.0,
        "dt": 0.02,
        "substeps": 4,
        "episode_length_s": 3,
        "cube_size": 0.05,
        "cube_init_x": 0.65,
        "cube_init_y": 0.65,
        "cube_init_z": 0.02,
        "cube_spawn_range_x": [0.60, 0.60],
        "cube_spawn_range_y": [0.0, 0.0],
    }
    obs_cfg = {
        # 9 dof pos, 9 dof vel, 2 dof force, 3 cube pos, 4 cube quat,
        # 3 finger_joint1 pos, 3 finger_joint2 pos, 9 last action, 1 is_grasped flag => total 43
        "num_obs": 43,
        "obs_scales": {
            "dof_vel": 1.0,
            "dof_force": 1.0,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "gripper_box": 4.0,
            "box_target": 8.0,
            "no_floor_collision": 0.25,
            "robot_target_qpos": 0.03,
            "is_grasped": 8.0,
        },
    }
    command_cfg = None  # Not used for a simple grasp

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="franka-grasp-v4")
    parser.add_argument("-B", "--num_envs", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--save_video", action="store_true", help="Save video of the environment")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = FrankaGraspEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        device="cuda:0",
        save_video=args.save_video,
    )

    # Create runner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    # Save config
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Start training
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)

if __name__ == "__main__":
    main()
