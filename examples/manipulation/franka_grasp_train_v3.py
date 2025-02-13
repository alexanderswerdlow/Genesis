import argparse
import os
import pickle
import shutil

from franka_grasp_env_v3 import FrankaGraspEnv
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
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "init_noise_std": 0.5,
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
            "num_steps_per_env": 64,
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
            "joint2": -0.5,
            "joint3": 0.0,
            "joint4": -1.5,
            "joint5": 0.0,
            "joint6": 1.0,
            "joint7": 0.7,
            "finger_joint1": 0.03,
            "finger_joint2": 0.03,
        },
        "kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
        "kd": [450, 450, 350, 350, 200, 200, 200, 10, 10],
        "force_lower": [-87, -87, -87, -87, -12, -12, -12, -20, -20],
        "force_upper": [87, 87, 87, 87, 12, 12, 12, 20, 20],
        "clip_actions": 1.0,
        "action_scale": 0.2,
        "dt": 0.005,
        "substeps": 15,
        "episode_length_s": 5.0,
        "cube_size": 0.04,
        "cube_init_x": 0.65,
        "cube_init_z": 0.02,
        "cube_spawn_range_x": [0.60, 0.70],
        "cube_spawn_range_y": [-0.05, 0.05],
    }
    obs_cfg = {
        # Example: 9 dof pos, 9 dof vel, 3 ee pos, 3 cube pos, 9 last action => total 33
        "num_obs": 33,
        "obs_scales": {
            "dof_vel": 1.0,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "distance_to_cube": 1.0,
            "lift_cube": 2.0,
            "action_rate": -0.01,
        },
    }
    command_cfg = None  # Not used for a simple grasp

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="franka-grasp-v3")
    parser.add_argument("-B", "--num_envs", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=1000)
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
