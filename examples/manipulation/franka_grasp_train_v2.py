import argparse
import os
import pickle
import shutil
from franka_grasp_env_v2 import FrankaEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 256, 256],
            "critic_hidden_dims": [256, 256, 256],
            "init_noise_std": 0.5,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 100,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        }
    }

def get_cfgs():
    env_cfg = {
        "num_actions": 9,  # 7 arm joints + 2 gripper joints
        "episode_length_s": 10.0,
        "arm_action_scale": 0.05,
        "gripper_action_scale": 0.01,
        "default_arm_pos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    }
    
    obs_cfg = {
        "num_obs": 3*2 + 3*2 + 7*2 + 2*2 + 9,  # ee(3) + cube(3) + joints(7) + gripper(2) + actions(9)
        "obs_scales": {
            "position": 2.0,
            "velocity": 0.1,
            "joint_pos": 1.0,
            "joint_vel": 0.05,
            "gripper_pos": 10.0,
            "gripper_vel": 0.1,
        }
    }
    
    reward_cfg = {
        "dist_sigma": 0.1,
        "scales": {
            "dist": 1.0,
            "lift": 2.0,
            "grasp": 0.5,
            "action": -0.001,
        }
    }
    
    return env_cfg, obs_cfg, reward_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="franka-grasp-v2")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=1000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = FrankaEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        device="cuda:0"
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    pickle.dump([env_cfg, obs_cfg, reward_cfg, train_cfg], open(f"{log_dir}/cfgs.pkl", "wb"))
    runner.learn(num_learning_iterations=args.max_iterations)
    

if __name__ == "__main__":
    main()
