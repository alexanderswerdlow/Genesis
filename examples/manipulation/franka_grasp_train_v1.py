import argparse
import os
import pickle
import shutil

from examples.manipulation.franka_grasp_env_v1 import FrankaGraspEnv
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
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
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
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }
    return train_cfg_dict

def get_cfgs():
    # Environment configuration
    env_cfg = {
        "num_actions": 9,  # Updated: 7 arm joints + 2 finger joints
        "episode_length_s": 10.0,
        "dt": 0.005,
        "cube_init_pos": [0.65, 0.0, 0.025],
        "cube_size": [0.04, 0.04, 0.04],
        "cube_euler": [0, 0, 0],
        "robot_init_pos": [0.0, 0.0, 0.0],
        "robot_mjcf": "xml/franka_emika_panda/panda.xml",
        "dof_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ],
        "default_joint_angles": {
            "joint1": 0.0,
            "joint2": -0.7854,
            "joint3": 0.0,
            "joint4": -2.3562,
            "joint5": 0.0,
            "joint6": 1.5708,
            "joint7": 0.7854,
            "finger_joint1": 0.04,
            "finger_joint2": 0.04,
        },
        "kp": 1000.0,
        "kd": 100.0,
        "action_scale": 0.05,
        "clip_actions": 0.1,
        "camera_pos": [3, -1, 1.5],
        "camera_lookat": [0.0, 0.0, 0.0],
        "substeps": 15,
    }
    # Observation configuration:
    # 9 joint positions, 9 joint velocities, 3D end-effector pos, 3D cube pos, 3D relative pos => total 27
    obs_cfg = {
        "num_obs": 27,
        "obs_scales": {
            "joint_pos": 1.0,
            "joint_vel": 0.1,
            "ee_pos": 1.0,
            "cube_pos": 1.0,
            "rel_pos": 1.0,
        },
    }
    # Reward configuration:
    reward_cfg = {
        "reward_scales": {
            "grasp_distance": 1.0,
            "action_rate": 0.1,
        },
        "grasp_sigma": 0.005,
        "grasp_threshold": 0.02,
    }
    return env_cfg, obs_cfg, reward_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="franka-grasp-v1")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()

    gs.init(logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = FrankaGraspEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        device="cuda"
    )
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda")
    pickle.dump([env_cfg, obs_cfg, reward_cfg, train_cfg], open(f"{log_dir}/cfgs.pkl", "wb"))
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
