import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class FrankaEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False, device="cuda"):

        self.num_privileged_obs = None
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        
        self.dt = 0.005
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg

        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=15),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=30,
                max_FPS=60,
            ),
            show_viewer=show_viewer,
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=True,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(0.55, -0.1, -0.05),
                upper_bound=(0.75, 0.1, 0.3),
                grid_density=128,
            ),
        )

        # Add plane
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        
        # Add cube
        self.cube_size = (0.04, 0.04, 0.04)
        self.cube = self.scene.add_entity(
            # material=gs.materials.MPM.Elastic(),
            morph=gs.morphs.Box(
                size=self.cube_size,
                pos=(0.65, 0.0, 0.025),
                euler=(0, 0, 0),
            ),
            # surface=gs.surfaces.Default(
            #     vis_mode="particle",
            # ),
        )

        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
            material=gs.materials.Rigid(coup_friction=1.0),
        )

        self.cam = self.scene.add_camera(
            res    = (1280, 960),
            pos    = (3.5, 0.0, 2.5),
            lookat = (0, 0, 0.5),
            fov    = 30,
            GUI    = False
        )

        self.scene.build(n_envs=num_envs)

        # Joint setup
        self.motor_dofs = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        
        # PD control parameters
        self.franka.set_dofs_kp(torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], device=device))
        self.franka.set_dofs_kv(torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10], device=device))

        # Buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=device)
        self.rew_buf = torch.zeros((self.num_envs,), device=device)
        self.reset_buf = torch.ones((self.num_envs,), device=device)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=device)
        
        # End effector reference
        self.end_effector = self.franka.get_link("hand")
        self.default_gripper_pos = torch.tensor([0.03, 0.03], device=device)
        
        # Initialize targets
        self.cube_target_height = 0.3
        self.reset()

    def step(self, actions):
        self.actions = torch.clip(actions, -1, 1)
        
        # Control arm (first 7 dimensions) and gripper (last 2 dimensions)
        arm_actions = self.actions[:, :7] * self.env_cfg["arm_action_scale"]
        gripper_actions = self.actions[:, 7:] * self.env_cfg["gripper_action_scale"]
        
        # TODO
        target_arm_pos = arm_actions + torch.tensor(self.env_cfg["default_arm_pos"], device=self.device)[None]
        target_gripper_pos = gripper_actions + self.default_gripper_pos[None]
        
        self.franka.control_dofs_position(target_arm_pos, self.motor_dofs)
        self.franka.control_dofs_position(target_gripper_pos, self.fingers_dof)
        
        self.scene.step()

        # Update buffers
        self.episode_length_buf += 1
        self._update_observations()
        
        # Check termination
        dist = torch.norm(self.end_effector.get_pos() - self.cube.get_pos(), dim=-1)
        grasp_success = dist < 0.02
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | grasp_success
        print(f"self.reset_buf: {self.reset_buf.shape}, {self.reset_buf.float().mean()}, {self.reset_buf.float().max()}, {self.reset_buf.float().min()}")
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Compute reward
        self.rew_buf[:] = self._compute_reward()
        print(f"self.rew_buf: {self.rew_buf.shape}, {self.rew_buf.mean()}, {self.rew_buf.max()}, {self.rew_buf.min()}")
        return self.obs_buf, None, self.rew_buf, self.reset_buf, {}

    def _update_observations(self):
        # End effector state
        ee_pos = self.end_effector.get_pos()
        ee_vel = self.end_effector.get_vel()
        
        # Cube state
        cube_pos = self.cube.get_pos()
        cube_vel = self.cube.get_vel()
        
        # Joint states
        joint_pos = self.franka.get_dofs_position(self.motor_dofs)
        joint_vel = self.franka.get_dofs_velocity(self.motor_dofs)
        
        # Gripper state
        gripper_pos = self.franka.get_dofs_position(self.fingers_dof)
        gripper_vel = self.franka.get_dofs_velocity(self.fingers_dof)
        
        self.obs_buf = torch.cat([
            ee_pos * self.obs_cfg["obs_scales"]["position"],
            ee_vel * self.obs_cfg["obs_scales"]["velocity"],
            cube_pos * self.obs_cfg["obs_scales"]["position"],
            cube_vel * self.obs_cfg["obs_scales"]["velocity"],
            joint_pos * self.obs_cfg["obs_scales"]["joint_pos"],
            joint_vel * self.obs_cfg["obs_scales"]["joint_vel"],
            gripper_pos * self.obs_cfg["obs_scales"]["gripper_pos"],
            gripper_vel * self.obs_cfg["obs_scales"]["gripper_vel"],
            self.actions
        ], dim=-1)

    def _compute_reward(self):
        # Reward components
        ee_pos = self.end_effector.get_pos()
        cube_pos = self.cube.get_pos()
        
        # Distance reward
        dist = torch.norm(ee_pos - cube_pos, dim=-1)
        dist_reward = torch.exp(-dist / self.reward_cfg["dist_sigma"])
        
        # Grasp reward
        gripper_width = self.franka.get_dofs_position(self.fingers_dof).sum(dim=-1)
        cube_height = cube_pos[:, 2]
        lifting_reward = torch.clamp((cube_height - self.cube_size[2]/2) / self.cube_target_height, 0, 1)
        
        # Combine rewards
        reward = (
            self.reward_cfg["scales"]["dist"] * dist_reward +
            self.reward_cfg["scales"]["lift"] * lifting_reward +
            self.reward_cfg["scales"]["grasp"] * (gripper_width < 0.01) +
            self.reward_cfg["scales"]["action"] * torch.mean(torch.square(self.actions), dim=-1)
        )
        
        return reward

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Reset arm
        default_arm_pos = torch.tensor(
            self.env_cfg["default_arm_pos"], 
            device=self.device
        ).repeat(len(envs_idx), 1)
        
        self.franka.set_dofs_position(
            default_arm_pos,
            self.motor_dofs,
            envs_idx=envs_idx
        )
        
        # Reset gripper
        self.franka.set_dofs_position(
            self.default_gripper_pos,
            self.fingers_dof,
            envs_idx=envs_idx
        )
        
        # Reset cube
        cube_pos = torch.zeros((len(envs_idx), 3), device=self.device)
        cube_pos[:, 0] = gs_rand_float(0.6, 0.7, (len(envs_idx),), self.device)
        cube_pos[:, 1] = gs_rand_float(-0.1, 0.1, (len(envs_idx),), self.device)
        cube_pos[:, 2] = self.cube_size[2]/2  # Place on ground
        
        self.cube.set_pos(cube_pos, envs_idx=envs_idx)
        # self.cube.set_vel(torch.zeros_like(cube_pos), envs_idx=envs_idx)
        
        # Reset buffers
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = False

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return None
