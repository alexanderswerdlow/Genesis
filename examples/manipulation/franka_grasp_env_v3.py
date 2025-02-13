import torch
import math
import genesis as gs
import numpy as np

from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class FrankaGraspEnv:
    """
    Adapted from the 'Go2Env' pattern but now controlling a Franka arm to grasp a cube.
    Observations, actions, and reward definitions differ from the quadruped version.
    """

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg=None,  # Not really used for a simple grasp task
        show_viewer=False,
        device="cuda",
    ):
        self.device = torch.device(device)

        # Basic config
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None  # If you want privileged obs, set a size here
        self.num_actions = env_cfg["num_actions"]

        # A typical RL loop uses a dt of a few ms for manip. 
        # Adjust as needed to match your desired control freq.
        self.dt = env_cfg["dt"]  
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg  # might not be used for a single-grasp scenario

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # Create the scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=env_cfg["substeps"]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=30,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            show_viewer=show_viewer,
        )

        # Add a plane (the table/floor)
        self.scene.add_entity(gs.morphs.Plane())

        # Add the cube. Here we use a small rigid or MPM-based box. 
        # Adjust size/material as you wish.
        self.cube_size = env_cfg["cube_size"]
        self.cube_init_z = env_cfg["cube_init_z"]  # e.g., on table
        self.cube_handles = self.scene.add_entity(
            # material=gs.materials.MPM.Elastic(),  # or Rigid if you prefer
            morph=gs.morphs.Box(
                size=(self.cube_size, self.cube_size, self.cube_size),
                pos=(env_cfg["cube_init_x"], 0.0, self.cube_init_z),
            ),
            # surface=gs.surfaces.Default(vis_mode="particle"),
        )

        # Add the Franka. You can use URDF or MJCF:
        # ex: MJCF with "xml/franka_emika_panda/panda.xml", or URDF version
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file=env_cfg["franka_mjcf_path"]),
            material=gs.materials.Rigid(coup_friction=1.0),
        )

        self.cam = self.scene.add_camera(
            res    = (1280, 960),
            pos    = (3.5, 0.0, 2.5),
            lookat = (0, 0, 0.5),
            fov    = 30,
            GUI    = False
        )

        # Build the scene
        self.scene.build(n_envs=num_envs)

        # Identify dofs for the Franka – typically 7 arm joints + 2 fingers
        self.arm_dofs = list(range(7))
        self.finger_dofs = list(range(7, 9))
        self.motor_dofs = self.arm_dofs + self.finger_dofs  # total 9 by default

        # Set PD gains, force limits, etc.
        self.franka.set_dofs_kp(np.array(env_cfg["kp"]))
        self.franka.set_dofs_kv(np.array(env_cfg["kd"]))
        self.franka.set_dofs_force_range(
            np.array(env_cfg["force_lower"]),
            np.array(env_cfg["force_upper"]),
        )

        # Prepare reward functions
        self.reward_functions = {}
        self.episode_sums = {}
        for name in self.reward_scales:
            self.reward_scales[name] *= self.dt  # scale by dt
            func = getattr(self, f"_reward_{name}", None)
            if func is not None:
                self.reward_functions[name] = func
                self.episode_sums[name] = torch.zeros(
                    (self.num_envs,), device=self.device, dtype=gs.tc_float
                )

        # Buffers
        self.obs_buf = torch.zeros((num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_int)

        self.actions = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)

        # We'll keep track of the cube and end-effector positions in world frame
        self.ee_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.cube_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)

        # Default joint angles, e.g. an upright pose, fingers open
        self.default_dof_pos = torch.tensor(
            [
                env_cfg["default_joint_angles"][name]
                for name in env_cfg["dof_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )

        # (Optional) find link index for end effector
        self._ee_link_handle = self.franka.get_link("hand")  # or "panda_hand", etc.

        self.extras = {}

    def step(self, actions):
        # Clip & store
        self.actions[:] = torch.clamp(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )

        # Convert actions to target joint positions around default, for example
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.franka.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # Housekeeping
        self.episode_length_buf += 1

        # Retrieve states
        self.dof_pos[:] = self.franka.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.franka.get_dofs_velocity(self.motor_dofs)

        # Get end-effector pos & cube pos
        # (If MPM, it's a bit different to get the centroid, but for simple rigid or small MPM, approximate.)
        self.ee_pos[:] = self._ee_link_handle.get_pos()[:, :3]
        # If using a single shape for the cube, get its center:
        self.cube_pos[:] = self.cube_handles.get_pos()

        # Check termination
        dist = torch.norm(self.ee_pos - self.cube_pos, dim=-1)
        grasp_success = dist < 0.02
        self.reset_buf = (self.episode_length_buf >= self.max_episode_length) | grasp_success

        # Do resets
        reset_env_ids = (self.reset_buf > 0).nonzero(as_tuple=False).flatten()
        self.reset_idx(reset_env_ids)

        # Compute rewards
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Build obs
        # For example, you might want: 7 arm joint positions, 2 finger positions,
        # joint velocities, end effector pos, cube pos, etc.
        # Adjust to your liking:
        self.obs_buf = torch.cat(
            [
                self.dof_pos,                               # 9
                self.dof_vel * self.obs_scales["dof_vel"],  # 9
                self.ee_pos,                                # 3
                self.cube_pos,                              # 3
                self.actions,                               # 9 (last commanded action)
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset Franka dofs
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        self.franka.set_dofs_position(
            position=self.dof_pos[env_ids],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=env_ids,
        )

        # Randomize cube location on the table, etc.
        # For instance, x in [0.60..0.70], y in [-0.05..0.05], z = table height + half cube
        x_vals = gs_rand_float(
            self.env_cfg["cube_spawn_range_x"][0],
            self.env_cfg["cube_spawn_range_x"][1],
            (len(env_ids),),
            self.device,
        )
        y_vals = gs_rand_float(
            self.env_cfg["cube_spawn_range_y"][0],
            self.env_cfg["cube_spawn_range_y"][1],
            (len(env_ids),),
            self.device,
        )
        z_vals = torch.ones_like(x_vals) * self.cube_init_z

        cube_pos_reset = torch.stack((x_vals, y_vals, z_vals), dim=-1)
        self.cube_handles.set_pos(cube_pos_reset, zero_velocity=True, envs_idx=env_ids)

        # Reset counters
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # Logging extras
        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = torch.mean(
                self.episode_sums[key][env_ids]
            ).item()
            self.episode_sums[key][env_ids] = 0.0

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    # ---------------- Reward functions below ---------------- #
    # Example shaping terms
    def _reward_distance_to_cube(self):
        # Negative distance between end effector and cube center
        dist = torch.norm(self.ee_pos - self.cube_pos, dim=-1)
        return -dist

    def _reward_lift_cube(self):
        # Encourage the cube’s center-of-mass to be above table
        # E.g. reward = cube_height_in_world - some baseline
        z = self.cube_pos[:, 2]
        return z  # or (z - 0.02) to offset the table height

    def _reward_action_rate(self):
        # Penalize rapid changes in actions
        return -torch.sum((self.last_actions - self.actions) ** 2, dim=-1)
