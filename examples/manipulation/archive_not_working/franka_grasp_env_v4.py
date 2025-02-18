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
        save_video=False,
    ):
        self.device = torch.device(device)

        # Basic config
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None  # If you want privileged obs, set a size here
        self.num_actions = env_cfg["num_actions"]

        # A typical RL loop uses a dt of a few ms for manip. 
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
                pos=(env_cfg["cube_init_x"], env_cfg["cube_init_y"], self.cube_init_z),
            ),
            # surface=gs.surfaces.Default(vis_mode="particle"),
        )

        # Add the Franka. You can use URDF or MJCF:
        # ex: MJCF with "xml/franka_emika_panda/panda.xml", or URDF version
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file=env_cfg["franka_mjcf_path"]),
            material=gs.materials.Rigid(coup_friction=1.0),
            visualize_contact=True,
            vis_mode="collision",
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
        self.dof_force = torch.zeros((num_envs, len(self.finger_dofs)), device=self.device, dtype=gs.tc_float)

        self.cube_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.cube_quat = torch.zeros((num_envs, 4), device=self.device, dtype=gs.tc_float)

        self.finger_joint1_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.finger_joint2_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)

        # Default joint angles, e.g. an upright pose, fingers open
        self.default_dof_pos = torch.tensor(
            [
                env_cfg["default_joint_angles"][name]
                for name in env_cfg["dof_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )

        # NEW: Create handles for both finger joints
        self.finger_joint1_handle = self.franka.get_joint("finger_joint1")
        self.finger_joint2_handle = self.franka.get_joint("finger_joint2")

        self.extras = {}

        self.save_video = save_video
        if self.save_video:
            print("start recording")
            self.cam.start_recording()

        self.reached_cube = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def step(self, actions):
        # Clip, store, and convert actions
        self.actions[:] = torch.clamp(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        target_dof_pos = self.actions * self.env_cfg["action_scale"]
        self.franka.control_dofs_position(target_dof_pos[..., :-2] + self.default_dof_pos[:-2], self.arm_dofs)
        self.franka.control_dofs_force(target_dof_pos[..., -2:], self.finger_dofs)
        self.scene.step()

        # Housekeeping and counter updates
        self.episode_length_buf += 1

        # Retrieve states
        self.dof_pos[:] = self.franka.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.franka.get_dofs_velocity(self.motor_dofs)
        self.dof_force[:] = self.franka.get_dofs_force(self.finger_dofs)

        # Get finger and cube states
        self.cube_pos[:] = self.cube_handles.get_pos()
        self.cube_quat[:] = self.cube_handles.get_quat()

        # NEW: Retrieve finger joint positions for both finger_joint1 and finger_joint2
        self.finger_joint1_pos[:] = self.finger_joint1_handle.get_pos()
        self.finger_joint2_pos[:] = self.finger_joint2_handle.get_pos()

        # Update reached_cube state (using distance from the finger joint pos as before)
        finger1_distance = torch.norm(self.finger_joint1_pos - self.cube_pos, dim=-1)
        finger2_distance = torch.norm(self.finger_joint2_pos - self.cube_pos, dim=-1)
        print(f"finger1_distance: {finger1_distance.min()}, finger2_distance: {finger2_distance.min()}")
        self.reached_cube = torch.maximum(
            self.reached_cube,
            torch.minimum(
                (finger1_distance < 0.0405).float(),
                (finger2_distance < 0.0405).float()
            )
        )

        # Check termination and perform resets if needed
        self.reset_buf = (self.episode_length_buf >= self.max_episode_length)
        reset_env_ids = (self.reset_buf > 0).nonzero(as_tuple=False).flatten()
        self.reset_idx(reset_env_ids)

        # Compute rewards from all components
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            print(f"{name} rew: {rew.mean()}")
        print(f"rew_buf: {self.rew_buf.mean()}")
        print(f"reached_cube: {self.reached_cube.mean()}")

        # Build obs: append the new finger joint positions into the observation
        self.obs_buf = torch.cat(
            [
                self.dof_pos,                               # 9 dof positions
                self.dof_vel * self.obs_scales["dof_vel"],  # 9 dof velocities
                self.dof_force * self.obs_scales["dof_force"],  # 2 dof forces
                self.cube_pos,                              # 3 cube position
                self.cube_quat,                             # 4 cube orientation
                self.finger_joint1_pos,                     # 3 finger_joint1 position (NEW)
                self.finger_joint2_pos,                     # 3 finger_joint2 position (NEW)
                self.actions,                               # 9 last commanded actions
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]
        if self.save_video:
            if self.episode_length_buf.max() % 100 == 0:
                self.cam.render()
                if self.episode_length_buf.max() % (self.max_episode_length // 10) == 0:
                    import datetime
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
                    self.cam.stop_recording(save_to_filename=f'video_{now_str}.mp4', fps=60)

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.reached_cube[:] = 0.0
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

    def _reward_components(self):
        """Composite reward with multiple components"""
        target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # Identity rotation
        target_pos = torch.tensor(
            [self.env_cfg["cube_spawn_range_x"][0],
             self.env_cfg["cube_spawn_range_y"][0],
             0.5],
            device=self.device,
        )

        pos_err = torch.norm(self.cube_pos - target_pos, dim=-1)
        rot_err = self._quaternion_distance(self.cube_quat, target_quat)
        box_target_err = 0.9 * pos_err + 0.01 * rot_err
        box_target_err = torch.where(self.reached_cube > 0.5, -box_target_err, 0.0)
        
        # NEW: Instead of using the end-effector only, compute the average distance for both finger joints.
        dist1 = torch.norm(self.cube_pos - self.finger_joint1_pos, dim=-1)
        dist2 = torch.norm(self.cube_pos - self.finger_joint2_pos, dim=-1)
        gripper_box_err = -(dist1 + dist2) / 2.0

        # Joint alignment
        joint_err = torch.norm(self.dof_pos - self.default_dof_pos, dim=-1)
        robot_target_qpos = 1 - torch.tanh(5 * joint_err)
        robot_target_qpos = torch.where(self.reached_cube > 0.5, robot_target_qpos, 0.0)
        
        # Floor collision check
        floor_collision = (self.cube_pos[:, 2] < 0.05).float()
        no_floor_collision = 1 - floor_collision

        rewards = {
            "target_cube": box_target_err,
            "gripper_cube": gripper_box_err,
            "joint_alignment": robot_target_qpos,
            "collision_avoidance": no_floor_collision
        }
        return rewards

    def _quaternion_distance(self, q1, q2):
        """Compute quaternion distance metric"""
        # Using dot product similarity
        dot_prod = torch.sum(q1 * q2, dim=-1)
        return 1 - dot_prod  # Range [0, 2]

    def _reward_target_cube(self):
        return self._reward_components()["target_cube"]

    def _reward_gripper_cube(self):
        return self._reward_components()["gripper_cube"]

    def _reward_joint_alignment(self):
        return self._reward_components()["joint_alignment"]

    def _reward_collision_avoidance(self):
        return self._reward_components()["collision_avoidance"]
