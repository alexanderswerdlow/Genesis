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


def smooth_tanh(x, lam=0.2098612289):
    return torch.tanh(lam * x)

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
        train_cfg=None,
        command_cfg=None,  # Not really used for a simple grasp task
        show_viewer=False,
        device="cuda",
        save_video=False,
        add_camera=False,
    ):
        self.device = device
        self.train_cfg = train_cfg

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

        scene_kwargs = {}
        if add_camera or save_video:
            scene_kwargs["vis_options"] = gs.options.VisOptions(n_rendered_envs=1)
            scene_kwargs["viewer_options"] = gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=30,
                max_FPS=int(1.0 / self.dt),
            )

        if env_cfg["use_jenga"]:
            scene_kwargs["rigid_options"] = gs.options.RigidOptions(
                dt=self.dt / 4,
                constraint_solver=gs.constraint_solver.Newton,
                # use_contact_island=True,
                # use_hibernation=True,
            )

        # Create the scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.env_cfg["substeps"]),
            show_viewer=show_viewer,
            **scene_kwargs,
        )

        # Add a plane (the table/floor)
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.use_jenga = env_cfg["use_jenga"]
        if self.use_jenga:
            # self.jenga = self.scene.add_entity(
            #     gs.morphs.MJCF(file='examples/manipulation/jenga.xml'),
            # )
            # self.cube_handles = self.jenga.links[-2]

            num_layers = 3
            scale_factor = 1.0
            base_pos = (0.0, 0.0, 0.05)
            block_height = (0.04 * scale_factor) + 5e-3
            delta_x = (0.05 * scale_factor) + 5e-3

            self.jenga = []
            for layer in range(num_layers):
                euler = (0, 0, (layer % 2) * 90)
                z_pos = base_pos[2] + layer * block_height        
                offsets = [
                    (0, -delta_x, 0),  # Left block
                    (0, 0, 0),      # Center block
                    (0, delta_x, 0)    # Right block
                ] if layer % 2 == 0 else [
                    (-delta_x, 0, 0),  # Left block
                    (0, 0, 0),      # Center block
                    (delta_x, 0, 0)    # Right block
                ]
                
                for dx, dy, dz in offsets:
                    self.jenga.append(self.scene.add_entity(
                        gs.morphs.URDF(
                            file="examples/manipulation/assets/jenga.urdf",
                            pos=(
                                base_pos[0] + dx,
                                base_pos[1] + dy,
                                z_pos
                            ),
                            euler=euler,
                            scale=scale_factor,
                            fixed=False,
                        )
                    ))
            self.cube_handles = self.jenga[-1]
            self.jenga_base = self.jenga[:-1]
        else:
            self.cube_size = env_cfg["cube_size"]
            self.cube_init_z = env_cfg["cube_init_z"]  # e.g., on table
            self.cube_handles = self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(self.cube_size, self.cube_size, self.cube_size),
                    pos=(env_cfg["cube_init_x"], env_cfg["cube_init_y"], self.cube_init_z),
                ),
            )
        
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file=env_cfg["franka_mjcf_path"]),
        )
        
        self.save_video = save_video
        self.add_camera = add_camera
        if self.save_video or self.add_camera:
            self.cam = self.scene.add_camera(
                res    = (1280, 960),
                pos    = (3.5, 0.0, 2.5),
                lookat = (0, 0, 0.5),
                fov    = 30,
                GUI    = False
            )

        # Build the scene
        self.scene.build(n_envs=num_envs)

        if self.use_jenga:
            self.num_obs += len(self.jenga) * 7
            assert sum(_jenga.get_qpos().shape[-1] for _jenga in self.jenga) == len(self.jenga) * 7

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
                self.episode_sums[f"sc_{name}"] = torch.zeros(
                    (self.num_envs,), device=self.device, dtype=gs.tc_float
                )

        self.episode_sums["met_is_target_reached"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.episode_sums["met_is_grasped"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.episode_sums["met_total_reward"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        self.obs_buf = torch.zeros((num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_int)

        self.default_dof_pos = torch.tensor(
            [
                env_cfg["default_joint_angles"][name]
                for name in env_cfg["dof_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.num_dof = len(self.default_dof_pos)

        self.actions = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.dof_pos = torch.zeros((num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.dof_force = torch.zeros((num_envs, len(self.finger_dofs)), device=self.device, dtype=gs.tc_float)

        self.cube_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.cube_quat = torch.zeros((num_envs, 4), device=self.device, dtype=gs.tc_float)

        self.finger_joint1_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.finger_joint2_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.finger_joint1_handle = self.franka.get_joint("finger_joint1")
        self.finger_joint2_handle = self.franka.get_joint("finger_joint2")

        self.extras = {}

        self.save_video = save_video
        if self.save_video:
            self.cam.start_recording()

        self.reached_cube = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.reached_target = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        self.franka_arm_links = torch.tensor([
            self.franka.get_joint("finger_joint1").link.idx,
            self.franka.get_joint("finger_joint2").link.idx,
        ], device=self.device, dtype=gs.tc_int)
        
        self.num_steps = 0
        self.iter = 0

        self.clamp_arr = torch.tensor([
            [-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897,  0.   ],
            [ 2.897,  1.763,  2.897, -0.07 ,  2.897,  3.752,  2.897,  0.08 ]
        ], device=self.device, dtype=gs.tc_float)

        if self.use_jenga:
            if self.add_camera:
                self.cam.start_recording()

            jenga = self.jenga
            for _jenga in jenga:
                for i in range(len(_jenga.links)):
                    q_idxs = _jenga.links[i].joint.q_idx_local
                    cur_pos = _jenga.get_qpos(qs_idx_local=q_idxs)
                    delta = torch.zeros_like(cur_pos)
                    delta[..., :3] = torch.tensor([0.4, 0.4, 0.0], device=cur_pos.device)
                    _jenga.set_qpos(cur_pos + delta, qs_idx_local=q_idxs)

            self.pre_init_jenga_pos = torch.stack([
                _jenga.get_qpos()
                for _jenga in self.jenga
            ], dim=1)

            settle_steps = self.env_cfg.get("settle_steps", 50)
            for _ in range(settle_steps):
                self.scene.step()
                if self.add_camera:
                    self.cam.render()

            if self.add_camera:
                self.cam.stop_recording(save_to_filename=f'video_jenga_v0.mp4', fps=1 / self.dt)

            self.init_jenga_pos = torch.stack([
                _jenga.get_qpos()
                for _jenga in self.jenga
            ], dim=1)
            self.init_jenga_vel = torch.stack([
                _jenga.get_dofs_velocity()
                for _jenga in self.jenga
            ], dim=1)

            for i in range(len(self.jenga)):
                assert len(self.jenga[i].links) == 1

        self.reset()

    def step(self, actions):
        self.num_steps += 1

        actions = torch.tanh(actions)
        delta = actions
        delta[:, :-1] *= self.env_cfg["action_scale"]
        delta[:, -1:] *= self.env_cfg["action_scale_fingers"]
        ctrl = self.actions + delta
        ctrl = torch.clamp(ctrl, min=self.clamp_arr[None, 0, :], max=self.clamp_arr[None, 1, :])
        self.actions[:] = ctrl
        final_act = torch.cat([self.actions, torch.zeros_like(self.actions[:, -1:])], dim=-1)
        final_act[:, -2:] = (self.actions[:, -1] / 2.0)[:, None]
        self.franka.control_dofs_position(final_act, self.motor_dofs)
        
        self.scene.step()
        self.episode_length_buf += 1

        self.dof_pos[:] = self.franka.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.franka.get_dofs_velocity(self.motor_dofs)
        self.dof_force[:] = self.franka.get_dofs_force(self.finger_dofs)

        # Get finger and cube states
        self.cube_pos[:] = self.cube_handles.get_pos()
        self.cube_quat[:] = self.cube_handles.get_quat()

        self.finger_joint1_pos[:] = self.finger_joint1_handle.get_pos()
        self.finger_joint2_pos[:] = self.finger_joint2_handle.get_pos()

        finger1_distance = torch.norm(self.finger_joint1_pos - self.cube_pos, dim=-1)
        finger2_distance = torch.norm(self.finger_joint2_pos - self.cube_pos, dim=-1)
        self.reached_cube = torch.maximum(
            self.reached_cube,
            torch.minimum(
                (finger1_distance < 0.0185).float(),
                (finger2_distance < 0.0185).float()
            )
        )

        target_distance = torch.norm(self.cube_pos - self.target_pos, dim=-1)
        self.reached_target = ((self.reached_cube > 0.5) * (target_distance < 0.02)).float()

        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn()
            start_ep, end_ep = 200 * self.train_cfg['runner']['num_steps_per_env'], 1000 * self.train_cfg['runner']['num_steps_per_env']
            if (name == "robot_target_qpos" or name == "no_floor_collision"):
                if self.num_steps > start_ep:
                    _scale = self.reward_scales[name] * min(1, max(0, (self.num_steps - start_ep) / (end_ep - start_ep)))
                else:
                    _scale = 0
            else:
                _scale = self.reward_scales[name]
            self.rew_buf += rew * _scale
            self.episode_sums[name] += rew
            self.episode_sums[f"sc_{name}"] += rew * _scale

        self.episode_sums["met_is_target_reached"] = torch.max(self.episode_sums["met_is_target_reached"], self.reached_target.float())
        self.episode_sums["met_is_grasped"] = torch.max(self.episode_sums["met_is_grasped"], self.reached_cube.float())
        self.episode_sums["met_total_reward"] += self.rew_buf

        additional_obs = []
        if self.use_jenga:
            additional_obs = [
                torch.cat([_jenga.get_qpos() for _jenga in self.jenga], dim=-1)
            ]

        self.obs_buf = torch.cat(
            [
                self.dof_pos,                               # 9 dof positions
                self.dof_vel * self.obs_scales["dof_vel"],  # 9 dof velocities
                self.dof_force * self.obs_scales["dof_force"],  # 2 dof forces
                self.cube_pos,                              # 3 cube position
                self.cube_quat,                             # 4 cube orientation
                self.finger_joint1_pos,                     # 3 finger_joint1 position (NEW)
                self.finger_joint2_pos,                     # 3 finger_joint2 position (NEW)
                *additional_obs,
                self.actions,                               # 9 last commanded actions
                (self.reached_cube > 0.5).float().unsqueeze(-1),  # 1 is_grasped flag
                self.target_pos,                            # 3 target position for conditioning
            ],
            dim=-1,
        )

        is_nan = (torch.isnan(self.obs_buf).any()) | (torch.isnan(self.actions).any()) | (torch.isnan(self.rew_buf).any())
        self.rew_buf = torch.nan_to_num(self.rew_buf, nan=0.0)
        self.reset_buf = (self.episode_length_buf >= self.max_episode_length) | is_nan
        reset_env_ids = (self.reset_buf > 0).nonzero(as_tuple=False).flatten()
        self.reset_idx(reset_env_ids)

        if self.save_video:
            if self.episode_length_buf.max() % 100 == 0:
                self.cam.render()
                if self.episode_length_buf.max() % (self.max_episode_length // 10) == 0:
                    import datetime
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
                    self.cam.stop_recording(save_to_filename=f'video_{now_str}.mp4', fps=60)

        self.extras['is_grasped'] = self.reached_cube

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.reached_cube[:] = 0.0
        return self.obs_buf, None

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self.obs_buf[env_ids] = 0.0
        self.actions[env_ids, :-1] = self.default_dof_pos[:-2]
        self.actions[env_ids, -1] = (self.default_dof_pos[-1] + self.default_dof_pos[-2])
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        self.franka.set_dofs_position(
            position=self.dof_pos[env_ids],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=env_ids,
        )

        if self.use_jenga:
            for i, _jenga in enumerate(self.jenga):
                init_pos = self.pre_init_jenga_pos[env_ids, i, :]
                _jenga.set_qpos(init_pos, envs_idx=env_ids)

            cube_pos_reset = self.cube_handles.get_pos()
        else:
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

        target_offset_range_x = self.env_cfg.get("target_offset_range_x")
        target_offset_range_y = self.env_cfg.get("target_offset_range_y")
        target_offset_range_z = self.env_cfg.get("target_offset_range_z")
        target_offset_x = gs_rand_float(target_offset_range_x[0], target_offset_range_x[1], (len(env_ids),), self.device)
        target_offset_y = gs_rand_float(target_offset_range_y[0], target_offset_range_y[1], (len(env_ids),), self.device)
        target_offset_z = gs_rand_float(target_offset_range_z[0], target_offset_range_z[1], (len(env_ids),), self.device)
        target_offsets = torch.stack((target_offset_x, target_offset_y, target_offset_z), dim=-1)
        new_target_pos = cube_pos_reset + target_offsets
        self.target_pos[env_ids] = new_target_pos

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        self.extras["episode"] = {}
        for key in self.episode_sums:
            if key in self.reward_scales:
                self.extras["episode"]["rew_" + key] = torch.mean(
                    self.episode_sums[key][env_ids]
                ).item()
            else:
                self.extras["episode"][f"{key}"] = torch.mean(
                    self.episode_sums[key][env_ids]
                ).item()
            self.episode_sums[key][env_ids] = 0.0

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None
    
    def _reward_components(self):
        """
        Computes a composite reward with the following components:
          - box_target: measures how close the cube is to the target position/orientation,
                        but is gated by whether the cube has been 'reached' by the gripper.
          - gripper_box: rewards the gripper for being close to the cube.
          - robot_target_qpos: rewards the robot for keeping close to its default joint configuration.
          - no_floor_collision: penalizes the cube dropping too low (i.e. colliding with the floor).
        """    
        # Positional error between cube and target position
        pos_err = torch.norm(self.cube_pos - self.target_pos, dim=-1)
    
        # Rotation error: compare the cube's rotation vs. an identity rotation
        # target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 4)
        # target_mat = self.quat_to_rotmat(target_quat)
        # cube_mat = self.quat_to_rotmat(self.cube_quat)
        # target_mat_flat = target_mat.reshape(self.num_envs, 9)[:, :6]
        # cube_mat_flat = cube_mat.reshape(self.num_envs, 9)[:, :6]
        # rot_err = torch.norm(target_mat_flat - cube_mat_flat, dim=1)
    
        # Box target reward: combines pos_err and rot_err
        box_target = 1 - torch.tanh(5 * (1.0 * pos_err)) # + 0.1 * rot_err
        box_target = box_target * self.reached_cube
    
        # Gripper box reward: use the average position of the two finger joints as the gripper location
        gripper_pos = (self.finger_joint1_pos + self.finger_joint2_pos) / 2.0
        gripper_box = 1 - torch.tanh(5 * torch.norm(self.cube_pos - gripper_pos, dim=-1))
    
        # Robot target qpos reward: how close the robot joints are to default. ignore fingers.
        joint_err = torch.norm((self.dof_pos - self.default_dof_pos)[:, :-2], dim=-1)
        robot_target_qpos = 1 - torch.tanh(5 * joint_err)

        contacts = self.plane.get_contacts(with_entity=self.franka)
        contacts['valid_mask'] = contacts['valid_mask'].to(self.device)
        contacts['geom_b'] = contacts['geom_b'].to(self.device)
        is_collided = ((contacts['geom_b'][:, :, None] == self.franka_arm_links[None, None, :]).any(dim=-1) & contacts['valid_mask']).any(dim=-1)
        no_floor_collision = (1 - is_collided.float()).float()
        # no_floor_collision = torch.zeros_like(gripper_box)

        additional_rewards = {}
        if self.use_jenga:
            cur_block_pos = torch.stack([_jenga.get_qpos() for _jenga in self.jenga_base], dim=1)
            cur_block_vel = torch.stack([_jenga.get_dofs_velocity() for _jenga in self.jenga_base], dim=1)
            tower_pos_rew = 1 - torch.tanh(5 * torch.norm((cur_block_pos - self.init_jenga_pos[:, :-1, :]).reshape(self.init_jenga_pos.shape[0], -1), dim=-1))
            tower_vel_rew = 1 - torch.tanh(5 * torch.norm((cur_block_vel).reshape(self.init_jenga_pos.shape[0], -1), dim=-1))
            additional_rewards["jenga_tower_pos"] = tower_pos_rew
            additional_rewards["jenga_tower_vel"] = tower_vel_rew

        rewards = {
            "gripper_box": gripper_box,
            "box_target": box_target,
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
            "is_grasped": (self.reached_cube > 0.5).float(),
            "is_target_reached": (self.reached_target > 0.5).float(),
            **additional_rewards,
        }
        for k in rewards.keys():
            rewards[k] = torch.nan_to_num(rewards[k], nan=0.0)

        return rewards

    def _quaternion_distance(self, q1, q2):
        """Compute quaternion distance metric"""
        # Using dot product similarity
        dot_prod = torch.sum(q1 * q2, dim=-1)
        return 1 - dot_prod  # Range [0, 2]

    def _reward_box_target(self):
        return self._reward_components()["box_target"]

    def _reward_gripper_box(self):
        return self._reward_components()["gripper_box"]

    def _reward_robot_target_qpos(self):
        return self._reward_components()["robot_target_qpos"]

    def _reward_no_floor_collision(self):
        return self._reward_components()["no_floor_collision"]
    
    def _reward_is_grasped(self):
        return self._reward_components()["is_grasped"]

    def _reward_is_target_reached(self):
        return self._reward_components()["is_target_reached"]

    def _reward_jenga_tower_pos(self):
        return self._reward_components()["jenga_tower_pos"]

    def _reward_jenga_tower_vel(self):
        return self._reward_components()["jenga_tower_vel"]

    @staticmethod
    def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of quaternions (assumed to be in scalar-first [w,x,y,z] format)
        to rotation matrices.
        """
        w, x, y, z = q.unbind(dim=-1)
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - z * w)
        r02 = 2 * (x * z + y * w)
    
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - x * w)
    
        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x * x + y * y)
    
        rot_mat = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
        rot_mat = rot_mat.reshape(q.shape[0], 3, 3)
        return rot_mat