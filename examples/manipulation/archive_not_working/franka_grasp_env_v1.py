import torch
import math
import genesis as gs
import numpy as np

class FrankaGraspEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, device="cuda"):
        self.num_privileged_obs = None
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.dt = env_cfg.get("dt", 0.005)  # use smaller dt as in the Franka example
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.reward_scales = reward_cfg["reward_scales"]

        # Create scene with sim options adapted for robotic grasping
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=env_cfg.get("substeps", 15)),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / self.dt),
                camera_pos=env_cfg["camera_pos"],
                camera_lookat=env_cfg["camera_lookat"],
                camera_fov=env_cfg.get("camera_fov", 30),
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=env_cfg.get("show_viewer", False),
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add cube (grasp target)
        self.cube_init_pos = torch.tensor(env_cfg["cube_init_pos"], device=self.device, dtype=gs.tc_float)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["cube_size"],
                pos=self.cube_init_pos.cpu().numpy(),
                euler=env_cfg.get("cube_euler", (0, 0, 0)),
            ),
        )

        # add Franka robot arm via MJCF (using the provided franka XML file)
        self.robot_init_pos = torch.tensor(env_cfg["robot_init_pos"], device=self.device, dtype=gs.tc_float)
        self.robot = self.scene.add_entity(
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

        # obtain joint indices for the controlled arm joints
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in env_cfg["dof_names"]]
        self.num_dofs = len(self.motor_dofs)

        # PD control parameters for the arm joints
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_dofs, self.motor_dofs)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_dofs, self.motor_dofs)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=gs.tc_float)
        self.end_effector_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.cube_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        # default joint positions for the arm
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.extras = dict()

    def step(self, actions):
        # Clip and apply actions with a simple PD control
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions  # if simulating latency (like the locomotion env)
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # Update simulation buffers
        self.episode_length_buf += 1
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.last_actions[:] = self.actions[:]

        # get end-effector position (using the "hand" link)
        ee_pos_np = self.robot.get_link("hand").get_pos()
        self.end_effector_pos[:] = torch.tensor(ee_pos_np, device=self.device, dtype=gs.tc_float)
        # get cube (target) position
        cube_pos_np = self.cube.get_pos()
        self.cube_pos[:] = torch.tensor(cube_pos_np, device=self.device, dtype=gs.tc_float)

        # Form observation vector: concatenate joint errors, velocities, ee pos, cube pos, and their difference
        joint_pos_diff = (self.dof_pos - self.default_dof_pos) * self.obs_cfg["obs_scales"]["joint_pos"]
        joint_vel_scaled = self.dof_vel * self.obs_cfg["obs_scales"]["joint_vel"]
        ee_pos_scaled = self.end_effector_pos * self.obs_cfg["obs_scales"]["ee_pos"]
        cube_pos_scaled = self.cube_pos * self.obs_cfg["obs_scales"]["cube_pos"]
        rel_pos = (self.end_effector_pos - self.cube_pos) * self.obs_cfg["obs_scales"]["rel_pos"]
        self.obs_buf = torch.cat([joint_pos_diff, joint_vel_scaled, ee_pos_scaled, cube_pos_scaled, rel_pos], dim=-1)

        # Reward is based on the distance between end-effector and cube (with an optional action smoothness penalty)
        dist = torch.norm(self.end_effector_pos - self.cube_pos, dim=1)
        grasp_reward = torch.exp(- (dist ** 2) / self.reward_cfg["grasp_sigma"])
        action_rate_penalty = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        reward = self.reward_scales["grasp_distance"] * grasp_reward + self.reward_scales["action_rate"] * (-action_rate_penalty)
        self.rew_buf[:] = reward

        # Terminate if grasp is successful (distance below threshold) or max episode length reached
        grasp_success = dist < self.reward_cfg["grasp_threshold"]
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | grasp_success
        print(f"self.reset_buf: {self.reset_buf.shape}, {self.reset_buf.float().mean()}, {self.reset_buf.float().max()}, {self.reset_buf.float().min()}")

        self.extras["grasp_success"] = grasp_success.float()

        # Reset environments where episode is done
        if torch.any(self.reset_buf):
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        print(f"self.rew_buf: {self.rew_buf.shape}, {self.rew_buf.mean()}, {self.rew_buf.max()}, {self.rew_buf.min()}")
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Reset joint positions and velocities
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            self.default_dof_pos,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # Reset the cube position (with a little randomization for diversity)
        new_cube_pos = self.cube_init_pos + 0.02 * (torch.rand((len(envs_idx), 3), device=self.device) - 0.5)
        self.cube_pos[envs_idx] = new_cube_pos
        self.cube.set_pos(new_cube_pos.cpu().numpy(), envs_idx=envs_idx, zero_velocity=True)

        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def get_privileged_observations(self):
        return None
    
    def get_observations(self):
        return self.obs_buf