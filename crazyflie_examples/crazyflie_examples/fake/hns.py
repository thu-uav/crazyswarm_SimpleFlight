import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
import collections
from tensordict.tensordict import TensorDict, TensorDictBase
from functorch import vmap
import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeHns(FakeEnv):
    def __init__(self, cfg, connection, swarm):
        self.alpha = 0.8
        self.num_envs = 1
        self.cfg = cfg
        self.dt = 0.01
        self.num_agents = 3
        self.scenario_flag = '2wall'
        self.num_cylinders = 7
        self.cylinder_size = 0.1
        self.invalid_z = -20.0
        self.cylinder_height = 1.2
        self.arena_size = 0.9
        self.boundary = self.arena_size - 0.1
        self.obs_max_cylinder = 3
        self.max_episode_length = 800
        self.mask_value = -5.0
        self.history_step = 10
        self.future_predcition_step = 5
        self.max_height = 1.2
        self.target_detect_radius = 100
        self.drone_detect_radius = 100
        self.history_data = collections.deque(maxlen=self.history_step)
        self.v_prey = 1.3
        self.TP = None

        all_cylinders_x = torch.arange(self.num_cylinders) * 2 * self.cylinder_size
        self.cylinders_pos = torch.zeros(self.num_cylinders, 3)
        self.cylinders_pos[:, 0] = all_cylinders_x
        self.cylinders_pos[:, 1] = 0.0
        self.cylinders_pos[:, 2] = self.invalid_z

        super().__init__(cfg, connection, swarm)

        if self.scenario_flag == 'empty':
            num_fixed_cylinders = 0
        elif self.scenario_flag == 'passage':
            num_fixed_cylinders = 6
            self.cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                [2 * self.cylinder_size, self.cylinder_size, 0.5 * self.cylinder_height],
                                [2 * self.cylinder_size, -self.cylinder_size, 0.5 * self.cylinder_height],
                                [-2 * self.cylinder_size, -self.cylinder_size, 0.5 * self.cylinder_height],
                                [-2 * self.cylinder_size, -3 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [-2 * self.cylinder_size, self.cylinder_size, 0.5 * self.cylinder_height],
                                [-2 * self.cylinder_size, 3 * self.cylinder_size, 0.5 * self.cylinder_height],
                            ], device=self.device)
        elif self.scenario_flag == '2wall':
            num_fixed_cylinders = 6
            self.cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                [0.0, 1.5 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [0.0, -1.5 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [0.0, 3.5 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [0.0, -3.5 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [0.0, 5.5 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [0.0, 7.5 * self.cylinder_size, 0.5 * self.cylinder_height],
                            ], device=self.device)
        elif self.scenario_flag == 'wall':
            num_fixed_cylinders = 4
            self.cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                # [0.0, 0.0, 0.5 * self.cylinder_height],
                                [0.0, 1.5 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [0.0, -1.5 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [0.0, 4.5 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [0.0, -4.5 * self.cylinder_size, 0.5 * self.cylinder_height],
                            ], device=self.device)
        elif self.scenario_flag == 'random':
            num_fixed_cylinders = 6
            self.cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                [ -0.4000,   0.4000,   0.6000],
                                [ -0.6000,   0.4000,   0.6000],
                                [ -0.2000,   0.4000,   0.6000],
                                [  0.0000,   0.2000,   0.6000],
                                [  0.0000,  -0.1000,   0.6000],
                                [  0.0000,  -0.35000,   0.6000],
                        ], device=self.device)
        elif self.scenario_flag == 'narrow_gap':
            num_fixed_cylinders = 5
            self.cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                [3 * self.cylinder_size, -3 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [3 * self.cylinder_size, 3 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [-3 * self.cylinder_size, 3 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [-3 * self.cylinder_size, -3 * self.cylinder_size, 0.5 * self.cylinder_height],
                                [0, 3 * self.cylinder_size, 0.5 * self.cylinder_height],
                                # [3 * self.cylinder_size, 0, 0.5 * self.cylinder_height],
                            ], device=self.device)

        if self.scenario_flag == 'empty':
            drone_pos = torch.tensor([
                                [0.6000,  0.0000, 0.5],
                                [0.8000,  0.0000, 0.5],
                                [0.8000, -0.2000, 0.5],
                                [0.8000,  0.2000, 0.5],
                            ])[:self.num_agents]
            self.init_target_pos = torch.tensor([
                                [-0.8000,  0.0000, 0.5],
                            ])
        elif self.scenario_flag == 'wall':
            drone_pos = torch.tensor([
                                [0.6000,  0.4000, 0.5],
                                [0.6000,  0.0000, 0.5],
                                [0.6000, -0.4000, 0.5],
                                [0.8000,  0.2000, 0.5],
                            ])[:self.num_agents]
            self.init_target_pos = torch.tensor([
                                [-0.8000,  0.0000, 0.5],
                            ])
        elif self.scenario_flag == 'narrow_gap':
            drone_pos = torch.tensor([
                                [0.0000,  0.7000, 0.5],
                                [0.2000,  0.7000, 0.5],
                                [-0.2000, 0.7000, 0.5],
                                [0.8000,  0.2000, 0.5],
                            ])[:self.num_agents]
            self.init_target_pos = torch.tensor([
                                [-0.5000,  0.2000, 0.5],
                            ])
        elif self.scenario_flag == 'random':
            drone_pos = torch.tensor([
                                [0.5000,  0.0000, 0.5],
                                [0.5000,  0.3000, 0.5],
                                [0.5000, -0.3000, 0.5],
                                [0.8000,  0.2000, 0.5],
                            ])[:self.num_agents]
            self.init_target_pos = torch.tensor([
                                [-0.8000,  0.0000, 0.5],
                            ])
        elif self.scenario_flag == 'passage':
            drone_pos = torch.tensor([
                                [0.6000,  0.0000, 0.5],
                                [0.8000,  0.2000, 0.5],
                                [0.8000, -0.2000, 0.5],
                                [0.8000,  0.2000, 0.5],
                            ])[:self.num_agents]
            self.init_target_pos = torch.tensor([
                                [0,  0.6000, 0.6],
                            ])
        elif self.scenario_flag == '2wall':
            drone_pos = torch.tensor([
                                [0.6000,  0.0000, 0.5],
                                [0.8000,  0.2000, 0.5],
                                [0.8000, -0.2000, 0.5],
                                [0.8000,  0.2000, 0.5],
                            ])[:self.num_agents]
            self.init_target_pos = torch.tensor([
                                [-0.600,  0.0000, 0.5],
                            ])
        
        self.target_pos = self.init_target_pos.clone().unsqueeze(0)
        self.target_vel = torch.zeros(3)
        self.cylinders_pos = self.cylinders_pos.unsqueeze(0)
        # mask inactive cylinders(underground)
        self.cylinders_mask = (self.cylinders_pos[..., 2] < 0.0) # [num_envs, self.num_cylinders]

        self.active_cylinders = torch.ones(self.num_envs, 1) * num_fixed_cylinders

    def set_TP(self, TP):
        self.TP = TP

    def _set_specs(self):
        self.time_encoding_dim = 4

        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, 3 + 3 * self.future_predcition_step + self.time_encoding_dim + 13)),
            "state_others": UnboundedContinuousTensorSpec((self.num_agents - 1, 3)), # pos
            "cylinders": UnboundedContinuousTensorSpec((self.obs_max_cylinder, 5)), # pos + radius + height
        })
        state_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.num_agents, 3 + 3 * self.future_predcition_step + self.time_encoding_dim + 13)),
            "cylinders": UnboundedContinuousTensorSpec((self.obs_max_cylinder, 5)), # pos + radius + height
        })
        

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": observation_spec.expand(self.num_agents),
                "state": state_spec,
            })
        }).expand(self.num_envs)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action":  BoundedTensorSpec(-1, 1, 4).unsqueeze(0),
            })
        }).expand(self.num_envs)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((self.num_agents, 1))
            })
        }).expand(self.num_envs)
        self.info_spec = CompositeSpec({
            "agents": CompositeSpec({
                "target_position": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                "agent_position": UnboundedContinuousTensorSpec((self.num_agents, 3), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", self.num_agents,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state"),
        )

    def update_target_state(self):
        self.target_pos = self.target_pos + self.target_vel * self.dt

    def _get_dummy_policy_prey(self):
        # drone_pos, _ = self.get_env_poses(self.drone.get_world_poses(False))
        # target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        # cylinders_pos, _ = self.get_env_poses(self.cylinders.get_world_poses())

        drone_pos = self.drone_state[..., :3].unsqueeze(0)
        
        target_rpos = vmap(cpos)(drone_pos, self.target_pos)
        target_cylinders_rpos = vmap(cpos)(self.target_pos, self.cylinders_pos)
        
        force = torch.zeros(self.num_envs, 1, 3)

        # pursuers
        dist_pos = torch.norm(target_rpos, dim=-1).squeeze(1).unsqueeze(-1)

        blocked = is_line_blocked_by_cylinder(drone_pos, self.target_pos, self.cylinders_pos, self.cylinder_size)
        detect_drone = (dist_pos < self.target_detect_radius).squeeze(-1)
        # drone_pos_z_active = (drone_pos[..., 2] > 0.1).unsqueeze(-1)
        # active_drone: if drone is in th detect range, get force from it
        active_drone = detect_drone * (~blocked).unsqueeze(-1) # [num_envs, num_agents, 1]      
        
        force_r_xy_direction = - target_rpos / (dist_pos + 1e-5)
        force_p = force_r_xy_direction * (1 / (dist_pos + 1e-5)) * active_drone.unsqueeze(-1)
        # force_p = -target_rpos.squeeze(1) * (1 / (dist_pos**2 + 1e-5)) * active_drone.unsqueeze(-1)
        force += torch.sum(force_p, dim=1)

        # arena
        # 3D
        force_r = torch.zeros_like(force)
        target_origin_dist = torch.norm(self.target_pos[..., :2],dim=-1)
        force_r_xy_direction = - self.target_pos[..., :2] / (target_origin_dist.unsqueeze(-1) + 1e-5)
        # out of arena
        out_of_arena = self.target_pos[..., 0]**2 + self.target_pos[..., 1]**2 > self.arena_size**2

        force_r[..., 0] = out_of_arena.float() * force_r_xy_direction[..., 0] * (1 / 1e-5) + \
            (~out_of_arena).float() * force_r_xy_direction[..., 0] * (1 / ((self.arena_size - target_origin_dist) + 1e-5))
        force_r[..., 1] = out_of_arena.float() * force_r_xy_direction[..., 1] * (1 / 1e-5) + \
            (~out_of_arena).float() * force_r_xy_direction[..., 1] * (1 / ((self.arena_size - target_origin_dist) + 1e-5))
        
        higher_than_z = (self.target_pos[..., 2] > self.max_height)
        # up
        force_r[...,2] = higher_than_z.float() * (-1 / 1e-5) + \
            (~higher_than_z).float() * - (self.max_height - self.target_pos[..., 2]) / ((self.max_height - self.target_pos[..., 2])**2 + 1e-5)
        lower_than_ground = (self.target_pos[..., 2] < 0.0)
        # down
        force_r[...,2] += (lower_than_ground.float() * (1 / 1e-5) + \
            (~lower_than_ground).float() * - (0.0 - self.target_pos[..., 2]) / ((0.0 - self.target_pos[..., 2])**2 + 1e-5))
        force += force_r
        
        # get force from all cylinders
        # inactive mask, self.cylinders_mask
        force_c = torch.zeros_like(force)
        dist_target_cylinder = torch.norm(target_cylinders_rpos[..., :2], dim=-1)
        dist_target_cylinder_boundary = dist_target_cylinder - self.cylinder_size
        # detect cylinder
        detect_cylinder = (dist_target_cylinder < self.target_detect_radius)
        active_cylinders_force = (~self.cylinders_mask.unsqueeze(1).unsqueeze(-1) * detect_cylinder.unsqueeze(-1)).float()
        force_c_direction_xy = target_cylinders_rpos[..., :2] / (dist_target_cylinder + 1e-5).unsqueeze(-1)
        force_c[..., :2] = (active_cylinders_force * force_c_direction_xy * (1 / (dist_target_cylinder_boundary.unsqueeze(-1) + 1e-5))).sum(2)
        # force_c[..., :2] = (~self.cylinders_mask.unsqueeze(1).unsqueeze(-1) * detect_cylinder.unsqueeze(-1) * target_cylinders_rpos[..., :2] / (dist_target_cylinder**2 + 1e-5).unsqueeze(-1)).sum(2)    

        force += force_c

        force = force.type(torch.float32)
        
        # fixed velocity
        self.target_vel = self.v_prey * force / (torch.norm(force, dim=1).unsqueeze(1) + 1e-5)
        # self.target_vel = self.v_prey * force / (torch.norm(force, dim=-1).unsqueeze(1) + 1e-5)

    def _compute_state_and_obs(self):
        self.update_drone_state()
        self._get_dummy_policy_prey() # update prey vel
        self.update_target_state() # update prey pos

        drone_pos = self.drone_state[..., :3].unsqueeze(0)

        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)
        
        obs = TensorDict({}, [self.num_envs, self.num_agents])

        # cylinders
        cylinders_rpos = vmap(cpos)(drone_pos, self.cylinders_pos) # [num_envs, num_agents, num_cylinders, 3]
        self.cylinders_state = torch.concat([
            cylinders_rpos,
            self.cylinder_height * torch.ones(self.num_envs, self.num_agents, self.num_cylinders, 1),
            self.cylinder_size * torch.ones(self.num_envs, self.num_agents, self.num_cylinders, 1),
        ], dim=-1)
        # cylinders_mdist_z = torch.abs(cylinders_rpos[..., 2]) - 0.5 * self.cylinder_height
        cylinders_mdist = torch.norm(cylinders_rpos, dim=-1) - self.cylinder_size

        # use the kth nearest cylinders
        _, sorted_indices = torch.sort(cylinders_mdist, dim=-1)
        min_k_indices = sorted_indices[..., :self.obs_max_cylinder]
        self.min_distance_idx_expanded = min_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.cylinders_state.shape[-1])
        self.k_nearest_cylinders = self.cylinders_state.gather(2, self.min_distance_idx_expanded)
        # mask invalid cylinders
        self.k_nearest_cylinders_mask = self.cylinders_mask.unsqueeze(1).expand(-1, self.num_agents, -1).gather(2, min_k_indices)
        self.k_nearest_cylinders_masked = self.k_nearest_cylinders.clone()
        self.k_nearest_cylinders_masked.masked_fill_(self.k_nearest_cylinders_mask.unsqueeze(-1).expand_as(self.k_nearest_cylinders_masked), self.mask_value)
        obs["cylinders"] = self.k_nearest_cylinders_masked

        # state_self
        # target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        # target_vel = self.target.get_velocities()
        target_rpos = vmap(cpos)(drone_pos, self.target_pos) # [num_envs, num_agents, 1, 3]
        # self.blocked use in the _compute_reward_and_done
        # _get_dummy_policy_prey: recompute the blocked
        self.blocked = is_line_blocked_by_cylinder(drone_pos, self.target_pos, self.cylinders_pos, self.cylinder_size)
        in_detection_range = (torch.norm(target_rpos, dim=-1) < self.drone_detect_radius)
        # detect: [num_envs, num_agents, 1]
        detect = in_detection_range * (~ self.blocked.unsqueeze(-1))
        # broadcast the detect info to all drones
        self.broadcast_detect = torch.any(detect, dim=1)
        target_rpos_mask = (~ self.broadcast_detect).unsqueeze(-1).unsqueeze(-1).expand_as(target_rpos) # [num_envs, num_agents, 1, 3]
        target_rpos_masked = target_rpos.clone()
        target_rpos_masked.masked_fill_(target_rpos_mask, self.mask_value)
        
        t = (self.progress_buf / self.max_episode_length) * torch.ones((self.num_envs, self.num_agents, 4))

        # TP input
        target_mask = (~ self.broadcast_detect).unsqueeze(-1).expand_as(self.target_pos)
        target_pos_masked = self.target_pos.clone()
        target_pos_masked.masked_fill_(target_mask, self.mask_value)   
        target_vel_masked = self.target_vel[..., :3].clone()
        target_vel_masked.masked_fill_(target_mask, self.mask_value)


        # use the real target pos to supervise the TP network
        TP = TensorDict({}, [self.num_envs])
        frame_state = torch.concat([
            self.progress_buf * torch.ones((self.num_envs, 1)),
            target_pos_masked.reshape(self.num_envs, -1),
            target_vel_masked.squeeze(1),
            drone_pos.reshape(self.num_envs, -1)
        ], dim=-1)
        if len(self.history_data) < self.history_step:
            # init history data
            for i in range(self.history_step):
                self.history_data.append(frame_state)
        else:
            self.history_data.append(frame_state)
        TP['TP_input'] = torch.stack(list(self.history_data), dim=1)
        # target_pos_predicted, x, y -> [-0.5 * self.arena_size, 0.5 * self.arena_size]
        # z -> [0, self.max_height]
        self.target_pos_predicted = self.TP(TP['TP_input'].to(self.device)).to('cpu').reshape(self.num_envs, self.future_predcition_step, -1) # [num_envs, 3 * future_step]
        self.target_pos_predicted[..., :2] = self.target_pos_predicted[..., :2] * 0.5 * self.arena_size
        self.target_pos_predicted[..., 2] = (self.target_pos_predicted[..., 2] + 1.0) / 2.0 * self.max_height

        target_rpos_predicted = (drone_pos.unsqueeze(2) - self.target_pos_predicted.unsqueeze(1)).view(self.num_envs, self.num_agents, -1)

        obs["state_self"] = torch.cat(
            [
            target_rpos_masked.reshape(self.num_envs, self.num_agents, -1),
            target_rpos_predicted,
            self.drone_state[..., 3:10].unsqueeze(0),
            self.drone_state[..., 13:19].unsqueeze(0),
            t,
            ], dim=-1
        ).unsqueeze(2)
                        
        # state_others
        obs["state_others"] = self.drone_rpos
        
        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "TP": TP,
                    "target_position": self.target_pos[..., :3],
                    "real_position": self.drone_state[..., :3].unsqueeze(0)
                }
            },
            self.num_envs,
        )

    def _compute_reward_and_done(self) -> TensorDictBase:
        reward = torch.zeros(self.num_envs, self.num_agents, 1)
        done = torch.zeros((self.num_envs, 1, 1)).bool()
        return TensorDict(
            {
                "agents": {
                    "reward": reward,
                },
                "done": done,
                "terminated": done,
                "truncated": done
            },
            self.num_envs,
        )

def is_perpendicular_line_intersecting_segment(a, b, c):
    # a: drones, b: target, c: cylinders
    
    # the direction of ab
    dx = b[:, :, 0] - a[:, :, 0]  # [batch, num_drones]
    dy = b[:, :, 1] - a[:, :, 1]  # [batch, num_drones]
    
    # c to ab, cd is perpendicular to ab
    num = (c[:, :, 0].unsqueeze(1) - a[:, :, 0].unsqueeze(2)) * dx.unsqueeze(2) + \
          (c[:, :, 1].unsqueeze(1) - a[:, :, 1].unsqueeze(2)) * dy.unsqueeze(2)  # [batch, num_drones, num_cylinders]
    
    denom = dx.unsqueeze(2)**2 + dy.unsqueeze(2)**2  # [batch, num_drones, 1]
    
    t = num / (denom + 1e-5)  # [batch, num_drones, num_cylinders]
    
    # check d in or not in ab
    is_on_segment = (t >= 0) & (t <= 1)  # [batch, num_drones, num_cylinders]
    
    return is_on_segment

def is_line_blocked_by_cylinder(drone_pos, target_pos, cylinder_pos, cylinder_size):
    '''
        # only consider cylinders on the ground
        # 1. compute_reward: for catch reward, not blocked
        # 2. compute_obs: for drones' state, mask the target state in the shadow
        # 3. dummy_prey_policy: if not blocked, the target gets force from the drone
    '''
    # drone_pos: [num_envs, num_agents, 3]
    # target_pos: [num_envs, 1, 3]
    # cylinder_pos: [num_envs, num_cylinders, 3]
    # consider the x-y plane, the distance of c to the line ab
    # d = abs((x2 - x1)(y3 - y1) - (y2 - y1)(x3 - x1)) / sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    batch, num_agents, _ = drone_pos.shape
    _, num_cylinders, _ = cylinder_pos.shape
    
    diff = drone_pos - target_pos
    diff2 = cylinder_pos - target_pos
    # numerator: [num_envs, num_agents, num_cylinders]
    numerator = torch.abs(torch.matmul(diff[..., 0].unsqueeze(-1), diff2[..., 1].unsqueeze(1)) - torch.matmul(diff[..., 1].unsqueeze(-1), diff2[..., 0].unsqueeze(1)))
    # denominator: [num_envs, num_agents, 1]
    denominator = torch.sqrt(diff[..., 0].unsqueeze(-1) ** 2 + diff[..., 1].unsqueeze(-1) ** 2)
    dist_to_line = numerator / (denominator + 1e-5)

    # which cylinder blocks the line between the ith drone and the target
    # blocked: [num_envs, num_agents, num_cylinders]
    blocked = dist_to_line <= cylinder_size
    
    # whether the cylinder between the drone and the target
    flag = is_perpendicular_line_intersecting_segment(drone_pos, target_pos, cylinder_pos)
    
    # cylinders on the ground
    on_ground = (cylinder_pos[..., -1] > 0.0).unsqueeze(1).expand(-1, num_agents, num_cylinders)
    
    blocked = blocked * flag * on_ground

    return blocked.any(dim=(-1))

import functools
def manual_batch(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = set(arg.shape[:-1] for arg in args if isinstance(arg, torch.Tensor))
        if not len(batch_shapes) == 1:
            raise ValueError
        batch_shape = batch_shapes.pop()
        args = (
            arg.reshape(-1, arg.shape[-1]) if isinstance(arg, torch.Tensor) else arg 
            for arg in args
        )
        kwargs = {
            k: v.reshape(-1, v.shape[-1]) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        out = func(*args, **kwargs)
        return out.unflatten(0, batch_shape)
    return wrapped

@manual_batch
def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def cpos(p1: torch.Tensor, p2: torch.Tensor):
    assert p1.shape[1] == p2.shape[1]
    return p1.unsqueeze(1) - p2.unsqueeze(0)

def off_diag(a: torch.Tensor) -> torch.Tensor:
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    return (
        a.flatten(0, 1)[1:]
        .unflatten(0, (n - 1, n + 1))[:, :-1]
        .reshape(n, n - 1, *a.shape[2:])
    )