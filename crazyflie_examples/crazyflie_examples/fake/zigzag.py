import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
import collections
from tensordict.tensordict import TensorDict, TensorDictBase
from functorch import vmap

class FakeZigZag(FakeEnv):
    def __init__(self, cfg, connection, swarm):
        self.alpha = 0.8
        self.num_envs = 1
        self.cfg = cfg
        self.future_traj_steps = 4
        self.dt = 0.01
        self.num_cf = 1

        super().__init__(cfg, connection, swarm)

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.0, -0.0, 0.], device=self.device) * torch.pi,
            torch.tensor([0.0, 0.0, 0.], device=self.device) * torch.pi
        )
        
        # eval
        self.target_times_dist = D.Uniform(
            torch.tensor(1.3, device=self.device),
            torch.tensor(1.3, device=self.device)
        )
        self.origin = torch.tensor([0., 0., 1.], device=self.device)
        self.num_points = 20

        self.traj_t0 = 0.0
        self.target_times = torch.zeros(self.num_envs, self.num_points - 1, device=self.device)
        self.target_points = torch.zeros(self.num_envs, self.num_points, 2, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)

        # reset / initialize
        env_ids = torch.tensor([0])
        self.target_times[env_ids] = torch.ones((env_ids.shape[0], self.num_points - 1), device=self.device) * self.target_times_dist.sample(torch.Size([env_ids.shape[0], 1]))
        
        # star_points = torch.Tensor([[0.0, 0.0], [0.5, 0.0], [-0.5, 0.4], [0.25, -0.6], [0.25, 0.6], [-0.5, -0.4], \
        #     [0.5, 0.0], [-0.5, 0.4], [0.25, -0.6], [0.25, 0.6], [-0.5, -0.4], \
        #     [0.5, 0.0], [-0.5, 0.4], [0.25, -0.6], [0.25, 0.6], [-0.5, -0.4], \
        #     [0.5, 0.0], [-0.5, 0.4], [0.25, -0.6], [0.25, 0.6]]).to(self.device) * 2.0
        star_points = torch.Tensor([[0.0, 0.0], [0.5, 0.0], [-0.5, 0.4], [0.25, -0.6], [0.25, 0.6], [-0.5, -0.4], \
            [0.5, 0.0], [-0.5, 0.4], [0.25, -0.6], [0.25, 0.6], [-0.5, -0.4], \
            [0.5, 0.0], [-0.5, 0.4], [0.25, -0.6], [0.25, 0.6], [-0.5, -0.4], \
            [0.5, 0.0], [-0.5, 0.4], [0.25, -0.6], [0.25, 0.6]]).to(self.device)
        self.target_points[env_ids] = star_points


        self.target_poses = []

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 3 + 3 + 4 + 3 + 3 # position, velocity, quaternion, heading, up, relative heading
        observation_dim += 3 * (self.future_traj_steps-1)

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action":  BoundedTensorSpec(-1, 1, 4, device=self.device).unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        self.info_spec = CompositeSpec({
            "agents": CompositeSpec({
                "target_position": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                "real_position": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

    def _compute_state_and_obs(self) -> TensorDictBase:
        self.update_drone_state()
        self.target_pos[:] = self._compute_traj(self.future_traj_steps, step_size=1)
        # print(self.target_pos[:, 0])
        self.rpos = self.target_pos.cpu() - self.drone_state[..., :3]
        obs = [self.rpos.flatten(1), self.drone_state[..., 3:10], self.drone_state[..., 13:], torch.zeros((self.num_cf, 4))]
        obs = torch.concat(obs, dim=1).unsqueeze(0)

        # self.target_poses.append(self.target_pos[-1].clone())

        return TensorDict({
            "agents": {
                "observation": obs,
                "target_position": self.target_pos[..., 0, :],
                "real_position": self.drone_state[..., :3]
            },
        }, self.num_envs)

    def _compute_reward_and_done(self) -> TensorDictBase:
        distance = torch.norm(self.rpos[:, [0]][:2], dim=-1)
        # reward = torch.zeros((self.num_envs, 1, 1))
        # reward[..., 0] = distance.mean()
        reward = distance
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

    def _compute_traj(self, steps: int, env_ids=torch.tensor([0]), step_size: float=1.):
        t = self.progress_buf + step_size * torch.arange(steps, device=self.device)
        t = (self.traj_t0 + t * self.dt).unsqueeze(0)
        target_pos = vmap(zigzag)(t, self.target_times[env_ids], self.target_points[env_ids])

        return self.origin + target_pos

    def save_target_traj(self, name):
        torch.save(self.target_poses, name)

def zigzag(t, target_times, target_points):
    # target_times: [batch, num_points]
    # target_points: [batch, num_points, 2]
    
    target_times = torch.concat([torch.zeros(1, device=target_times.device), torch.cumsum(target_times, dim=0)])
    num_points = target_times.shape[0]
    
    times_expanded = target_times.unsqueeze(0).expand(t.shape[-1], -1)
    t_expanded = t.unsqueeze(-1)
    prev_idx = num_points - (times_expanded > t_expanded).sum(dim=-1) - 1
    next_idx = num_points - (times_expanded > t_expanded).sum(dim=-1)
    # clip
    prev_idx = torch.clamp(prev_idx, max=num_points - 2) # [batch, future_step]
    next_idx = torch.clamp(next_idx, max=num_points - 1) # [batch, future_step]

    prev_x = torch.gather(target_points[:,0], 0, prev_idx) # [batch, future_step]
    next_x = torch.gather(target_points[:,0], 0, next_idx)
    prev_y = torch.gather(target_points[:,1], 0, prev_idx)
    next_y = torch.gather(target_points[:,1], 0, next_idx)
    prev_times = torch.gather(target_times, 0, prev_idx)
    next_times = torch.gather(target_times, 0, next_idx)
    k_x = (next_x - prev_x) / (next_times - prev_times)
    k_y = (next_y - prev_y) / (next_times - prev_times)
    x = prev_x + k_x * (t - prev_times) # [batch, future_step]
    y = prev_y + k_y * (t - prev_times)
    z = torch.zeros_like(x)
    
    return torch.stack([x, y, z], dim=-1)

def pentagram(t):
    x = -1.0 * torch.sin(2 * t) - 0.5 * torch.sin(3 * t)
    y = 1.0 * torch.cos(2 * t) - 0.5 * torch.cos(3 * t)
    # x = -1.1 * torch.sin(2 * t) - 0.5 * torch.sin(3 * t)
    # y = 1.1 * torch.cos(2 * t) - 0.5 * torch.cos(3 * t)
    z = torch.zeros_like(t)
    return torch.stack([x,y,z], dim=-1)

def lemniscate(t, c):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    sin2p1 = torch.square(sin_t) + 1

    x = torch.stack([
        cos_t, sin_t * cos_t, c * sin_t
    ], dim=-1) / sin2p1.unsqueeze(-1)

    return x

def circle(t):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    x = torch.stack([
        cos_t, sin_t, torch.zeros_like(sin_t)
    ], dim=-1)

    return x

def square(t_s):
    x_s = []
    for t_ in t_s[0]:
        t = torch.abs(t_).item()
        while t >= 8:
            t -= 8
        if t < 2:
            x = torch.tensor([-1., 1-t, 0.])
        elif t < 4:
            x = torch.tensor([t-3, -1., 0.])
        elif t < 6:
            x = torch.tensor([1., t-5, 0.])
        elif t < 8:
            x = torch.tensor([7-t, 1., 0.])
        x_s.append(x)
    x_s = torch.stack(x_s, dim=0).unsqueeze(0).to(t_s.device)
    return x_s

def scale_time(t, a: float=1.0):
    return t / (1 + 1/(a*torch.abs(t)))

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
