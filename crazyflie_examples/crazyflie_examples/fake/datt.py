import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
import collections
from tensordict.tensordict import TensorDict, TensorDictBase
from .utils.chained_polynomial import ChainedPolynomial
from .utils.zigzag import RandomZigzag
from .utils.pointed_star import NPointedStar
from functorch import vmap

class FakeDATT(FakeEnv):
    def __init__(self, cfg, connection, swarm):
        self.alpha = 0.8
        self.num_envs = 1
        self.cfg = cfg
        self.future_traj_steps = 10
        self.dt = 0.01
        self.num_cf = 1
        self.use_time_encoding = False
        self.eval_traj = 2 # 0: smooth, 1: zigzag, 2: star
        seed = 3 # times of resetting

        super().__init__(cfg, connection, swarm)

        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        )
        self.origin = torch.tensor([0., 0., 1.], device=self.device)

        if self.eval_traj == 0:
            self.ref = ChainedPolynomial(num_trajs=self.num_envs,
                                    scale=1.0,
                                    use_y=True,
                                    min_dt=1.5,
                                    max_dt=4.0,
                                    degree=5,
                                    origin=self.origin,
                                    device=self.device)
        elif self.eval_traj == 1:
            self.ref = RandomZigzag(num_trajs=self.num_envs,
                                    max_D=[1.0, 1.0, 0.0],
                                    min_dt=1.0,
                                    max_dt=1.5,
                                    diff_axis=True,
                                    origin=self.origin,
                                    device=self.device)
        else:
            self.ref = NPointedStar(num_trajs=self.num_envs,
                            num_points=5,
                            origin=self.origin,
                            speed=0.5,
                            radius=0.7,
                            device=self.device)

        for _ in range(seed):
            self.ref.reset()

        self.traj_t0 = torch.ones(self.num_envs, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, 4, device=self.device)

        # reset / initialize
        env_ids = torch.tensor([0])
        self.T_scale = torch.ones(self.num_envs, device=self.device)
        self.traj_t0[env_ids] = 0.0 * self.T_scale[env_ids]

        # add init_action to self.action_history_buffer
        # init: hover
        self.prev_actions[env_ids] = torch.tensor([[0.0000, 0.0000, 0.0000, 0.5828]]).to(self.device)
        for _ in range(self.action_history):
            self.action_history_buffer.append(self.prev_actions) # add all prev_actions, not len(env_ids)

        self.target_poses = []

    def _set_specs(self):
        self.use_action_history = False
        self.action_history_step = 1
        self.use_obs_norm = False

        drone_state_dim = 3 + 3 + 3 + 3
        observation_dim = drone_state_dim + 3 * self.future_traj_steps

        if self.use_time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        # action history
        self.action_history = self.action_history_step if self.use_action_history else 0
        self.action_history_buffer = collections.deque(maxlen=self.action_history)
        if self.action_history > 0:
            observation_dim += self.action_history * 4

        if self.use_obs_norm:
            rpos_range = [5.0, 5.0, 5.0] * self.future_traj_steps
            if self.task == 'slow':
                vel_range = [0.5, 0.5, 0.5]
            elif self.task == 'normal':
                vel_range = [2.0, 2.0, 2.0]
            else:
                vel_range = [3.0, 3.0, 3.0]
            rotation_range = [1.0] * 9
            self.obs_norm_range = torch.tensor(rpos_range + vel_range + rotation_range)

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
        self.target_pos[:] = self._compute_traj(self.future_traj_steps, step_size=5)
        # print(self.target_pos[:, 0])
        self.rpos = self.target_pos.cpu() - self.drone_state[..., :3]
        
        # obs = [self.rpos.flatten(1), self.drone_state[..., 3:10], self.drone_state[..., 13:19]] # old version
        obs = [
            self.rpos.flatten(1),
            self.drone_state[..., 7:10], # linear v
            self.drone_state[..., 19:28], # rotation
        ]

        obs = torch.concat(obs, dim=1).unsqueeze(0)

        if self.use_obs_norm:
            obs = obs / self.obs_norm_range.unsqueeze(0).unsqueeze(0).repeat(self.num_envs, 1, 1)

        # add action history to actor
        if self.action_history > 0:
            self.action_history_buffer.append(self.prev_actions)
            all_action_history = torch.concat(list(self.action_history_buffer), dim=-1).unsqueeze(1).cpu()
            obs = torch.concat([obs, all_action_history], dim=-1)
        # print('prev_actions', all_action_history)

        return TensorDict({
            "agents": {
                "observation": obs,
                "target_position": self.target_pos[..., 0, :],
                "real_position": self.drone_state[..., :3],
                "drone_state": self.drone_state,
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
        t = self.traj_t0 + t * self.dt * torch.ones(self.num_envs, 1, device=self.device)

        target_pos = []
        for ti in range(t.shape[1]):
            target_pos.append(self.ref.pos(t[:, ti]))
        target_pos = torch.stack(target_pos, dim=1)[env_ids]

        return target_pos

    def save_target_traj(self, name):
        torch.save(self.target_poses, name)

def pentagram(t):
    x = -1.0 * torch.sin(2 * t) - 0.5 * torch.sin(3 * t)
    y = 1.0 * torch.cos(2 * t) - 0.5 * torch.cos(3 * t)
    # x = -1.1 * torch.sin(2 * t) - 0.5 * torch.sin(3 * t)
    # y = 1.1 * torch.cos(2 * t) - 0.5 * torch.cos(3 * t)
    z = torch.zeros_like(t)
    return torch.stack([x,y,z], dim=-1)

def lemniscate_v(t, T):
    sin_t = torch.sin(2 * torch.pi * t / T)
    cos_t = torch.cos(2 * torch.pi * t / T)

    x = torch.stack([
        cos_t, sin_t * cos_t, torch.zeros_like(t)
    ], dim=-1)

    return x

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
