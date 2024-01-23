import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase
import numpy as np
from functorch import vmap


REGULAR_HEXAGON = [
    [0, 0, 0],
    [1.7321, -1, 0],
    [0, -2, 0],
    [-1.7321, -1, 0],
    [-1.7321, 1.0, 0],
    [0.0, 2.0, 0.0],
    [1.7321, 1.0, 0.0],
]

REGULAR_TETRAGON = [
    [0, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
]

REGULAR_TRIANGLE = [
    [1, 0, 0],
    [-0.5, 0.866, 0],
    [-0.5, -0.866, 0]
]

SINGLE = [
    #[0.618, -1.9021, 0],
    [0, 0, 0],
    [2, 0, 0]
    #[0.618, 1.9021, 0],
]

REGULAR_PENTAGON = [
    [2., 0, 0],
    [0.618, 1.9021, 0],
    [-1.618, 1.1756, 0],
    [-1.618, -1.1756, 0],
    [0.618, -1.9021, 0],
    [0, 0, 0]
]

REGULAR_SQUARE = [
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
]

DENSE_SQUARE = [
    [1, 1, 0],
    [1, 0, 0],
    [1, -1, 0],
    [0, 1, 0],
    [0, 0, 0],
    [0, -1, 0],
    [-1, -1, 0],
    [-1, 0, 0],
    [-1, 1, 0],
]

FORMATIONS = {
    "hexagon": REGULAR_HEXAGON,
    "tetragon": REGULAR_TETRAGON,
    "square": REGULAR_SQUARE,
    "dense_square": DENSE_SQUARE,
    "regular_pentagon": REGULAR_PENTAGON,
    "single": SINGLE,
    'triangle': REGULAR_TRIANGLE,
}

class FormationBall(FakeEnv):
    def __init__(self, cfg, connection, swarm):
        self.alpha = 0.8
        self.cfg = cfg
        self.num_cf = 3
        self.num_obstacle = 1
        super().__init__(cfg, connection, swarm)
        self.time_encoding = True
        self.drone_id = torch.Tensor(np.arange(self.num_cf))
        
        
    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 3 + 3 + 4 + 3 + 3 # position, velocity, quaternion, heading, up, relative heading
        
        if self.cfg.algo.share_actor:
            self.id_dim = 3
            observation_dim += self.id_dim

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(
                    (self.num_cf, observation_dim + (self.num_cf-1)*4 + self.num_obstacle * 7), device=self.device),
                # CompositeSpec({
                #     "obs_self": UnboundedContinuousTensorSpec((1, observation_dim)), # 23
                #     "obs_others": UnboundedContinuousTensorSpec((self.num_cf-1, 10+1)), # 11
                #     "attn_obs_ball": UnboundedContinuousTensorSpec((self.num_obstacle, 3+1+3)),
                # }).expand(self.num_cf),
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": torch.stack([BoundedTensorSpec(-1, 1, 4, device=self.device).unsqueeze(0)]*self.num_cf, dim=0),
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
        self.agent_spec["drone"] = AgentSpec(
            "drone", self.num_cf,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

    def _compute_state_and_obs(self) -> TensorDictBase:
        self.update_drone_state()
        # print(self.drone_state[..., :3])

        obs_self = [self.drone_state, torch.zeros((self.num_cf, 4))]
        if self.cfg.algo.share_actor:
            obs_self.append(self.drone_id.reshape(-1, 1).expand(-1, self.id_dim))
        obs_self = torch.concat(obs_self, dim=1).unsqueeze(0)

        pos = self.drone_state[..., :3].unsqueeze(0) # TODO: check

        relative_pos = vmap(cpos)(pos, pos)
        self.drone_pdist = vmap(off_diag)(torch.norm(relative_pos, dim=-1, keepdim=True))   # pair wise distance
        relative_pos = vmap(off_diag)(relative_pos)

        obs_others = torch.cat([
            relative_pos,
            self.drone_pdist,
            # vmap(others)(self.drone_state[..., 3:10].unsqueeze(0))
        ], dim=-1)

        balls_pos = self.ball_state[..., :3].unsqueeze(0) # [env_num, ball_num, 3]

        relative_b_pos = pos[..., :3].unsqueeze(2) - balls_pos.unsqueeze(1) # [env_num, drone_num, 1, 3] - [env_num, 1, ball_num, 3]
        balls_vel = self.ball_state[..., 3:].unsqueeze(0).unsqueeze(0) # [env_num, 1, ball_num, 3]
        self.relative_b_dis = torch.norm(relative_b_pos, p=2, dim=-1) # [env_num, drone_num, ball_num, 3] -> [env_num, drone_num, ball_num]
        relative_b_dis = self.relative_b_dis # [env_num, drone_num, ball_num]

        obs_ball = torch.cat([
            relative_b_dis.unsqueeze(-1), 
            relative_b_pos, 
            balls_vel.expand_as(relative_b_pos)
        ], dim=-1) #[env, agent, ball_num, *]
        if relative_b_dis.min().detach().cpu().item() > 1.:
            obs_ball = torch.ones_like(obs_ball) * -1
        else:
            print("detect ball!")
        # fake_obs_ball = torch.ones_like(obs_ball) * -1

        obs = torch.cat([
            obs_self.reshape(1, self.num_cf, -1), 
            obs_others.reshape(1, self.num_cf,  -1), 
            obs_ball.reshape(1, self.num_cf, -1),
            ], dim=-1)
            
        return TensorDict({
            "agents": {
                "observation": obs,
            },
        }, self.num_envs)

    def _compute_reward_and_done(self) -> TensorDictBase:
        reward = torch.zeros((self.num_envs, 1, 1))
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

def off_diag(a: torch.Tensor) -> torch.Tensor:
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    return (
        a.flatten(0, 1)[1:]
        .unflatten(0, (n - 1, n + 1))[:, :-1]
        .reshape(n, n - 1, *a.shape[2:])
    )


def cpos(p1: torch.Tensor, p2: torch.Tensor):
    assert p1.shape[1] == p2.shape[1]
    return p1.unsqueeze(1) - p2.unsqueeze(0)


def others(x: torch.Tensor) -> torch.Tensor:
    return off_diag(x.expand(x.shape[0], *x.shape))