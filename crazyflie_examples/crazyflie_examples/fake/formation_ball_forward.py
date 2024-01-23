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

LINE = [
    [-0.1, -0.1, 0],
    [0.1, 0.1, 0]
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
    "line": LINE,
    'triangle': REGULAR_TRIANGLE,
}

class FormationBallForward(FakeEnv):
    def __init__(self, cfg, connection, swarm):
        self.alpha = 0.8
        self.cfg = cfg
        self.num_cf = 2
        self.num_obstacle = 1
        super().__init__(cfg, connection, swarm)
        self.time_encoding = True
        self.drone_id = torch.Tensor(np.arange(self.num_cf))
        self.target_vel = torch.Tensor([1,1,0]).float()
        self.mask_observation = torch.tensor([self.cfg.task.obs_range, -1, -1, -1, -1, -1, -1, -1, -1, -1]).float()
        
        
    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 3 + 3 + 4 + 3 + 3 + 3 
        # position, velocity, quaternion, heading, up, relative heading, relative velocity
        
        if self.cfg.algo.share_actor:
            self.id_dim = 3
            observation_dim += self.id_dim

        if not self.cfg.task.use_separate_obs:
            agent_obs_spec = CompositeSpec({
                    "obs_self": UnboundedContinuousTensorSpec((1, observation_dim)), # 23
                    "obs_others": UnboundedContinuousTensorSpec((self.num_cf-1, 7)), # 5 * 14 =70
                    "attn_obs_obstacles": UnboundedContinuousTensorSpec((self.num_ball + self.num_static_obstacle, 10)), # 7
                }).expand(self.num_cf)
        else:
            agent_obs_spec = CompositeSpec({
                    "obs_self": UnboundedContinuousTensorSpec((1, observation_dim)), # 23
                    "obs_others": UnboundedContinuousTensorSpec((self.num_cf-1, 7)), # 5 * 14 =70
                    "attn_obs_ball": UnboundedContinuousTensorSpec((self.num_ball, 10)), # 7
                    "attn_obs_static": UnboundedContinuousTensorSpec((self.num_static_obstacle, 10))
                }).expand(self.num_cf)
                
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": agent_obs_spec, 
            }
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": torch.stack([BoundedTensorSpec(-1, 1, 4, device=self.device).unsqueeze(0)]*self.num_cf, dim=0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((self.num_cf, 1))
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
        # self-observation
        self.update_drone_state()
        self.formation_center = self.drone_state[..., :3].mean(-2, keepdim=True)

        rel_vel = self.drone_state[..., 7:10] - self.target_vel #[env_num, drone_num, 3]

        obs_self = [self.drone_state, rel_vel]
        if self.cfg.algo.share_actor:
            obs_self.append(self.drone_id.reshape(-1, 1).expand(-1, self.id_dim))
        obs_self = torch.concat(obs_self, dim=1).unsqueeze(0)
        obs_self[..., 0] -= self.formation_center[..., 0]
        obs_self[..., 1] -= self.formation_center[..., 1]
        obs_self[..., 2] -= 1.5 # target height

        pos = self.drone_state[..., :3].unsqueeze(0) # TODO: check
        vel = self.drone_state[..., 7:10].unsqueeze(0) # TODO: check

        # other's observation
        relative_pos = vmap(cpos)(pos, pos)
        self.drone_pdist = vmap(off_diag)(torch.norm(relative_pos, dim=-1, keepdim=True))   # pair wise distance
        relative_pos = vmap(off_diag)(relative_pos)
        relative_vel = vmap(cpos)(vel, vel)
        relative_vel = vmap(off_diag)(relative_vel)

        obs_others = torch.cat([
            relative_pos,
            self.drone_pdist,
            relative_vel,
        ], dim=-1)

        # obstacle observation
        obstacle_vel = []
        relative_obs_pos = []
        relative_obs_dis = []

        if self.num_ball > 0:
            balls_pos = self.ball_state[..., :3].unsqueeze(0) # [env_num, ball_num, 3]
            balls_vel = self.ball_state[..., 3:].unsqueeze(0) # [env_num, 1, ball_num, 3]
            obstacle_vel.append(balls_vel)
            relative_b_pos =  balls_pos.unsqueeze(1) - pos[..., :3].unsqueeze(2) # [env_num, drone_num, 1, 3] - [env_num, 1, ball_num, 3]
            relative_b_dis = self.relative_b_dis = torch.norm(relative_b_pos, p=2, dim=-1) # [env_num, drone_num, ball_num, 3] -> [env_num, drone_num, ball_num]
            relative_obs_pos.append(relative_b_pos)
            relative_obs_dis.append(relative_b_dis)

        if self.num_static_obstacle > 0:
            cubes_pos = self.obstacle_state.unsqueeze(0)
            obstacle_vel.append(torch.zeros_like(cubes_pos))
            relative_c_pos = cubes_pos.unsqueeze(1) - pos[..., :3].unsqueeze(2) # [env_num, drone_num, 1, 3] - [env_num, 1, ball_num, 3]
            relative_c_dis = torch.norm(relative_c_pos[..., :2], p=2, dim=-1) # for columns, calculate x & y distance
            relative_obs_pos.append(relative_c_pos)
            relative_obs_dis.append(relative_c_dis)

        # calculate full obstacle observation
        obstacle_vel = torch.cat(obstacle_vel, dim=1)
        relative_obs_pos = torch.cat(relative_obs_pos, dim=2)
        self.relative_obs_dis = relative_obs_dis = torch.cat(relative_obs_dis, dim=2)
        relative_obs_vel = obstacle_vel.unsqueeze(1) - vel.unsqueeze(2)

        obs_obstacle = torch.cat([
            relative_obs_dis.unsqueeze(-1), 
            relative_obs_pos, # [n, k, m, 3]
            relative_obs_vel,
            obstacle_vel.unsqueeze(1).expand(-1,self.num_cf,-1,-1)
        ], dim=-1).view(self.num_envs, self.num_cf, -1, 10) #[env, agent, obstable_num, *]
        
        # mask_behind = relative_obs_pos[..., 1] < 0
        # obs_obstacle[mask_behind] = self.mask_observation

        # mask_range = relative_obs_pos[..., 1] > self.cfg.task.obs_range
        # obs_obstacle[mask_range] = self.mask_observation
        obs_obstacle[..., :] = self.mask_observation

        # obs = torch.cat([
        #     obs_self.reshape(1, self.num_cf, -1), 
        #     obs_others.reshape(1, self.num_cf,  -1), 
        #     obs_obstacle.reshape(1, self.num_cf, -1),
        #     ], dim=-1)

        if not self.cfg.task.use_separate_obs:
            obs = TensorDict({ 
                "obs_self": obs_self.unsqueeze(2),  # [N, K, 1, obs_self_dim]
                "obs_others": obs_others, # [N, K, K-1, obs_others_dim]
                "attn_obs_obstacles": obs_obstacle, # [N, K, ball_num+static_obs, *]
            }, [self.num_envs, self.num_cf]) # [N, K, n_i, m_i]
        else:
            obs_ball = obs_obstacle[:, :, :self.num_ball]
            obs_static = obs_obstacle[:, :, self.num_ball:]
        
            obs = TensorDict({ 
                "obs_self": obs_self.unsqueeze(2),  # [N, K, 1, obs_self_dim]
                "obs_others": obs_others, # [N, K, K-1, obs_others_dim]
                "attn_obs_ball": obs_ball, # [N, K, ball_num, *]
                "attn_obs_static": obs_static
            }, [self.num_envs, self.num_cf]) # [N, K, n_i, m_i]

        return TensorDict({
            "agents": {
                "observation": obs,
            },
        }, self.num_envs)

    def _compute_reward_and_done(self) -> TensorDictBase:
        reward = torch.zeros((self.num_envs, self.num_cf, 1))
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