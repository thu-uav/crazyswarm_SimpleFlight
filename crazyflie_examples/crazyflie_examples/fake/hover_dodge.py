import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase

class FakeHoverDodge(FakeEnv):
    def __init__(self, cfg, connection, swarm):
        self.alpha = 0.8
        self.cfg = cfg
        self.num_cf = 1
        super().__init__(cfg, connection, swarm)
        
        self.target_pos = torch.tensor([[0., 0., .5]])   

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 3 + 3 + 4 + 3 + 3 + 7 # position, velocity, quaternion, heading, up, relative heading

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
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

    def _compute_state_and_obs(self) -> TensorDictBase:
        self.update_drone_state()
        self.rpos = self.target_pos - self.drone_state[..., :3]
        obs = [self.rpos, self.drone_state[..., 3:], torch.zeros((self.num_cf, 4))]
       
        balls_pos = self.obstacle_state[..., :3]

        relative_b_pos = self.drone_state[..., :3] - balls_pos
        balls_vel = self.obstacle_state[..., 3:]
        self.relative_b_dis = torch.norm(relative_b_pos, p=2, dim=-1)
        relative_b_dis = self.relative_b_dis

        obs_ball = torch.cat([
            relative_b_dis.unsqueeze(-1), 
            relative_b_pos, 
            balls_vel.expand_as(relative_b_pos)
        ], dim=-1) #[env, agent, ball_num, *]
        mask = relative_b_dis.detach().cpu().item() > 2.
        fake_obs_ball = torch.zeros_like(obs_ball)
        if mask:
            obs_ball = fake_obs_ball
        obs.append(obs_ball)

        obs = torch.concat(obs, dim=1).unsqueeze(0)

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