import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase
from functorch import vmap

class FakeExchange(FakeEnv):
    def __init__(self, cfg, connection, swarm):
        self.alpha = 0.8
        self.cfg = cfg
        self.num_cf = 2
        super().__init__(cfg, connection, swarm)
        
        # self.target_pos = torch.tensor([[0., 0., 1.]])
        self.max_episode_length = 500

        # self.update_drone_state()

        self.target_pos = torch.concat([torch.tensor([0.5, 0.5, 1.0]), torch.tensor([-0.5, -0.5, 1.0])], dim=1)

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 3 + 3 + 4 + 3 + 3 # position, velocity, quaternion, heading, up, relative heading

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

        # other drone: relative position
        drone_pos = self.drone_state[..., :3]
        drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        drone_rpos = vmap(off_diag)(drone_rpos)

        # obs = [self.rpos, self.drone_state[..., 3:10], self.drone_state[..., 13:19], torch.zeros((self.num_cf, 4))]
        
        obs = [self.rpos, self.drone_state[..., 3:10], self.drone_state[..., 13:19]]
        t = (self.progress_buf / self.max_episode_length) * torch.ones((self.num_cf, 4))
        obs.append(t)

        obs.append(drone_rpos.squeeze(2))

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