import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase

class FakeGoto_static(FakeEnv):
    def __init__(self, cfg, connection, swarm):
        self.alpha = 0.8
        self.cfg = cfg
        self.num_cf = 1
        super().__init__(cfg, connection, swarm)

        self.cylinder_height = 2.0
        self.cylinder_radius = 0.2
        self.cylinder_y_init = 0.8
        self.narrow_width = 0.8
        self.cylinder_pos = torch.zeros(self.num_envs, 2, 3)
        # 2 cylinders
        self.cylinder_pos[:, 0, 0] = -(0.5 * self.narrow_width - self.cylinder_radius)
        self.cylinder_pos[:, 0, 1] = self.cylinder_y_init
        self.cylinder_pos[:, 0, 2] = 0.5 * self.cylinder_height
        self.cylinder_pos[:, 1, 0] = 0.5 * self.narrow_width - self.cylinder_radius
        self.cylinder_pos[:, 1, 1] = - self.cylinder_y_init
        self.cylinder_pos[:, 1, 2] = 0.5 * self.cylinder_height
        # wall
        self.wall_pos = torch.zeros(self.num_envs, 1, 2)
        self.wall_pos[..., 0] = self.narrow_width * 0.5
        self.wall_pos[..., 1] = - self.narrow_width * 0.5
        self.all_cylinder_height = torch.ones(self.num_envs, 1) * self.cylinder_height
        self.all_cylinder_radius = torch.ones(self.num_envs, 1) * self.cylinder_radius

        self.target_pos = torch.tensor([[0., -1.5, 1.]])
        self.max_episode_length = 500

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 3 + 3 + 4 + 3 + 3 # position, velocity, quaternion, heading, up, relative heading

        observation_dim += 3 * 2 + 1 + 1 # cylinder pos, radius and height
        
        observation_dim += 2 # wall

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
        # obs = [self.rpos, self.drone_state[..., 3:10], self.drone_state[..., 13:19], torch.zeros((self.num_cf, 4))]
        
        obs = [self.rpos, self.drone_state[..., 3:10], self.drone_state[..., 13:]]
        
        # cylinder_pos, _ = self.get_env_poses(self.cylinder.get_world_poses())
        self.rpos_cylinder = self.cylinder_pos - self.drone_state[..., :3]
        obs.append(self.rpos_cylinder.reshape(self.num_envs, -1))
        obs.append(self.all_cylinder_height)
        obs.append(self.all_cylinder_radius)
        self.rpos_wall = self.wall_pos - self.drone_state[..., 0].unsqueeze(-1)
        obs.append(self.rpos_wall.squeeze(1))
        

        t = (self.progress_buf / self.max_episode_length) * torch.ones((self.num_cf, 4))
        obs.append(t)

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