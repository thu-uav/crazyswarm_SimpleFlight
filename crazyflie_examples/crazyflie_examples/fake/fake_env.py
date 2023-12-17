import abc

from typing import Dict, List, Optional, Tuple, Type, Union, Callable

import torch

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from functorch import vmap
from omni_drones.utils.torchrl import AgentSpec
import os.path as osp

import sys
sys.path.append('..')
import time

def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


class FakeEnv(EnvBase):
    REGISTRY: Dict[str, Type["FakeEnv"]] = {}

    def __init__(self, cfg, connection, swarm):
        super().__init__(
            device=cfg.sim.device, batch_size=[cfg.env.num_envs], run_type_checks=False
        )
        # store inputs to class
        self.cfg = cfg
        # extract commonly used parameters
        self.num_envs = self.cfg.env.num_envs
        self.num_obstacle = 1
        self.connection = connection
        self.progress_buf = 0

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        self._set_specs()
        self.batch_size = [self.num_envs]
        if connection:
            self.swarm = swarm
            self.num_cf = self.swarm.num_cf
        else:
            self.drone_state = torch.zeros((self.num_cf, 16)) # position, velocity, quaternion, heading, up, relative heading
            self.drone_state[..., 3] = 1. # default rotation
            self.obstacle_state = torch.zeros((self.num_obstacle, 6))

    @property
    def agent_spec(self):
        if not hasattr(self, "_agent_spec"):
            self._agent_spec = {}
        return _AgentSpecView(self)
    
    @agent_spec.setter
    def agent_spec(self, value):
        raise AttributeError(
            "Do not set agent_spec directly."
            "Use `self.agent_spec[agent_name] = AgentSpec(...)` instead."
        )

    def close(self):
        return
        
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.progress_buf = 0
        return self._compute_state_and_obs()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = TensorDict({"next": {}}, self.batch_size)
        obs = self._compute_state_and_obs()
        tensordict["next"].update(obs)
        tensordict["next"].update(self._compute_reward_and_done())
        tensordict.update(obs)
        self.progress_buf += 1
        return tensordict

    @abc.abstractmethod
    def _compute_state_and_obs(self) -> TensorDictBase:
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_reward_and_done(self) -> TensorDictBase:
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int] = -1):
        torch.manual_seed(seed)

    def to(self, device) -> EnvBase:
        if torch.device(device) != self.device:
            raise RuntimeError(
                f"Cannot move IsaacEnv on {self.device} to a different device {device} once it's initialized."
            )
        return self
    
    def update_drone_state(self):
        if self.connection:
            self.drone_state, self.obstacle_state = self.swarm.get_drone_state()
        rot = self.drone_state[..., 3:7]
        self.drone_state[..., 10:13] = vmap(quat_axis)(rot.unsqueeze(0), axis=0).squeeze()
        self.drone_state[..., 13:16] = vmap(quat_axis)(rot.unsqueeze(0), axis=2).squeeze()


class _AgentSpecView(Dict[str, AgentSpec]):
    def __init__(self, env: FakeEnv):
        super().__init__(env._agent_spec)
        self.env = env

    def __setitem__(self, k: str, v: AgentSpec) -> None:
        v._env = self.env
        return self.env._agent_spec.__setitem__(k, v)

