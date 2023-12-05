import abc

from typing import Dict, List, Optional, Tuple, Type, Union, Callable

import torch
import logging
# import carb
import numpy as np
import yaml

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, TensorSpec, DiscreteTensorSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from functorch import vmap
from omni_drones.utils.torch import quaternion_to_euler
from omni_drones.utils.torchrl import AgentSpec
import os.path as osp

import sys
sys.path.append('..')
import time

from crazyflie_py import Crazyswarm
from crazyflie_interfaces.msg import LogDataGeneric
import rclpy
from multiprocessing import Process
from rclpy.executors import MultiThreadedExecutor
from .subscriber import Subscriber

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

class FakeRobot():
    def __init__(self, cfg, name, device, id):
        self.name = name
        self.device = device
        self.cfg = cfg
        if name == "Hummingbird":
            self.num_rotors = 4
        elif name == "Crazyflie" or "crazyflie":
            self.num_rotors = 4
        elif name == "Firefly":
            self.num_rotors = 6

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        self.intrinsics_spec = CompositeSpec({
            "mass": UnboundedContinuousTensorSpec(1),
            "inertia": UnboundedContinuousTensorSpec(3),
            "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            "KM": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_up": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_down": UnboundedContinuousTensorSpec(self.num_rotors),
            "drag_coef": UnboundedContinuousTensorSpec(1),
            "rotor_offset": UnboundedContinuousTensorSpec(1),
        }).to(self.device)

        if self.cfg.force_sensor:
            self.use_force_sensor = True
            state_dim = 19 + self.num_rotors + 3
        else:
            self.use_force_sensor = False
            state_dim = 19 + self.num_rotors
        self.state_spec = UnboundedContinuousTensorSpec(state_dim, device=self.device)

        self.n = 1
        self.id = id

    def update_drone_pos(self, log, drone_state):
        drone_state[0][0] = log.values[0]
        drone_state[0][1] = log.values[1]
        drone_state[0][2] = log.values[2]

    def update_drone_quat(self, log, drone_state):
        drone_state[0][3] = log.values[0]
        drone_state[0][4] = log.values[1]
        drone_state[0][5] = log.values[2]
        drone_state[0][6] = log.values[3]

    def update_drone_vel(self, log, drone_state):
        drone_state[0][7] = log.values[0]
        drone_state[0][8] = log.values[1]
        drone_state[0][9] = log.values[2]

    def update_drone_pos_1(self, log, drone_state):
        drone_state[1][0] = log.values[0]
        drone_state[1][1] = log.values[1]
        drone_state[1][2] = log.values[2]

    def update_drone_quat_1(self, log, drone_state):
        drone_state[1][3] = log.values[0]
        drone_state[1][4] = log.values[1]
        drone_state[1][5] = log.values[2]
        drone_state[1][6] = log.values[3]

    def update_drone_vel_1(self, log, drone_state):
        drone_state[1][7] = log.values[0]
        drone_state[1][8] = log.values[1]
        drone_state[1][9] = log.values[2]

    def update_drone_pos_2(self, log, drone_state):
        drone_state[2][0] = log.values[0]
        drone_state[2][1] = log.values[1]
        drone_state[2][2] = log.values[2]

    def update_drone_quat_2(self, log, drone_state):
        drone_state[2][3] = log.values[0]
        drone_state[2][4] = log.values[1]
        drone_state[2][5] = log.values[2]
        drone_state[2][6] = log.values[3]

    def update_drone_vel_2(self, log, drone_state):
        drone_state[2][7] = log.values[0]
        drone_state[2][8] = log.values[1]
        drone_state[2][9] = log.values[2]
        
class Swarm():
    def __init__(self, cfg, test=False):
        self.cfg = cfg
        self.test = test
        if self.test:
            return
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.cfs = self.swarm.allcfs.crazyflies
        self.num_cf = len(self.cfs)
        self.drone_state = torch.zeros((self.num_cf, 16)) # position, velocity, quaternion, heading, up, relative heading
        self.drone_state[..., 3] = 1. # default rotation
        self.nodes = []
        self.drones = []
        id = 0

        for cf in self.cfs:
            drone = FakeRobot(self.cfg.task, self.cfg.task.drone_model, device = cfg.sim.device, id=id)
            if id == 0:
                node = Subscriber(
                    cf.prefix, 
                    lambda x: drone.update_drone_pos(x, self.drone_state), 
                    lambda x: drone.update_drone_quat(x, self.drone_state), 
                    lambda x: drone.update_drone_vel(x, self.drone_state), 
                )
            elif id == 1:
                node = Subscriber(
                    cf.prefix, 
                    lambda x: drone.update_drone_pos_1(x, self.drone_state), 
                    lambda x: drone.update_drone_quat_1(x, self.drone_state), 
                    lambda x: drone.update_drone_vel_1(x, self.drone_state), 
                )
            elif id == 2:
                node = Subscriber(
                    cf.prefix, 
                    lambda x: drone.update_drone_pos_2(x, self.drone_state), 
                    lambda x: drone.update_drone_quat_2(x, self.drone_state), 
                    lambda x: drone.update_drone_vel_2(x, self.drone_state), 
                )
            self.drones.append(drone)
            self.nodes.append(node)
            id += 1

            # set to CTBR mode
            cf.setParam("flightmode.stabModeRoll", 0)
            cf.setParam("flightmode.stabModePitch", 0)
            cf.setParam("flightmode.stabModeYaw", 0)


    def get_drone_state(self):
        # update observation
        if rclpy.ok():
            for i in range(self.num_cf):
                rclpy.spin_once(self.nodes[i]) # pos
                rclpy.spin_once(self.nodes[i]) # quat
                rclpy.spin_once(self.nodes[i]) # vel
        return self.drone_state
    
    def act(self, all_action, rpy_scale=30):
        if self.test:
            return
        for id in range(self.num_cf):
            action = all_action[0][id].cpu().numpy().astype(float)
            cf = self.cfs[id]
            thrust = (action[3] + 1) / 2
            thrust = float(max(0, min(0.9, thrust)))
            cf.cmdVel(action[0] * rpy_scale, -action[1] * rpy_scale, action[2] * rpy_scale, thrust*2**16)
        self.timeHelper.sleepForRate(50)

    def init(self):
        if self.test:
            return
        # send several 0-thrust commands to prevent thrust deadlock
        for i in range(20):
            for cf in self.cfs:
                cf.cmdVel(0.,0.,0.,0.)
            self.timeHelper.sleepForRate(100)

    def end_program(self):
        if self.test:
            return
        # end program
        for i in range(20):
            for cf in self.cfs:
                cf.cmdVel(0., 0., 0., 0.)
            self.timeHelper.sleepForRate(100)
        for i in range(self.num_cf):    
            self.nodes[i].destroy_node()
        rclpy.shutdown()

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
            self.drone_state = self.swarm.get_drone_state()
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

