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
ASSET_PATH = "/home/tangwenhao/OmniDrones/omni_drones/robots/assets"

from .subscriber import Subscriber
import sys
sys.path.append('..')
# sys.path.append('...')
from crazyflie_py import Crazyswarm
from crazyflie_interfaces.msg import LogDataGeneric
import rclpy
from multiprocessing import Process
from rclpy.executors import MultiThreadedExecutor
TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0
# import asyncio

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
    def __init__(self, cfg, name, device):
        self.name = name
        self.device = device
        self.cfg = cfg
        if name == "Hummingbird":
            self.param_path = ASSET_PATH + "/usd/hummingbird.yaml"
        elif name == "crazyflie":
            self.param_path = ASSET_PATH + "/usd/crazyflie.yaml"
        elif name == "Firefly":
            self.param_path = ASSET_PATH + "/usd/firefly.yaml"

        with open(self.param_path, "r") as f:
            self.params = yaml.safe_load(f)
        self.num_rotors = self.params["rotor_configuration"]["num_rotors"]

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

class FakeEnv(EnvBase):
    REGISTRY: Dict[str, Type["FakeEnv"]] = {}

    def __init__(self, cfg, headless):
        super().__init__(
            device=cfg.sim.device, batch_size=[cfg.env.num_envs], run_type_checks=False
        )
        # store inputs to class
        self.cfg = cfg
        # extract commonly used parameters
        self.num_envs = self.cfg.env.num_envs
        self.max_episode_length = self.cfg.env.max_episode_length
        self.min_episode_length = self.cfg.env.min_episode_length

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        self._set_specs()
        self.batch_size = [1]
        
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.cfs = self.swarm.allcfs.crazyflies
        self.num_cf = len(self.cfs)
        self.drone_state = torch.zeros((self.num_cf, 16))
        self.drone_state[0][3] = 1. # default rotation
        
        # rclpy.init()
        self.executor = MultiThreadedExecutor()
        for cf in self.cfs:
            self.executor.add_node(cf.node)
            # for _ in range(20):
            #     cf.cmdVel(0., 0., 0., 0.)
            node = Subscriber(cf.prefix,  self.update_drone_pos, self.update_drone_quat, self.update_drone_vel) # note: not usable for sim
            # rospy.Subscriber('/cf'+str(cf.id) + '/pos', LogDataGeneric, self.update_drone_pos)
            # rospy.Subscriber('/cf'+str(cf.id) + '/quat', LogDataGeneric, self.update_drone_quat)
            # rospy.Subscriber('/cf'+str(cf.id) + '/vel', LogDataGeneric, self.update_drone_vel)
            
        
        self.executor.add_node(node)
        # executor.spin()
        # rclpy.spin(node)
        # while rclpy.ok():
        #     node.spin_once()
        #     await asyncio.sleep(0)
        # spin(node)
        
        # def spin(node):
        #     rclpy.spin(node)
        # p = Process(target=spin, args=(node))
        # p.start()
        # p.join()
    
    def update_drone_pos(self, log):
        id = 0
        # position
        print('called pos')
        self.drone_state[id][0] = log.values[0]
        self.drone_state[id][1] = log.values[1]
        self.drone_state[id][2] = log.values[2]

    def update_drone_quat(self, log):
        id = 0
        # rotation
        print('called quat')
        self.drone_state[id][3] = log.values[0]
        self.drone_state[id][4] = log.values[1]
        self.drone_state[id][5] = log.values[2]
        self.drone_state[id][6] = log.values[3]
        

    def update_drone_vel(self, log):
        id = 0
        # velocity
        print('called vel')
        self.drone_state[id][7] = log.values[0]
        self.drone_state[id][8] = log.values[1]
        self.drone_state[id][9] = log.values[2]

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
        for cf in self.cfs:
            cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
        self.timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
        for cf in self.cfs:
            for _ in range(10):
                cf.cmdVel(0., 0., 0., 0.61)
        return self._compute_state_and_obs()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if rclpy.ok():
            self.executor.spin_once()
        for id in range(self.num_cf):
            action = tensordict[("agents", "action")][0][id].cpu().numpy().astype(float)
            cf = self.cfs[id]
            
            # # position controller
            # cf.cmdFullState(
            #     pos = action[:3],
            #     vel = action[3:6],
            #     acc = np.zeros(3),
            #     yaw = action[-1],
            #     omega = np.zeros(3)
            # )
            # print(action[:3])
            
            # # # velocity controller
            # cf.cmdVelocityWorld(action[:3], action[-1])
            # print(action[:3])
            
            # rate controller
            # cf.cmdVel(0., 0., 0., 0.7*2**16)
            thrust = action[3] / 7 * 0.63
            thrust = max(0, min(1, thrust))
            # cf.cmdVel(action[0], action[1], action[2], float(thrust*2**16))
            cf.cmdVel(0., 0., 0., 0.62)
            print(action, action[3], thrust)

        self.timeHelper.sleepForRate(100)
        tensordict = TensorDict({"next": {}}, self.batch_size)
        tensordict["next"].update(self._compute_state_and_obs())
        tensordict["next"].update(self._compute_reward_and_done())
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
        if rclpy.ok():
            self.executor.spin_once()
            # self.executor.spin_once()
            # self.executor.spin_once()
            # rclpy.spin_once(self.executor)
            # self.node.spin_once()
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

