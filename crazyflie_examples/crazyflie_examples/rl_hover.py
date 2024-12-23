import logging
import os
import time

import hydra
import torch
import numpy as np
from functorch import vmap
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH #, init_simulation_app
from torchrl.collectors import SyncDataCollector 
from omni_drones.utils.torchrl import AgentSpec
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    VelController,
    AttitudeController,
    RateController,
    History
)
from omni_drones.learning.ppo import PPORNNPolicy, PPOPolicy
from omni_drones.learning import (
    MAPPOPolicy, 
)

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    Compose,
)

from tqdm import tqdm
from fake import FakeHover, FakeTrack, FakeNewTrack, Swarm, FakeTurn, FakeLine
import time

from crazyflie_py import Crazyswarm
from torchrl.envs.utils import step_mdp
import collections

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="deploy")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    algos = {
        "mappo": MAPPOPolicy, 
    }

    # swarm = Swarm(cfg, test=False, mass=31.6 / 34.3)
    swarm = Swarm(cfg, test=False)
    # for cf in swarm.cfs:
    #     cf.setParam("pid_rate.yaw_kp", 360)

    cmd_fre = 100
    rpy_scale = 180

    # load takeoff checkpoint
    takeoff_ckpt = "model/hover/Hover.pt"
    # takeoff_ckpt = "model/hover/Hover_wotime.pt"
    takeoff_env = FakeHover(cfg, connection=True, swarm=swarm)
    takeoff_agent_spec = takeoff_env.agent_spec["drone"]
    takeoff_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=takeoff_agent_spec, device=takeoff_env.device)
    takeoff_state_dict = torch.load(takeoff_ckpt)
    takeoff_policy.load_state_dict(takeoff_state_dict)


    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up

        data = takeoff_env.reset().to(dest=takeoff_env.device)
        data = takeoff_policy(data, deterministic=True)

        swarm.init()

        last_time = time.time()
        data_frame = []

        # update observation
        target_pos = takeoff_env.drone_state[..., :3]
        takeoff_env.target_pos = torch.tensor([[1.0, 1.0, 1.0]])

        takeoff_frame = []
        action_frame = []
        # takeoff
        for timestep in range(2000):
            data = takeoff_env.step(data)
            data = step_mdp(data)
            
            data = takeoff_policy(data, deterministic=True)
            takeoff_frame.append(data.clone())
            action = torch.tanh(data[("agents", "action")])

            # for log
            thrust = (action[...,3] + 1) / 2
            thrust = max(0, min(0.9, thrust))*2**16
            roll_rate = action[...,0] * rpy_scale
            pitch_rate = -action[...,1] * rpy_scale
            yaw_rate = -action[...,2] * rpy_scale
            action_frame.append(torch.concat([roll_rate, pitch_rate, yaw_rate, thrust], dim=-1))

            swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time
        
        print('start pos', takeoff_env.drone_state[..., :3])
        print('pos error xyz', torch.norm(takeoff_env.target_pos - takeoff_env.drone_state[..., :3]))
        print('pos error xy', torch.norm(takeoff_env.target_pos[..., :2] - takeoff_env.drone_state[..., :2]))

        takeoff_env.target_pos = torch.tensor([[0., 0., 1.0]])
        print('target', target_pos)
        # land
        for timestep in range(600):
            data = takeoff_env.step(data)
            data = step_mdp(data)

            data = takeoff_policy(data, deterministic=True)
            action = torch.tanh(data[("agents", "action")])

            swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time

            if timestep == 100:
                target_pos[..., 2] = 1.0
                takeoff_env.target_pos = torch.tensor([[0., 0., 1.0]])

            if timestep == 200:
                target_pos[..., 2] = 0.8
                takeoff_env.target_pos = torch.tensor([[0., 0., 0.8]])

            if timestep == 300:
                target_pos[..., 2] = 0.6
                takeoff_env.target_pos = torch.tensor([[0., 0., 0.6]])

            if timestep == 400:
                target_pos[..., 2] = 0.4
                takeoff_env.target_pos = torch.tensor([[0., 0., 0.4]])

            if timestep == 500:
                target_pos[..., 2] = 0.2
                takeoff_env.target_pos = torch.tensor([[0., 0., 0.2]])
        print('land pos', takeoff_env.drone_state[..., :3])


    swarm.end_program()
    
    torch.save(takeoff_frame, "sim2real_data/takeoff.pt")
    torch.save(action_frame, "sim2real_data/takeoff_action.pt")

if __name__ == "__main__":
    main()
