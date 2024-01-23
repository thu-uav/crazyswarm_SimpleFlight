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
from fake import FakeHover, FakeTrack, Swarm, PID, MultiHover
import time

from crazyflie_py import Crazyswarm
from torchrl.envs.utils import step_mdp

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="deploy")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    algos = {
        "mappo": MAPPOPolicy, 
    }
    swarm = Swarm(cfg, test=False)
    swarm.init()
    base_env = FakeHover(cfg, connection=True, swarm=swarm)
    controller = PID(device=base_env.device)

    base_env.set_seed(cfg.seed)
    data = base_env.reset().to(device=base_env.device)

    init_pos = base_env.drone_state[..., :3]
    target_pos = init_pos.clone()
    target_pos[..., 2] = 1.
    controller.set_pos(
        init_pos=init_pos,
        target_pos=target_pos
        )
    print("init pos", init_pos)
    print("target pos", target_pos)

    action = controller(base_env.drone_state, timestep=0)
    data['agents', 'action'] = action.to(base_env.device)
    data = base_env.step(data)
    data = step_mdp(data)
    
    init_pos = base_env.drone_state[..., :3]
    target_pos = init_pos.clone()
    target_pos[..., 2] = 1.
    controller.set_pos(
        init_pos=init_pos,
        target_pos=target_pos
        )
    print("init pos", init_pos)
    print("target pos", target_pos)

    # use PID controller to takeoff
    for i in range(200):
        action = controller(base_env.drone_state, timestep=i)
        data['agents', 'action'] = action.to(base_env.device)
        swarm.act_control(action)
        # print(action)

        data = base_env.step(data)
        data = step_mdp(data)

    init_pos = base_env.drone_state[..., :3]
    target_pos = init_pos.clone()
    target_pos[..., 0] = 0.5
    target_pos[..., 1] = 0.5
    target_pos[..., 2] = 1
    controller.set_pos(
        init_pos=init_pos,
        target_pos=target_pos
        )
    print("init pos", init_pos)
    print("target pos", target_pos)

    for i in range(200):
        action = controller(base_env.drone_state, timestep=i)
        data['agents', 'action'] = action.to(base_env.device)
        swarm.act_control(action)
        # print(action)

        data = base_env.step(data)
        data = step_mdp(data)

    # use PID controller to land
    init_pos = base_env.drone_state[..., :3]
    target_pos = init_pos.clone()
    target_pos[..., 2] = 0.04
    controller.set_pos(
        init_pos=init_pos,
        target_pos=target_pos
        )
    print("init pos", init_pos)
    print("target pos", target_pos)

    for i in range(200):
        action = controller(base_env.drone_state, timestep=i)
        data['agents', 'action'] = action.to(base_env.device)
        swarm.act_control(action)
        # print(action)

        data = base_env.step(data)
        data = step_mdp(data)


    # with torch.no_grad():
    #     # the first inference takes significantly longer time. This is a warm up
    #     data = base_env.reset().to(device=base_env.device)
    #     data = policy(data, deterministic=True)

    #     print('start to deploy rl policy')

    #     # update observation
    #     data = base_env.step(data) 

    #     last_time = time.time()
    #     data_frame = []

    #     swarm.init()

    #     # real policy rollout
    #     for timestep in range(10):
            
    #         data = policy(data, deterministic=True)
    #         print(data)
    #         action = torch.tanh(data[("agents", "action")])
    #         swarm.act(action)
    #         print(action)

    #         data = base_env.step(data)
    #         data = step_mdp(data)
    #         data_frame.append(data.clone())
            
    #         cur_time = time.time()
    #         dt = cur_time - last_time
    #         print('time', dt)
    #         last_time = cur_time

    swarm.end_program()
    # torch.save(data_frame, "rl_data/hover.pt")

if __name__ == "__main__":
    main()
