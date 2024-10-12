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
from fake import MultiHover, FakeHover, FakeTrack, FakeStar, FakeNewTrack, Swarm, FakeTurn, FakeLine, FakeHns
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
    data_frame = []

    # load takeoff checkpoint
    takeoff_ckpt = "model/hover/Hover_rapid.pt"
    takeoff_env = MultiHover(cfg, connection=True, swarm=swarm)
    takeoff_agent_spec = takeoff_env.agent_spec["drone"]
    takeoff_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=takeoff_agent_spec, device=takeoff_env.device)
    takeoff_state_dict = torch.load(takeoff_ckpt)
    takeoff_policy.load_state_dict(takeoff_state_dict)
    

    ckpt_name = "model/hns/finalfinal.pt"
    base_env = env = FakeHns(cfg, connection=True, swarm=swarm)
    agent_spec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, value_norm_space=torch.Size([1]), device=takeoff_env.device)
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)
    base_env.set_TP(policy.TP_net)

    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        data = base_env.reset().to(dest=base_env.device)
        data = policy(data, deterministic=True)

        data = takeoff_env.reset().to(dest=takeoff_env.device)
        data = takeoff_policy(data, deterministic=True)

        swarm.init()

        last_time = time.time()
        data_frame = []

        # update observation
        target_pos = takeoff_env.drone_state[..., :3]
        # takeoff_env.target_pos = torch.tensor([[0.5800,  0.0000, 0.56], [0.3000,  0.2800, 0.57], [0.5500, -0.2800, 0.53],]) # random
        takeoff_env.target_pos = torch.tensor([[0.4000,  -0.3000, 1.2], [0.4000,  0.0000, 1.2], [0.400, -0.6000, 1.2],]) # circle

        # takeoff
        for timestep in range(5000):
            data = takeoff_env.step(data)
            data = step_mdp(data)
            
            data = takeoff_policy(data, deterministic=True)
            action = torch.tanh(data[("agents", "action")])

            swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time
        print('start pos', takeoff_env.drone_state[..., :3])

        for track_step in range(600):
            data = base_env.step(data) 
            data = step_mdp(data)
            
            data = policy(data, deterministic=True)
            data_frame.append(data.clone())
            action = torch.tanh(data[("agents", "action")])
            
            swarm.act(action, rpy_scale=60, rate=cmd_fre)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time
        print('real policy done')

        new_target = base_env.drone_state[..., :3]
        print('new target', new_target)

        takeoff_env.target_pos = new_target
        takeoff_env.target_pos[..., 2] = 0.05
        # land
        for timestep in range(300):
            data = takeoff_env.step(data)
            data = step_mdp(data)

            data = takeoff_policy(data, deterministic=True)
            action = torch.tanh(data[("agents", "action")])

            swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time
        print('land pos', takeoff_env.drone_state[..., :3])


    swarm.end_program()
    
    torch.save(data_frame, "sim2real_data/traj_2wall_finalfinal.pt")

if __name__ == "__main__":
    main()
