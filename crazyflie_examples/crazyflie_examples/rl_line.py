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
from fake import FakeHover, FakeHover_old, FakeTrack, Swarm, FakeTurn, FakeLine
import time

from crazyflie_py import Crazyswarm
from torchrl.envs.utils import step_mdp

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
    swarm = Swarm(cfg, test=False, mass=1.0)
    # for cf in swarm.cfs:
    #     cf.setParam("pid_rate.yaw_kp", 360)

    cmd_fre = 100
    rpy_scale = 180

    # load takeoff checkpoint
    # takeoff_ckpt = "model/hover/Hover.pt"
    takeoff_ckpt = "model/hover.pt"
    takeoff_env = FakeHover_old(cfg, connection=True, swarm=swarm)
    takeoff_agent_spec = takeoff_env.agent_spec["drone"]
    takeoff_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=takeoff_agent_spec, device=takeoff_env.device)
    takeoff_state_dict = torch.load(takeoff_ckpt)
    takeoff_policy.load_state_dict(takeoff_state_dict)
    
    # load checkpoint for deployment
    ckpt_name = "rl_data/turn/Line_clip100.pt"
    # ckpt_name = "model/origin.pt"
    base_env = env = FakeLine(cfg, connection=True, swarm=swarm)
    # ckpt_name = "model/1128_mlp.pt"
    # base_env = env = FakeHover(cfg, connection=True, swarm=swarm)
    agent_spec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=base_env.device)
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)

    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        takeoff_data = env.reset().to(dest=base_env.device)
        takeoff_data = policy(takeoff_data, deterministic=True)

        takeoff_data = takeoff_env.reset().to(dest=takeoff_env.device)
        takeoff_data = takeoff_policy(takeoff_data, deterministic=True)

        swarm.init()

        last_time = time.time()
        data_frame = []

        # update observation
        target_pos = takeoff_env.drone_state[..., :3]
        target_pos[..., 2] = 0.5
        takeoff_env.target_pos = target_pos
        print('target_pos', target_pos)

        # takeoff
        for timestep in range(800):
            takeoff_data = takeoff_env.step(takeoff_data)
            takeoff_data = step_mdp(takeoff_data)
            
            takeoff_data = takeoff_policy(takeoff_data, deterministic=True)
            action = torch.tanh(takeoff_data[("agents", "action")])

            swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time


            if timestep == 400:
                target_pos = takeoff_env.drone_state[..., :3]
                target_pos[..., 2] = 1.0
                takeoff_env.target_pos = target_pos

        # # real policy rollout
        # for _ in range(500):
        #     takeoff_data = env.step(takeoff_data) 
        #     takeoff_data = step_mdp(takeoff_data)
            
        #     takeoff_data = policy(takeoff_data, deterministic=True)
        #     data_frame.append(takeoff_data.clone())
        #     action = torch.tanh(takeoff_data[("agents", "action")])

        #     swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre)

        #     cur_time = time.time()
        #     dt = cur_time - last_time
        #     # print('time', dt)
        #     last_time = cur_time

        # env.save_target_traj("8_1_demo.pt")
        # land
        for timestep in range(800):
            takeoff_data = takeoff_env.step(takeoff_data)
            takeoff_data = step_mdp(takeoff_data)

            takeoff_data = takeoff_policy(takeoff_data, deterministic=True)
            action = torch.tanh(takeoff_data[("agents", "action")])

            swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time

            if timestep == 400:
                takeoff_env.target_pos = torch.tensor([[0., 0., .5]])

            if timestep == 600:
                takeoff_env.target_pos = torch.tensor([[0., 0., .1]])
            

    swarm.end_program()
    
    torch.save(data_frame, "rl_data/turn_raw.pt")

if __name__ == "__main__":
    main()
