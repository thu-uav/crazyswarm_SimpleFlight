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
from fake import FakeHover, FakeTrack, Swarm
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

    base_env = FakeHover(cfg, connection=True, swarm=swarm)

    # load takeoff checkpoint
    takeoff_ckpt = "model/1128_mlp.pt"
    takeoff_env = FakeHover(cfg, connection=True, swarm=swarm)
    takeoff_agent_spec = takeoff_env.agent_spec["drone"]
    takeoff_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=takeoff_agent_spec, device=takeoff_env.device)
    takeoff_state_dict = torch.load(takeoff_ckpt)
    takeoff_policy.load_state_dict(takeoff_state_dict)
    
    # load checkpoint for deployment
    # ckpt_name = "model/test_model/origin_massrandom.pt"
    ckpt_name = "model/track_1130.pt"
    base_env = env = FakeTrack(cfg, connection=True, swarm=swarm)
    # ckpt_name = "model/1128_mlp.pt"
    # base_env = env = FakeHover(cfg, connection=True, swarm=swarm)
    agent_spec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=base_env.device)
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)

    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        data = env.reset().to(device=base_env.device)
        data = policy(data, deterministic=True)

        takeoff_data = takeoff_env.reset().to(device=takeoff_env.device)
        takeoff_data = takeoff_policy(takeoff_data, deterministic=True)

        swarm.init()

        last_time = time.time()
        data_frame = []

        # update observation
        takeoff_env.target_pos = torch.tensor([[0., 0., 0.5]])

        # takeoff
        for timestep in range(400):
            takeoff_data = takeoff_env.step(takeoff_data)
            takeoff_data = step_mdp(takeoff_data)
            
            takeoff_data = takeoff_policy(takeoff_data, deterministic=True)
            action = torch.tanh(takeoff_data[("agents", "action")])

            swarm.act(action)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time

            if timestep == 200:
                takeoff_env.target_pos = torch.tensor([[0., 0., 1.]])

        # real policy rollout
        for _ in range(1000):
            data = env.step(data) 
            data = step_mdp(data)
            
            data = policy(data, deterministic=True)
            data_frame.append(data.clone())
            action = torch.tanh(data[("agents", "action")])

            swarm.act(action, rpy_scale=60, rate=50)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time

        # env.save_target_traj("8_1_demo.pt")
        # land
        for timestep in range(800):
            takeoff_data = takeoff_env.step(takeoff_data)
            takeoff_data = step_mdp(takeoff_data)

            takeoff_data = takeoff_policy(takeoff_data, deterministic=True)
            action = torch.tanh(takeoff_data[("agents", "action")])

            swarm.act(action)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time

            if timestep == 250:
                takeoff_env.target_pos = torch.tensor([[0., 0., .5]])

            if timestep == 400:
                takeoff_env.target_pos = torch.tensor([[0., 0., .2]])
            
            if timestep == 650:
                takeoff_env.target_pos = torch.tensor([[0., 0., .1]])

    swarm.end_program()
    
    torch.save(data_frame, "rl_data/8_worandom_100Hz.pt")

if __name__ == "__main__":
    main()
