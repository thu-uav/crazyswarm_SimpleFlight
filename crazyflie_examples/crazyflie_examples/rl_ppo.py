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
from omni_drones.learning.ppo import PPORNNPolicy, PPOPolicy, PPOTConvPolicy 
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
    torch.manual_seed(cfg.seed)

    algos = {
        "mappo": MAPPOPolicy, 
        "ppo": PPOPolicy,
        "ppo_rnn": PPORNNPolicy,
        "ppo_tconv": PPOTConvPolicy,
    }

    swarm = Swarm(cfg)
    base_env = FakeHover(cfg, connection=True, swarm=swarm)
    base_env.eval()

    # load takeoff checkpoint
    takeoff_ckpt = "model/1128_mlp.pt"
    takeoff_env = FakeHover(cfg, connection=True, swarm=swarm)
    takeoff_agent_spec = takeoff_env.agent_spec["drone"]
    takeoff_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=takeoff_agent_spec, device=takeoff_env.device)
    takeoff_state_dict = torch.load(takeoff_ckpt)
    takeoff_policy.load_state_dict(takeoff_state_dict)
    
    # load checkpoint for deployment
    ckpt_name = "model/ppo_mlp/ppo_mlp.pt"
    base_env = env = FakeTrack(cfg, connection=True, swarm=swarm)
    agent_spec = env.agent_spec["drone"]
    ppo_config = cfg.algo_ppo
    policy = algos[cfg.algo_ppo.name.lower()](ppo_config, env=base_env, device=base_env.device)
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)

    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        data = env.reset().to(device=base_env.device)
        data = policy(data)

        takeoff_data = takeoff_env.reset().to(device=takeoff_env.device)
        takeoff_data = takeoff_policy(takeoff_data, deterministic=True)

        swarm.init()

        last_time = time.time()
        data_frame = []

        # update observation
        takeoff_env.target_pos = torch.tensor([[0., 0., 0.5]])

        # takeoff
        for timestep in range(300):
            takeoff_data = takeoff_env.step(takeoff_data)
            takeoff_data = step_mdp(takeoff_data)
            
            takeoff_data = takeoff_policy(takeoff_data, deterministic=True)
            action = torch.tanh(takeoff_data[("agents", "action")])

            swarm.act(action)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time

            if timestep == 150:
                takeoff_env.target_pos = torch.tensor([[0., 0., 1.]])

        # real policy rollout
        for _ in range(600):
            data = env.step(data) 
            data = step_mdp(data)
            
            data = policy(data)
            data_frame.append(data.clone())
            action = torch.tanh(data[("agents", "action")])
            swarm.act(action, rpy_scale=60)

            cur_time = time.time()
            dt = cur_time - last_time
            print('time', dt)
            last_time = cur_time

        # land
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
                takeoff_env.target_pos = torch.tensor([[0., 0., .5]])

            if timestep == 300:
                takeoff_env.target_pos = torch.tensor([[0., 0., .2]])

    swarm.end_program()
    
    torch.save(data_frame, "rl_data/track_ppo_mlp.pt")

if __name__ == "__main__":
    main()