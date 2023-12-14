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

    algos = {
        "ppo": PPOPolicy,
        # "ppo_adaptive": PPOAdaptivePolicy,
        "ppo_rnn": PPORNNPolicy,
        "mappo": MAPPOPolicy, 
    }
    swarm = Swarm(cfg)
    base_env = FakeHover(cfg, connection=True, swarm=swarm)

    base_env.set_seed(cfg.seed)

    agent_spec: AgentSpec = base_env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=base_env.device)
    
    ckpt_name = "model/cjy.pt"
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)


    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        data = base_env.reset().to(device=base_env.device)
        data = policy(data, deterministic=True)

        print('start to deploy rl policy')

        # update observation
        data = base_env.step(data) 

        last_time = time.time()
        data_frame = []

        swarm.init()

        # real policy rollout
        for timestep in range(1000):
            
            data = policy(data, deterministic=True)
            action = torch.tanh(data[("agents", "action")])
            swarm.act(action)

            data = base_env.step(data)
            data = step_mdp(data)
            data_frame.append(data.clone())

            # if i == 300:
            #     base_env.target_pos = torch.tensor([[0., 1., .5]])

            # if i == 600:
            #     base_env.target_pos = torch.tensor([[0., 1., 1.]])

            # if i == 900:
            #     base_env.target_pos = torch.tensor([[0., 0., .5]])

            if timestep == 600:
                base_env.target_pos = torch.tensor([[0., 0., .5]])
            
            if timestep == 800:
                base_env.target_pos = torch.tensor([[0., 0., .2]])
            
            cur_time = time.time()
            dt = cur_time - last_time
            print('time', dt)
            last_time = cur_time

    swarm.end_program()
    torch.save(data_frame, "rl_data/hover_cjy.pt")

if __name__ == "__main__":
    main()
