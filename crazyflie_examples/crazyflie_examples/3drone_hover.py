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
from fake import FakeHover, FakeTrack, Swarm, Formation, MultiHover, FormationBallForward
import time

from crazyflie_py import Crazyswarm
from torchrl.envs.utils import step_mdp

REGULAR_TRIANGLE = [
    [0, 0, 0],
    [-0.5, 0.3, 0],
    [-0.5, -0.3, 0]
]

REGULAR_SQUARE = [
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
]

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="deploy_xyq")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    algos = {
        "mappo": MAPPOPolicy, 
    }

    swarm = Swarm(cfg, test=False)

    # load takeoff checkpoint
    takeoff_ckpt = "model/1128_mlp.pt"
    takeoff_env = MultiHover(cfg, connection=True, swarm=swarm)
    takeoff_agent_spec = takeoff_env.agent_spec["drone"]
    takeoff_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=takeoff_agent_spec, device=takeoff_env.device)
    takeoff_state_dict = torch.load(takeoff_ckpt)
    takeoff_policy.load_state_dict(takeoff_state_dict)

    formation_pos = torch.tensor(REGULAR_TRIANGLE)

    ckpt_name = "test_model/3drone/hover.pt"
    base_env = FormationBallForward(cfg, connection=True, swarm=swarm)
    # transforms = []
    # transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    # env = TransformedEnv(base_env, Compose(*transforms))
    env = base_env
    agent_spec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=base_env.device)
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)

    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        data = env.reset().to(dest=base_env.device)
        data = policy(data, deterministic=True)

        takeoff_data = takeoff_env.reset().to(dest=takeoff_env.device)
        takeoff_data = takeoff_policy(takeoff_data, deterministic=True)

        swarm.init()

        last_time = time.time()
        data_frame = []

        # takeoff
        takeoff_env.target_pos = formation_pos + torch.tensor([[0., 0, 0.5]]*swarm.num_cf)
        print(takeoff_env.target_pos)
        for timestep in range(500):
            takeoff_data = takeoff_env.step(takeoff_data)
            takeoff_data = step_mdp(takeoff_data)
            
            takeoff_data = takeoff_policy(takeoff_data, deterministic=True)
            action = torch.tanh(takeoff_data[("agents", "action")])
            swarm.act(action)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time

            # if timestep == 300:
            #     takeoff_env.target_pos = formation_pos + torch.tensor([0., 0, 1.]).expand_as(formation_pos)

        # real policy rollout
        for _ in range(100):
            data = env.step(data) 
            data = step_mdp(data)
            
            data = policy(data, deterministic=True)
            action = torch.tanh(data[("agents", "action")])

            swarm.act(action)
            data_frame.append(data.clone())

            cur_time = time.time()
            dt = cur_time - last_time
            print('time', dt)
            last_time = cur_time

        # land
        takeoff_env.target_pos = formation_pos + torch.tensor([[0., 0, 0.5]]*swarm.num_cf)
        for timestep in range(600):
            takeoff_data = takeoff_env.step(takeoff_data)
            takeoff_data = step_mdp(takeoff_data)

            takeoff_data = takeoff_policy(takeoff_data, deterministic=True)
            action = torch.tanh(takeoff_data[("agents", "action")])

            swarm.act(action)
            # data_frame.append(takeoff_data.clone())

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time

            # if timestep == 300:
            #     takeoff_env.target_pos = formation_pos + torch.tensor([0, 0, .5]).expand_as(formation_pos)

            if timestep == 400:
                takeoff_env.target_pos = formation_pos + torch.tensor([[0., 0, 0.2]]*swarm.num_cf)


    swarm.end_program()
    
    torch.save(data_frame, "rl_data/3drone_hover.pt")

if __name__ == "__main__":
    main()
