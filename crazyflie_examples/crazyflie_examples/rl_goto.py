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
from fake import FakeGoto, Swarm, FakeTurn, FakeLine
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
    # takeoff_ckpt = "model/hover/hover_targetrpy0_wosmooth.pt"
    takeoff_ckpt = "model/hover/Hover_rapid.pt"
    # takeoff_ckpt = "model/1128_mlp.pt"
    takeoff_env = FakeGoto(cfg, connection=True, swarm=swarm)
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

        check_pos = torch.tensor([[0.0, 0.15, 1.0]])

        # update observation
        target_pos = takeoff_env.drone_state[..., :3]
        target_pos[..., 2] = 1.0
        # takeoff_env.target_pos = target_pos
        takeoff_env.target_pos = check_pos

        # takeoff
        for timestep in range(500):
            data = takeoff_env.step(data)
            data = step_mdp(data)
            
            data = takeoff_policy(data, deterministic=True)
            action = torch.tanh(data[("agents", "action")])
            data_frame.append(data.clone())

            swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time
            # if timestep == 200:
            #     takeoff_env.target_pos = torch.tensor([[.5, .5, 1.0]])
            # if timestep == 400:
            #     takeoff_env.target_pos = torch.tensor([[.5, -.5, 1.0]])
            # if timestep == 600:
            #     takeoff_env.target_pos = torch.tensor([[-.5, -.5, 1.0]])
            # if timestep == 800:
            #     takeoff_env.target_pos = torch.tensor([[-.5, .5, 1.0]])

        print('start pos', takeoff_env.drone_state[..., :3])

        # # goto
        # length = 1.5
        # for timestep in range(int(800 * length)):
        #     data = takeoff_env.step(data)
        #     data = step_mdp(data)
            
        #     data = takeoff_policy(data, deterministic=True)
        #     action = torch.tanh(data[("agents", "action")])

        #     swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre)

        #     cur_time = time.time()
        #     dt = cur_time - last_time
        #     # print('time', dt)
        #     last_time = cur_time

        #     if timestep == int(150 * length):
        #         takeoff_env.target_pos = torch.tensor([[0., length, 1.0]])

        #     if timestep == int(300 * length):
        #         takeoff_env.target_pos = torch.tensor([[length, length, 1.0]])

        #     if timestep == int(450 * length):
        #         takeoff_env.target_pos = torch.tensor([[length, 0., 1.0]])

        #     if timestep == int(600 * length):
        #         takeoff_env.target_pos = torch.tensor([[0., 0., 1.0]])


        # env.save_target_traj("8_1_demo.pt")
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
                takeoff_env.target_pos = check_pos
                takeoff_env.target_pos[..., 2] = 1.0
                # takeoff_env.target_pos = torch.tensor([[0.0000,   0.0000, 1.0]])

            if timestep == 200:
                takeoff_env.target_pos[..., 2] = 0.8
                # takeoff_env.target_pos = torch.tensor([[0.0000,   0.0000, .8]])

            if timestep == 300:
                takeoff_env.target_pos[..., 2] = 0.6
                # takeoff_env.target_pos = torch.tensor([[0.0000,   0.0000, .6]])

            if timestep == 400:
                takeoff_env.target_pos[..., 2] = 0.4
                # takeoff_env.target_pos = torch.tensor([[0.0000,   0.0000, .4]])

            if timestep == 500:
                takeoff_env.target_pos[..., 2] = 0.2
                # takeoff_env.target_pos = torch.tensor([[0.0000,   0.0000, .2]])
        print('land pos', takeoff_env.drone_state[..., :3])


    swarm.end_program()
    
    torch.save(data_frame, "rl_data/goto_debug.pt")

if __name__ == "__main__":
    main()
