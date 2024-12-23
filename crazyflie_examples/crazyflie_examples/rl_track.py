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

    swarm = Swarm(cfg, test=False)


    cmd_fre = 100
    rpy_scale = 180
    min_thrust = 0.0
    max_thrust = 0.9
    use_LPF_action = False
    use_track = True

    # load takeoff checkpoint
    takeoff_ckpt = "model/hover/Hover.pt"
    # takeoff_ckpt = "model/1128_mlp.pt"
    takeoff_env = FakeHover(cfg, connection=True, swarm=swarm)
    takeoff_agent_spec = takeoff_env.agent_spec["drone"]
    takeoff_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=takeoff_agent_spec, device=takeoff_env.device)
    takeoff_state_dict = torch.load(takeoff_ckpt)
    takeoff_policy.load_state_dict(takeoff_state_dict)
    

    ckpt_name = "model/new_dynamics/inertia_add_30.pt"
    # ckpt_name = "model/datt/datt_mixed_traj.pt"
    base_env = env = FakeTrack(cfg, connection=True, swarm=swarm, dt=1.0 / cmd_fre)

    agent_spec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=base_env.device)
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)

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
        takeoff_env.target_pos = torch.tensor([[0.0, 0.0, 1.1]]) # 0.25T
        # takeoff_env.target_pos = torch.tensor([[0.8, -1.1, 1.1]]) # 0.25T

        takeoff_frame = []
        # takeoff
        for timestep in range(500):
            data = takeoff_env.step(data)
            data = step_mdp(data)
            
            data = takeoff_policy(data, deterministic=True)
            takeoff_frame.append(data.clone())
            action = torch.tanh(data[("agents", "action")])

            swarm.act(action, rpy_scale=180, rate=100)

            cur_time = time.time()
            dt = cur_time - last_time
            # print('time', dt)
            last_time = cur_time
        
        print('start pos', takeoff_env.drone_state[..., :3])

        # real policy rollout
        action_frame = []
        if use_LPF_action:
            last_action = action.clone()
            alpha = 0.05

        if use_track:
            for track_step in range(3500):
                data = base_env.step(data) 
                data = step_mdp(data)
                
                data = policy(data, deterministic=True)
                data_frame.append(data.clone())
                action = torch.tanh(data[("agents", "action")])
                action_frame.append(action)

                if use_LPF_action:
                    action = alpha * action + (1 - alpha) * last_action
                    last_action = action.clone()
                
                swarm.act(action, rpy_scale=rpy_scale, rate=cmd_fre, min_thrust=min_thrust, max_thrust=max_thrust)
                
                # setup prev_action, for prev_actions in obs
                target_rpy, target_thrust = action[:, 0, 0:3], action[:, 0, 3:]
                target_thrust = torch.clamp((target_thrust + 1) / 2, min=min_thrust, max=max_thrust)
                base_env.prev_actions = torch.concat([target_rpy, target_thrust], dim=-1)

                cur_time = time.time()
                dt = cur_time - last_time
                # print('time', dt)
                last_time = cur_time
            print('real policy done')

        takeoff_env.target_pos = torch.tensor([[0., 0., 1.0]])
        # takeoff_env.target_pos = torch.tensor([[0.8, -1.1, 1.1]]) # 0.25T
        print('target', target_pos)
        # land
        for timestep in range(600):
            data = takeoff_env.step(data)
            data = step_mdp(data)

            data = takeoff_policy(data, deterministic=True)
            action = torch.tanh(data[("agents", "action")])

            swarm.act(action, rpy_scale=180, rate=100)

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
    
    torch.save(data_frame, "sim2real_data/datt/star_slow.pt")

if __name__ == "__main__":
    main()
