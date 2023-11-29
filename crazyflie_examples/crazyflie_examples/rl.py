import logging
import os
import time

import hydra
import torch
import numpy as np
import wandb
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
from omni_drones.utils.wandb import init_wandb
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
from fake import FakeHover
import time

from crazyflie_py import Crazyswarm
import rclpy
from torchrl.envs.utils import step_mdp

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="deploy")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    algos = {
        "ppo": PPOPolicy,
        # "ppo_adaptive": PPOAdaptivePolicy,
        "ppo_rnn": PPORNNPolicy,
        "mappo": MAPPOPolicy, 
        # "happo": HAPPOPolicy,
        # "qmix": QMIXPolicy,
        # "dqn": DQNPolicy,
        # "sac": SACPolicy,
        # "td3": TD3Policy,
        # "matd3": MATD3Policy,
        # "tdmpc": TDMPCPolicy,
        # "test": Policy
    }

    base_env = FakeHover(cfg, headless=cfg.headless)

    def log(info):
        print(OmegaConf.to_yaml(info))
        run.log(info)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]

    transforms=[]

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=base_env.device)
    
    ckpt_name = "model/1128_mlp.pt"
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)

    # initialize real drone
    timeHelper = env.timeHelper
    cfs = env.cfs
    num_cf = env.num_cf

    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        data = env.reset().to(device=base_env.device)
        data = policy(data, deterministic=True)

        # cfs[0].takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
        # timeHelper.sleep(TAKEOFF_DURATION)

        print('start to deploy rl policy')

        # update observation
        if rclpy.ok():
            rclpy.spin_once(env.node) # pos
            rclpy.spin_once(env.node) # quat
            rclpy.spin_once(env.node) # vel

        # update observation
        data = env.step(data) 

        last_time = time.time()
        data_frame = []

        # set to CTBR mode
        for cf in cfs:
            cf.setParam("flightmode.stabModeRoll", 0)
            cf.setParam("flightmode.stabModePitch", 0)
            cf.setParam("flightmode.stabModeYaw", 0)

        # send several 0-thrust commands to prevent thrust deadlock
        for i in range(20):
            for cf in cfs:
                cf.cmdVel(0.,0.,0.,0.)
            timeHelper.sleepForRate(100)

        # real policy rollout
        for i in range(1000):
            
            data = policy(data, deterministic=True)
            data_frame.append(data)

            # update observation
            if rclpy.ok():
                rclpy.spin_once(env.node) # pos
                rclpy.spin_once(env.node) # quat
                rclpy.spin_once(env.node) # vel

            for id in range(num_cf):
                # TODO: automatically add tanh
                action = torch.tanh(data[("agents", "action")])
                action = action[0][id].cpu().numpy().astype(float)
                cf = cfs[id]
                thrust = (action[3] + 1) / 2
                print('thrust', thrust)
                thrust = float(max(0, min(0.99, thrust)))
                rpy_scale = 30
                # r = float(max(-0.5, min(0.5, action[0])))
                # p = float(max(-0.5, min(0.5, action[1])))
                # y = float(max(-0.5, min(0.5, action[2])))
                print('ctbr', thrust, action[0] * rpy_scale, action[1] * rpy_scale, action[2] * rpy_scale)
                cf.cmdVel(action[0] * rpy_scale, -action[1] * rpy_scale, action[2] * rpy_scale, thrust*2**16)
                # cf.cmdVel(r * 180, p * 180, y * 180, thrust*2**16)
                # cf.cmdVel(0.,0.,0., thrust*2**16)

            data = env.step(data)
            data = step_mdp(data)
            
            # print(data[('agents', 'observation')][0][0])

            timeHelper.sleepForRate(50)
            cur_time = time.time()
            dt = cur_time - last_time
            print('time', dt)
            last_time = cur_time

    # end program
    for i in range(20):
        for cf in cfs:
            cf.cmdVel(0., 0., 0., 0.)
        timeHelper.sleepForRate(100)
    wandb.finish()
    env.node.destroy_node()
    rclpy.shutdown()
    torch.save(data_frame, "data.pt")

if __name__ == "__main__":
    main()
