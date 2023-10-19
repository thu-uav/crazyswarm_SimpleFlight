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
    LogOnEpisode, 
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    History
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import (
    MAPPOPolicy, 
    HAPPOPolicy,
    QMIXPolicy,
    DQNPolicy,
    SACPolicy,
    TD3Policy,
    MATD3Policy,
    TDMPCPolicy,
    Policy,
    PPOPolicy,
    PPOAdaptivePolicy, PPORNNPolicy
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

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    algos = {
        "ppo": PPOPolicy,
        "ppo_adaptive": PPOAdaptivePolicy,
        "ppo_rnn": PPORNNPolicy,
        "mappo": MAPPOPolicy, 
        "happo": HAPPOPolicy,
        "qmix": QMIXPolicy,
        "dqn": DQNPolicy,
        "sac": SACPolicy,
        "td3": TD3Policy,
        "matd3": MATD3Policy,
        "tdmpc": TDMPCPolicy,
        "test": Policy
    }

    base_env = FakeHover(cfg, headless=cfg.headless)

    def log(info):
        print(OmegaConf.to_yaml(info))
        run.log(info)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=stats_keys,
        log_keys=stats_keys,
        logger_func=log,
    )
    transforms = [InitTracker(), logger]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # flatten it to use a MLP encoder instead
    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, "state"))
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
    ):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")]))

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device="cuda")
    
    ckpt_name = "checkpoints/checkpoint_takeoff_1017.pt"
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)

    # initialize real drone
    timeHelper = env.timeHelper
    cfs = env.cfs
    num_cf = env.num_cf

    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        data = env.reset()
        data = policy(data)

        # set to CTBR mode
        for cf in cfs:
            cf.setParam("flightmode.stabModeRoll", 0)
            cf.setParam("flightmode.stabModePitch", 0)
            cf.setParam("flightmode.stabModeYaw", 0)

        # send several 0-thrust commands to prevent thrust deadlock
        for i in range(20):
            for cf in cfs:
                cf.cmdVel(0., 0., 0., 0.)
            timeHelper.sleepForRate(100)
        #     cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
        # timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
        
        # update observation
        data = env.step(data) 

        # for _ in range(50):
        #     for cf in cfs:
        #         cf.cmdVel(0., 0., 0., 0.75 * 2**16)
        #     timeHelper.sleepForRate(100)

        last_time = time.time()

        for i in range(300):
            
            data = policy(data)

            # update observation
            if rclpy.ok():
                rclpy.spin_once(env.node) # pos
                rclpy.spin_once(env.node) # quat
                rclpy.spin_once(env.node) # vel

            for id in range(num_cf):
                action = data[("agents", "action")][0][id].cpu().numpy().astype(float)
                cf = cfs[id]
                thrust = action[3] / 0.28 * 0.4 / 4
                print('thrust', thrust)
                thrust = float(max(0, min(1, thrust)))
                # print('ctbr', thrust, action[0], action[1], action[2])
                cf.cmdVel(action[0] / 3.14 * 180, action[1] / 3.14 * 180, action[2] / 3.14 * 180, thrust*2**16)

            data = env.step(data)
            # print(data[('agents', 'observation')][0][0])

            timeHelper.sleepForRate(100)
            # cur_time = time.time()
            # dt = cur_time - last_time
            # print('time', dt)
            # last_time = cur_time

    wandb.finish()
    env.node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
