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
from fake import FakeHover, FakeTrack
import time

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
        "mappo": MAPPOPolicy, 
    }

    # base_env = FakeTrack(cfg, connection=False, swarm=0)
    # ckpt_name = "model/track_1130.pt"

    ckpt_name = "model/1128_mlp.pt"
    base_env = env = FakeHover(cfg, connection=True, swarm=0)

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
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device="cuda")
    
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)

    num_cf = env.num_cf

    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        data = env.reset().to(device=base_env.device)
        data = policy(data, deterministic=True)

        # update observation
        data = env.step(data) 

        last_time = time.time()
        for i in range(15):
            
            data = policy(data, deterministic=True)

            for id in range(num_cf):
                action = torch.tanh(data[("agents", "action")])
                action = action[0][id].cpu().numpy().astype(float)
                thrust = (action[3] + 1) / 2
                print('thrust', thrust)
                thrust = float(max(0, min(1, thrust)))
                print('ctbr', thrust, action[0] * 180, action[1] * 180, action[2] * 180)

            data = env.step(data)
            data = step_mdp(data)
            # print(data[('agents', 'observation')][0][0])

            cur_time = time.time()
            dt = cur_time - last_time
            print('time', dt)
            last_time = cur_time

    wandb.finish()

if __name__ == "__main__":
    main()