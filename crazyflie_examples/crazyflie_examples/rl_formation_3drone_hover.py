import hydra
import torch
import numpy as np
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH #, init_simulation_app

from omni_drones.learning.ppo import PPORNNPolicy, PPOPolicy
from omni_drones.learning import (
    MAPPOPolicy, 
)

from setproctitle import setproctitle

from fake import FakeHover, FakeTrack, Swarm, Formation, MultiHover, FormationBall, FormationBallForward, PID
import time

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

LINE = [
    [-0.3, -0.3, 0],
    [0.3, 0.3, 0]
]

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="deploy_formation")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    algos = {
        "mappo": MAPPOPolicy, 
    }
    ckpt_name = "test_model/3drone_hover_rand.pt"
    data_frame = []

    debug = False
    swarm = Swarm(cfg, test=debug)
    base_env = FormationBallForward(cfg, connection=not debug, swarm=swarm)
    
    base_env.set_seed(cfg.seed)
    data = base_env.reset().to(dest=base_env.device)

    env = base_env
    agent_spec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=base_env.device)
    state_dict = torch.load(ckpt_name)
    policy.load_state_dict(state_dict)

    controller = PID(device=base_env.device)
    formation_pos = torch.tensor(REGULAR_TRIANGLE)
    formation_pos[..., 0] += -1.
    formation_pos[..., 1] += 1.
    formation_pos[..., 2] += 1.

    if debug:
        base_env.drone_state = torch.rand_like(base_env.drone_state)

    with torch.no_grad():
        # the first inference takes significantly longer time. This is a warm up
        data = env.reset().to(dest=base_env.device)
        data = policy(data, deterministic=True)

        # use PID controller to take off
        init_pos = base_env.drone_state[..., :3]
        target_pos = init_pos.clone()
        target_pos[..., 2] = 1.
        controller.set_pos(
            init_pos=init_pos,
            target_pos=target_pos
            )
        action = controller(base_env.drone_state, timestep=0)
        data['agents', 'action'] = action.to(base_env.device)
        data = base_env.step(data)
        data = step_mdp(data)
        
        init_pos = base_env.drone_state[..., :3]
        target_pos = formation_pos
        # target_pos = init_pos.clone()
        target_pos[..., 2] = 1.
        controller.set_pos(
            init_pos=init_pos,
            target_pos=target_pos
            )
        print("init pos", init_pos)
        print("target pos", target_pos)

        swarm.init()

        for i in range(200):
            action = controller(base_env.drone_state, timestep=i)
            data['agents', 'action'] = action.to(base_env.device)
            swarm.act_control(action)

            data = base_env.step(data)
            data = step_mdp(data)

        last_time = time.time()



        # # real policy rollout
        # for _ in range(200):
        #     data = env.step(data) 
        #     data = step_mdp(data)
            
        #     data = policy(data, deterministic=True)
        #     # data_frame.append(data.clone())
        #     action = torch.tanh(data[("agents", "action")])

        #     swarm.act(action,)
        #     data_frame.append(data.clone())

        #     cur_time = time.time()
        #     dt = cur_time - last_time
        #     print('time', dt)
        #     last_time = cur_time

        data_frame.append(data.clone())



    # use PID controller to land
    init_pos = base_env.drone_state[..., :3]
    target_pos = init_pos.clone()
    target_pos[..., 2] = 0.04
    controller.set_pos(
        init_pos=init_pos,
        target_pos=target_pos
        )
    print("init pos", init_pos)
    print("target pos", target_pos)

    for i in range(200):
        action = controller(base_env.drone_state, timestep=i)
        data['agents', 'action'] = action.to(base_env.device)
        swarm.act_control(action)

        data = base_env.step(data)
        data = step_mdp(data)

    swarm.end_program()
    if not debug:
        torch.save(data_frame, "rl_data/formation_pid_pos_3.pt")
        

if __name__ == "__main__":
    main()
