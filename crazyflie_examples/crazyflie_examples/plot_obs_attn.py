import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
name = 'formation_ball_3_12_2'
id = 0
trajs = torch.load('rl_data/'+name+'.pt')
x = np.linspace(0, len(trajs), len(trajs))
fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(10,10))
fig.tight_layout()

# plot ctbr
for r in range(10):
    for c in range(5):
        i = c + r*5
        plt.subplot(10, 5, i+1)
        if i < 25:
            data = [trajs[_]['agents']['observation']['obs_self'][0, id, 0, i].cpu() for _ in range(len(trajs))]
            plt.plot(x, data)
            plt.title("obs_self_"+str(i))
        elif i >= 25 and i < 32:
            data = [trajs[_]['agents']['observation']['obs_others'][0, id, 0, i-25].cpu() for _ in range(len(trajs))]
            plt.plot(x, data)
            plt.title("obs_others_"+str(i-25))
        elif i >= 32 and i < 39:
            data = [trajs[_]['agents']['observation']['obs_others'][0, id, 1, i-32].cpu() for _ in range(len(trajs))]
            plt.plot(x, data)
            plt.title("obs_others_"+str(i-25))
        # elif i >= 39 and i < 32:
        #     data = [trajs[_]['agents']['observation']['obs_others'][0, id, 0, i-25].cpu() for _ in range(len(trajs))]
        #     plt.plot(x, data)
        #     plt.title("obs_others_"+str(i-25))          
        # else:
        #     data = [trajs[_]['agents']['action'][0, id, i-20].cpu() for _ in range(len(trajs))]
        #     plt.plot(x, data)
        #     plt.title("action_"+str(i-20))
        elif i >= 39 and i < 49:
            data = [trajs[_]['agents']['observation']['attn_obs_obstacles'][0, id, 0, i-39].cpu() for _ in range(len(trajs))]
            plt.plot(x, data)
            plt.title("obstacle_"+str(i-39))         

plt.savefig('rl_data/'+name+"_obs_"+str(id))
