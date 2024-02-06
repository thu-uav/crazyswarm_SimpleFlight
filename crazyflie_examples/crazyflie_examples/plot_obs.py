import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
name = 'multi_hover_0204'
id=2
trajs = torch.load('rl_data/'+name+'.pt')
x = np.linspace(0, len(trajs), len(trajs))

fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(10,10))
fig.tight_layout()

# plot ctbr
for r in range(6):
    for c in range(4):
        i = c + r*4
        plt.subplot(6, 4, i+1)
        if i < 20:
            data = [trajs[_]['agents']['observation'][0, id, i].cpu() for _ in range(len(trajs))]
            plt.plot(x, data)
            plt.title("obs_"+str(i))
        else:
            data = [trajs[_]['agents']['action'][0, id, i-20].cpu() for _ in range(len(trajs))]
            plt.plot(x, data)
            plt.title("action_"+str(i-20))

plt.savefig('rl_data/'+name+"_obs_"+str(id))
