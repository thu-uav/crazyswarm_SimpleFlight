import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
name = 'circle_remake2'
trajs = torch.load('rl_data/'+name+'.pt')
x = np.linspace(0, len(trajs), len(trajs))

# plot ctbr
plt.subplot(1,1,1)
actions = []
for i in range(len(trajs)):
    actions.append(trajs[i]['agents']['action'][0].cpu())
actions = torch.cat(actions, dim=0)
print(actions.shape)
actions = torch.tanh(actions)
actions = actions.numpy()
plt.plot(x,actions[..., 3], label='thrust')
plt.plot(x,actions[..., 0], label='r')
plt.plot(x,actions[..., 1], label='p')
plt.plot(x,actions[..., 2], label='y')
plt.legend()
plt.title("ctbr")

plt.savefig('rl_data/'+name)