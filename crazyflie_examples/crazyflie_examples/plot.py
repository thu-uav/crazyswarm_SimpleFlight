import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

import torch
data = torch.load('rl_data/8_remake.pt')

x = []
y = []
z = []
error = []
target_x = []
target_y = []
target_z = []
cnt = 0
for frame in data:
    cnt += 1
    if cnt < 300:
        continue
    target_x.append(frame['agents', 'target_position'][0][0].cpu().item())
    target_y.append(frame['agents', 'target_position'][0][1].cpu().item())
    target_z.append(frame['agents', 'target_position'][0][2].cpu().item())
    x.append(frame['agents', 'real_position'][0][0].cpu().item())
    y.append(frame['agents', 'real_position'][0][1].cpu().item())
    z.append(frame['agents', 'real_position'][0][2].cpu().item())
    e = torch.norm(frame['agents', 'observation'][0, 0, :3]).cpu().item()
    error.append(e)

error = np.array(error)
mean_e = np.mean(error)
error = 0.5 - (error - np.min(error)) / (np.max(error) - np.min(error)) / 2

color_map = get_cmap('gist_rainbow')
colors = color_map(error)
ax.scatter(x, y, z, s=5, c=colors)
ax.set_zlim3d(0.,1.1)
ax.plot(target_x, target_y, target_z)
plt.savefig('8_remake_'+str(mean_e) + '.png')