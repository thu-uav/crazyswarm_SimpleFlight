import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D
import torch
name = 'debug'
data = torch.load('sim2real_data/'+name+'.pt')

agent1 = []
agent2 = []
agent3 = []
target = []
for frame in data:
    agent1.append(frame['agents', 'real_position'][0,0])
    agent2.append(frame['agents', 'real_position'][0,1])
    agent3.append(frame['agents', 'real_position'][0,2])
    target.append(frame['agents', 'target_position'][0,0])
agent1 = torch.stack(agent1).to('cpu')
agent2 = torch.stack(agent2).to('cpu')
agent3 = torch.stack(agent3).to('cpu')
target = torch.stack(target).to('cpu')

# 创建一个函数来生成每个时刻的图像
def generate_frame(t, agents):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, agent in enumerate(agents):
        ax.scatter(agent[t, 0], agent[t, 1], agent[t, 2], label=f'Agent {i+1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Time Step {t}')
    ax.legend()

    # 保存图像到临时文件
    filename = f'frame_{t}.png'
    plt.savefig(filename)
    plt.close(fig)
    return filename

# 生成所有帧并保存为图像
agents = [agent1, agent2, agent3, target]
frames = []
for t in range(len(agent1)):
    frame_filename = generate_frame(t, agents)
    frames.append(imageio.imread(frame_filename))

# 将所有帧拼接成GIF
imageio.mimsave('agents_movement.gif', frames, fps=10)

# 清理临时文件
import os
for t in range(len(agent1)):
    os.remove(f'frame_{t}.png')