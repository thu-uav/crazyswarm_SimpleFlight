import torch
import numpy as np

data = torch.load('formation.pt')
print(len(data))

for i in range(len(data)):
    # action = data[i]
    action = data[i]['agents', 'action']
    action = torch.tanh(action)
    thrust = (action[..., 3] + 1) / 2
    print(thrust)
    if i > 200 and i < 240:
        print(data[i]['agents', 'observation'][..., :3])
    print(i, thrust)
    if i > 300:
        break
    # if i > 1090:
    #     print(data[i]['agents', 'observation'][..., :3])