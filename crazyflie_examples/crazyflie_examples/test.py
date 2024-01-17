import torch
import numpy as np

# data = torch.load('rl_data/track_circle_medium.pt')
# print(len(data))

# # data = torch.stack(data)
# # print(data.mean())

# for i in range(300,301):
#     # action = data[i]['agents', 'action']
#     # action = torch.tanh(action)
#     # thrust = (action[..., 3] + 1) / 2
#     # print(thrust)
#     print(data[i])
#     # obs = data[i]['agents', 'observation']
#     # rpos = obs[..., :3]
#     # print(rpos)


# data = torch.load('rl_data/circle_remake2.pt')
# for i in range(200,205):
#     print(data[i]['agents', 'target_position'])
#     print(data[i]['agents', 'real_position'])
#     print(data[i]['agents', 'observation'][..., :3])

data = torch.load('rl_data/hoverdodge.pt')
# print(len(data))
# for i in range(5):
#     # print(data[i]['agents', 'observation'][..., :3])
#     print(torch.tanh(data[i]['agents', 'action']))
for i in range(100, 108):
    print(data[i]['agents', 'observation'][0])
    action = torch.tanh(data[i]['agents', 'action'])
    print((action+1)/2)
# data = torch.load('8_1_demo.pt')
# import csv
# with open('8_1_target.csv', 'w', newline='') as csvfile:
#     fieldnames = [
#         'pos.x',
#         'pos.y',
#         'pos.z',
#         ]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     for d in data:
#         writer.writerow({
#             'pos.x':d[0][0].item(),
#             'pos.y':d[0][1].item(),
#             'pos.z':d[0][2].item(),
#         })



# data = torch.load('rl_data/hover_cjy.pt')
# import csv
# with open('hover_cjy.csv', 'w', newline='') as csvfile:
#     fieldnames = [
#         'pos.x',
#         'pos.y',
#         'pos.z',
#         ]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     for d in data:
#         writer.writerow({
#             'pos.x':d['agents', 'observation'][0][0][0].item(),
#             'pos.y':d['agents', 'observation'][0][0][1].item(),
#             'pos.z':d['agents', 'observation'][0][0][2].item(),
#         })