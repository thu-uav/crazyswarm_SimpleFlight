from .rotate_utils import *
import numpy as np
from crazyflie_py import Crazyswarm

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3

class PID():
    
    def __init__(self, ctrl_freq=50, device='cpu'):
        self.g = 9.81
        
        self.kp_pos = 6
        self.kd_pos = 4
        self.ki_pos = 1.2
        self.kp_rot = 9.375
        self.yaw_gain = 13.75
        self.kp_ang = 16
        
        self.ctrl_freq = ctrl_freq
        self.ctrl_dt = 1. / self.ctrl_freq
        
        self.pos_err_int = None
        self.count = 0
        self.v_prev = None
        self.dt = self.ctrl_dt
        self.thrust_scale = 43300. / 9.81

        self.moving_vel = 0.5
        self.device = device

    def set_pos(self, init_pos, target_pos):
        self.init_pos = init_pos
        self.target_pos = target_pos
        
    def __call__(self, drone_state, timestep=0):
        pos = drone_state[..., :3]
        quat = drone_state[..., 3:7]
        vel = drone_state[..., 7:10]
        yaw = quat2euler(quat)[..., 2]
        
        pos_err = pos - self.get_ref_pos(timestep)
        vel_err = vel - torch.zeros_like(vel)
        yaw_err = yaw - quat2euler(torch.zeros_like(quat))[..., 2]
        
        if self.pos_err_int is None:
            self.pos_err_int = torch.zeros_like(pos)
        self.pos_err_int = self.pos_err_int + pos_err * self.dt
        
        g_vec = torch.tensor([0, 0, self.g]).float()# .to(self.device)
        g_vec = g_vec.unsqueeze(0).expand_as(pos)
        
        z_vec = torch.tensor([0, 0, 1]).float()# .to(self.device)
        z_vec = z_vec.unsqueeze(0).expand_as(pos)
        
        acc_des = g_vec \
            - self.kp_pos * pos_err \
            - self.kd_pos * vel_err \
            - self.ki_pos * self.pos_err_int \
            + 0.5 * torch.zeros_like(pos)
            
        u_des = inv_rotate_vector(acc_des, quat, mode='quat')
        acc_des = torch.norm(u_des, dim=-1, keepdim=True)
        
        rot_err = torch.cross(u_des / acc_des, z_vec)
        omega_des = - self.kp_rot * rot_err
        omega_des[..., 2] = self.yaw_gain * yaw_err
        
        self.count += 1
        self.v_prev = vel
        
        thrust = torch.clamp(acc_des * self.thrust_scale, min=0., max=2**16)
        roll_rate = omega_des[..., 0] #* 180. / torch.pi
        pitch_rate = omega_des[..., 1] #* 180. / torch.pi
        yaw_rate = omega_des[..., 2] #* 180. / torch.pi

        action = torch.cat([roll_rate.unsqueeze(-1), pitch_rate.unsqueeze(-1), yaw_rate.unsqueeze(-1), thrust], dim=-1).unsqueeze(0)

        # print(action)

        return action

    def get_ref_pos(self, t):
        moving_prog = np.clip(self.moving_vel * t * self.ctrl_dt, a_min=0, a_max=1.)
        ref_pos = self.init_pos + moving_prog * (self.target_pos - self.init_pos)
        return ref_pos
