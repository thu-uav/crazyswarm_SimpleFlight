from .rotate_utils import *
import numpy as np

class PID():
    
    def __init__(self, ctrl_freq=50, device='cpu'):
        self.g = 9.81
        
        self.kp_pos_xy = 10
        self.kp_pos_z = 11
        self.kd_pos_xy = 5
        self.kd_pos_z = 7
        self.ki_pos_xy = 1.5
        self.ki_pos_z = 1.1
        self.int_limit_xy = 1.3
        self.int_limit_z = 2.4
        self.kp_rot = 8
        self.kp_yaw = 9
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
        rpy = quat2euler(quat)
        yaw = rpy[..., 2]
        
        pos_err = pos - self.get_ref_pos(timestep)
        vel_err = vel - torch.zeros_like(vel)
        yaw_err = yaw - quat2euler(torch.zeros_like(quat))[..., 2]
        
        if torch.any(yaw_err >=  torch.pi):
            yaw_err_wraped = torch.where(yaw_err >= torch.pi, yaw_err - 2.0 * torch.pi, yaw_err)
        elif torch.any(yaw_err < -torch.pi):
            yaw_err_wraped = torch.where(yaw_err < -torch.pi, yaw_err + 2.0 * torch.pi, yaw_err)
        else:
            yaw_err_wraped = torch.where((yaw_err >= -torch.pi) & (yaw_err < torch.pi), yaw_err, yaw_err)
        
        if self.pos_err_int is None:
            self.pos_err_int = torch.zeros_like(pos)
        self.pos_err_int = self.pos_err_int + pos_err * self.dt
        self.pos_err_int[..., :2] = torch.clamp(self.pos_err_int[..., :2], -self.int_limit_xy, self.int_limit_xy)
        self.pos_err_int[...,  2] = torch.clamp(self.pos_err_int[...,  2], -self.int_limit_z,  self.int_limit_z )
        
        g_vec = torch.tensor([0, 0, self.g]).float()# .to(self.device)
        g_vec = g_vec.unsqueeze(0).expand_as(pos)
        
        z_vec = torch.tensor([0, 0, 1]).float()# .to(self.device)
        z_vec = z_vec.unsqueeze(0).expand_as(pos)
        
        acc_des = g_vec.clone()
        acc_des[..., :2] = acc_des[..., :2] \
            -self.kp_pos_xy * pos_err[..., :2] \
            -self.kd_pos_xy * vel_err[..., :2] \
            -self.ki_pos_xy * self.pos_err_int[..., :2]
        acc_des[..., 2] = acc_des[..., 2] \
            -self.kp_pos_z * pos_err[..., 2] \
            -self.kd_pos_z * vel_err[..., 2] \
            -self.ki_pos_z * self.pos_err_int[..., 2]
            
        u_des = inv_rotate_vector(acc_des, quat, mode='quat')
        acc_des = torch.norm(u_des, dim=-1, keepdim=True)
        
        rot_err = torch.cross(u_des / acc_des, z_vec, dim=-1)
        omega_des = - self.kp_rot * rot_err
        euler_feedback_des = torch.zeros_like(omega_des)
        euler_feedback_des[..., 2] = self.kp_yaw * yaw_err_wraped
        omega_des_yaw = omega_rotate_from_euler(euler_feedback_des, rpy)
        omega_des[..., 2] = omega_des[..., 2] + omega_des_yaw[..., 2]

        self.count += 1
        self.v_prev = vel
        
        thrust = torch.clamp(acc_des * self.thrust_scale, min=0., max=2**16)
        action = torch.cat([omega_des, thrust], dim=-1).unsqueeze(0)

        return action

    def get_ref_pos(self, t):
        moving_prog = np.clip(self.moving_vel * t * self.ctrl_dt, a_min=0, a_max=1.)
        ref_pos = self.init_pos + moving_prog * (self.target_pos - self.init_pos)
        return ref_pos
