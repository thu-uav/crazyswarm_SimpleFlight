from crazyflie_py import Crazyswarm
from crazyflie_interfaces.msg import LogDataGeneric
import rclpy
import torch
from multiprocessing import Process
from rclpy.executors import MultiThreadedExecutor
from .subscriber import TFSubscriber
from torchrl.data import CompositeSpec, TensorSpec, DiscreteTensorSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec
from omni_drones.utils.torch import quaternion_to_euler
import numpy as np

class FakeRobot():
    def __init__(self, cfg, name, device, id):
        self.name = name
        self.device = device
        self.cfg = cfg
        if name == "Hummingbird":
            self.num_rotors = 4
        elif name == "Crazyflie" or "crazyflie":
            self.num_rotors = 4
        elif name == "Firefly":
            self.num_rotors = 6

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        self.intrinsics_spec = CompositeSpec({
            "mass": UnboundedContinuousTensorSpec(1),
            "inertia": UnboundedContinuousTensorSpec(3),
            "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            "KM": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_up": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_down": UnboundedContinuousTensorSpec(self.num_rotors),
            "drag_coef": UnboundedContinuousTensorSpec(1),
            "rotor_offset": UnboundedContinuousTensorSpec(1),
        }).to(self.device)

        if self.cfg.force_sensor:
            self.use_force_sensor = True
            state_dim = 19 + self.num_rotors + 3
        else:
            self.use_force_sensor = False
            state_dim = 19 + self.num_rotors
        self.state_spec = UnboundedContinuousTensorSpec(state_dim, device=self.device)

        self.n = 1
        self.id = id

class Swarm():
    def __init__(self, cfg, test=False, mass=1.):
        self.cfg = cfg
        self.test = test
        if self.test:
            self.num_cf = 3
            return
        self.log=None
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.cfs = self.swarm.allcfs.crazyflies
        self.num_cf = len(self.cfs)
        self.drone_state = torch.zeros((self.num_cf, 19)) # position, quaternion, velocity, omega, heading, up
        self.num_ball = cfg.task.ball_num
        self.ball_state = torch.zeros((self.num_ball, 6)) # position, velocity
        self.num_static_obstacle = cfg.task.static_obs_num
        self.obstacle_state = torch.zeros((self.num_static_obstacle, 3)) # position
        self.drone_state[..., 3] = 1. # default rotation
        self.drones = []
        self.node = TFSubscriber(
            self.update_drone_state
        )
        self.last_time = 0.
        self.mass = mass
        
        id = 0
        self.cf_map = {}
        for cf in self.cfs:
            drone = FakeRobot(self.cfg.task, self.cfg.task.drone_model, device = cfg.sim.device, id=id)
            self.drones.append(drone)
            self.cf_map[cf.prefix[1:]] = id
            id += 1

            # set to CTBR mode
            cf.setParam("flightmode.stabModeRoll", 0)
            cf.setParam("flightmode.stabModePitch", 0)
            cf.setParam("flightmode.stabModeYaw", 0)

    def update_drone_state(self, log):
        self.log = log

    def get_drone_state(self):
        # update observation
        rclpy.spin_once(self.node)
        if self.log is not None:
            last_pos = self.drone_state[...,:3].clone()
            last_quat = self.drone_state[...,3:7].clone()
            last_rpy = quaternion_to_euler(last_quat)
            if self.num_ball > 0:
                last_ball = self.ball_state[..., :3].clone()
            for tf in self.log.transforms:
                time = tf.header.stamp.sec + tf.header.stamp.nanosec/1e9
                if tf.child_frame_id == "ball":
                    ball_id = int(tf.child_frame_id[4:])
                    self.ball_state[ball_id][0] = tf.transform.translation.x
                    self.ball_state[ball_id][1] = tf.transform.translation.y
                    self.ball_state[ball_id][2] = tf.transform.translation.z
                if tf.child_frame_id == "obs": 
                    obs_id = int(tf.child_frame_id[3:])
                    self.obstacle_state[obs_id][0] = tf.transform.translation.x
                    self.obstacle_state[obs_id][1] = tf.transform.translation.y
                    self.obstacle_state[obs_id][2] = 1.5 #tf.transform.translation.z
                if tf.child_frame_id not in self.cf_map.keys():
                    continue
                drone_id = self.cf_map[tf.child_frame_id]
                self.drone_state[drone_id][0] = tf.transform.translation.x
                self.drone_state[drone_id][1] = tf.transform.translation.y
                self.drone_state[drone_id][2] = tf.transform.translation.z
                self.drone_state[drone_id][3] = tf.transform.rotation.w
                self.drone_state[drone_id][4] = tf.transform.rotation.x
                self.drone_state[drone_id][5] = tf.transform.rotation.y
                self.drone_state[drone_id][6] = tf.transform.rotation.z
                if self.drone_state[drone_id][3] < 0:
                    self.drone_state[drone_id][..., 3:7] *= -1
            self.drone_state[..., 7:10] = (self.drone_state[..., :3] - last_pos) / (time - self.last_time)
            curr_rpy = quaternion_to_euler(self.drone_state[..., 3:7])
            self.drone_state[..., 10:13] = (curr_rpy - last_rpy) / (time - self.last_time)
            if self.num_ball > 0:
                self.ball_state[..., 3:6] = (self.ball_state[..., :3] - last_ball) / (time - self.last_time)
            self.last_time = time

        return self.drone_state.clone(), self.ball_state.clone(), self.obstacle_state.clone()
    
    def act(self, all_action, rpy_scale=180, rate=100):
        if self.test:
            return
        for id in range(self.num_cf):
            action = all_action[0][id].cpu().numpy().astype(float)
            cf = self.cfs[id]
            thrust = (action[3] + 1) / 2 * self.mass
            thrust = float(max(0, min(0.9, thrust)))
            cf.cmdVel(action[0] * rpy_scale, -action[1] * rpy_scale, action[2] * rpy_scale, thrust*2**16)
        self.timeHelper.sleepForRate(rate)
    
    def act_control(self, all_action, rate=100):
        if self.test:
            return
        for id in range(self.num_cf):
            action = all_action[0][id].cpu().numpy().astype(float)
            cf = self.cfs[id]
            roll_rate = float(np.clip(action[0] * 180. / torch.pi, -200., 200.))
            pitch_rate = float(np.clip(action[1] * 180. / torch.pi, -200., 200.))
            yaw_rate = float(np.clip(action[2] * 180. / torch.pi, -100., 100.))
            thrust = (action[3] + 1) / 2 * self.mass
            thrust = float(max(0, min(0.9, thrust)))
            cf.cmdVel(roll_rate, -pitch_rate, yaw_rate, thrust*2**16)
        self.timeHelper.sleepForRate(rate)

    # # give cmd and act
    # def cmd_act(self, action, rate=50):
    #     if self.test:
    #         return
    #     for id in range(self.num_cf):
    #         # action = all_action[0][id].cpu().numpy().astype(float)
    #         cf = self.cfs[id]
    #         # thrust = 43300.0
    #         thrust = 45000.0
    #         cf.cmdVel(action[0], -action[1], action[2], thrust)
    #     self.timeHelper.sleepForRate(rate)

    # give cmd and act
    def cmd_act(self, action, rate=50):
        if self.test:
            return
        for id in range(self.num_cf):
            # action = all_action[0][id].cpu().numpy().astype(float)
            cf = self.cfs[id]
            # thrust = 43300.0
            thrust = 45000.0
            cf.cmdVel(action[0], -action[1], action[2], thrust)
        self.timeHelper.sleepForRate(rate)

    def init(self):
        if self.test:
            return
        # send several 0-thrust commands to prevent thrust deadlock
        for i in range(20):
            for cf in self.cfs:
                cf.cmdVel(0.,0.,0.,0.)
            self.timeHelper.sleepForRate(50)

    def end_program(self):
        if self.test:
            return
        # end program
        for i in range(20):
            for cf in self.cfs:
                cf.notifySetpointsStop()
            self.timeHelper.sleepForRate(50)  
        self.node.destroy_node()
        rclpy.shutdown()