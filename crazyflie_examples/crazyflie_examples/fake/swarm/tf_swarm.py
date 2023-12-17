from crazyflie_py import Crazyswarm
from crazyflie_interfaces.msg import LogDataGeneric
import rclpy
import torch
from multiprocessing import Process
from rclpy.executors import MultiThreadedExecutor
from .subscriber import TFSubscriber
from torchrl.data import CompositeSpec, TensorSpec, DiscreteTensorSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec


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
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.cfs = self.swarm.allcfs.crazyflies
        self.num_cf = len(self.cfs)
        self.drone_state = torch.zeros((self.num_cf, 16)) # position, velocity, quaternion, heading, up, relative heading
        self.num_obstacle = 1
        self.obstacle_state = torch.zeros((self.num_obstacle, 6)) # position, velocity
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
        last_pos = self.drone_state[...,:3].clone()
        last_obstacle = self.obstacle_state[..., :3].clone()
        for tf in log.transforms:
            time = tf.header.stamp.sec + tf.header.stamp.nanosec/1e9
            if tf.child_frame_id == "obs":
                self.obstacle_state[0][0] = tf.transform.translation.x
                self.obstacle_state[0][1] = tf.transform.translation.y
                self.obstacle_state[0][2] = tf.transform.translation.z
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
        self.drone_state[..., 7:10] = (self.drone_state[..., :3] - last_pos) / (time - self.last_time)
        self.obstacle_state[..., 3:6] = (self.obstacle_state[..., :3] - last_obstacle) / (time - self.last_time)
        self.last_time = time
        
    def get_drone_state(self):
        # update observation
        rclpy.spin_once(self.node) 
        return self.drone_state, self.obstacle_state
    
    def act(self, all_action, rpy_scale=30):
        if self.test:
            return
        for id in range(self.num_cf):
            action = all_action[0][id].cpu().numpy().astype(float)
            cf = self.cfs[id]
            thrust = (action[3] + 1) / 2 * self.mass
            thrust = float(max(0, min(0.9, thrust)))
            cf.cmdVel(action[0] * rpy_scale, -action[1] * rpy_scale, action[2] * rpy_scale, thrust*2**16)
        self.timeHelper.sleepForRate(50)

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
                cf.cmdVel(0., 0., 0., 0.)
            self.timeHelper.sleepForRate(50)  
        self.node.destroy_node()
        rclpy.shutdown()