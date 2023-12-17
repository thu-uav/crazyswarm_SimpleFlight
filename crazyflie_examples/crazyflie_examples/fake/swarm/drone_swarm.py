from crazyflie_py import Crazyswarm
from crazyflie_interfaces.msg import LogDataGeneric
import rclpy
import torch
from multiprocessing import Process
from rclpy.executors import MultiThreadedExecutor
from .subscriber import Subscriber
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

    def update_drone_pos(self, log, drone_state):
        drone_state[0][0] = log.values[0]
        drone_state[0][1] = log.values[1]
        drone_state[0][2] = log.values[2]

    def update_drone_quat(self, log, drone_state):
        drone_state[0][3] = log.values[0]
        drone_state[0][4] = log.values[1]
        drone_state[0][5] = log.values[2]
        drone_state[0][6] = log.values[3]

    def update_drone_vel(self, log, drone_state):
        drone_state[0][7] = log.values[0]
        drone_state[0][8] = log.values[1]
        drone_state[0][9] = log.values[2]

    def update_drone_pos_1(self, log, drone_state):
        drone_state[1][0] = log.values[0]
        drone_state[1][1] = log.values[1]
        drone_state[1][2] = log.values[2]

    def update_drone_quat_1(self, log, drone_state):
        drone_state[1][3] = log.values[0]
        drone_state[1][4] = log.values[1]
        drone_state[1][5] = log.values[2]
        drone_state[1][6] = log.values[3]

    def update_drone_vel_1(self, log, drone_state):
        drone_state[1][7] = log.values[0]
        drone_state[1][8] = log.values[1]
        drone_state[1][9] = log.values[2]

    def update_drone_pos_2(self, log, drone_state):
        drone_state[2][0] = log.values[0]
        drone_state[2][1] = log.values[1]
        drone_state[2][2] = log.values[2]

    def update_drone_quat_2(self, log, drone_state):
        drone_state[2][3] = log.values[0]
        drone_state[2][4] = log.values[1]
        drone_state[2][5] = log.values[2]
        drone_state[2][6] = log.values[3]

    def update_drone_vel_2(self, log, drone_state):
        drone_state[2][7] = log.values[0]
        drone_state[2][8] = log.values[1]
        drone_state[2][9] = log.values[2]
        
class Swarm():
    def __init__(self, cfg, test=False, mass=1.):
        self.cfg = cfg
        self.test = test
        if self.test:
            return
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.cfs = self.swarm.allcfs.crazyflies
        self.num_cf = len(self.cfs)
        self.drone_state = torch.zeros((self.num_cf, 16)) # position, velocity, quaternion, heading, up, relative heading
        self.drone_state[..., 3] = 1. # default rotation
        self.nodes = []
        self.drones = []
        id = 0
        self.mass = mass

        for cf in self.cfs:
            drone = FakeRobot(self.cfg.task, self.cfg.task.drone_model, device = cfg.sim.device, id=id)
            if id == 0:
                node = Subscriber(
                    cf.prefix, 
                    lambda x: drone.update_drone_pos(x, self.drone_state), 
                    lambda x: drone.update_drone_quat(x, self.drone_state), 
                    lambda x: drone.update_drone_vel(x, self.drone_state), 
                )
            elif id == 1:
                node = Subscriber(
                    cf.prefix, 
                    lambda x: drone.update_drone_pos_1(x, self.drone_state), 
                    lambda x: drone.update_drone_quat_1(x, self.drone_state), 
                    lambda x: drone.update_drone_vel_1(x, self.drone_state), 
                )
            elif id == 2:
                node = Subscriber(
                    cf.prefix, 
                    lambda x: drone.update_drone_pos_2(x, self.drone_state), 
                    lambda x: drone.update_drone_quat_2(x, self.drone_state), 
                    lambda x: drone.update_drone_vel_2(x, self.drone_state), 
                )
            self.drones.append(drone)
            self.nodes.append(node)
            id += 1

            # set to CTBR mode
            cf.setParam("flightmode.stabModeRoll", 0)
            cf.setParam("flightmode.stabModePitch", 0)
            cf.setParam("flightmode.stabModeYaw", 0)


    def get_drone_state(self):
        # update observation
        if rclpy.ok():
            for i in range(self.num_cf):
                rclpy.spin_once(self.nodes[i]) # pos
                rclpy.spin_once(self.nodes[i]) # quat
                rclpy.spin_once(self.nodes[i]) # vel
        return self.drone_state
    
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
            self.timeHelper.sleepForRate(100)

    def end_program(self):
        if self.test:
            return
        # end program
        for i in range(20):
            for cf in self.cfs:
                cf.cmdVel(0., 0., 0., 0.)
            self.timeHelper.sleepForRate(100)
        for i in range(self.num_cf):    
            self.nodes[i].destroy_node()
        rclpy.shutdown()
