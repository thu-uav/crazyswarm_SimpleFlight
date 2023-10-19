#!/usr/bin/env python

import numpy as np
from crazyflie_py import Crazyswarm

import rclpy
from rclpy.node import Node
# from tf2_msgs.msg import TFMessage
# from geometry_msgs.msg import Vector3
import time
Z = 1.0
from crazyflie_interfaces.msg import LogDataGeneric

global pos, vel

pos = 0.
vel = 0.

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber_0')
        
        self.subscription_vel = self.create_subscription(
            LogDataGeneric,
            '/cf4/vel',
            self.call_back_vel,
            10)
        
        self.subscription_pos = self.create_subscription(
            LogDataGeneric,
            '/cf4/pos',
            self.call_back_pos,
            10)

    def call_back_pos(self, msg):
        global pos
        pos = msg.values[2]

    def call_back_vel(self, msg):
        global vel
        vel = msg.values[2]

def goCircle(timeHelper, cf):
    print("starting")
    subscriber = Subscriber()
    # while True:
    # publisher = subscriber.create_publisher(Vector3, '/test_vel', 10)
    cf.setParam("flightmode.stabModeRoll", 0)
    cf.setParam("flightmode.stabModePitch", 0)
    cf.setParam("flightmode.stabModeYaw", 0)
    last_time = time.time()
    pos_gain = 4.
    vel_gain = 2.
    target_height = 1.

    for _ in range(20):
        cf.cmdVel(0., 0., 0., 0.)
        rclpy.spin_once(subscriber)
        now_time = time.time()
        print(now_time - last_time)
        last_time = now_time

    # for _ in range(100):
    #     cf.cmdVel(0., 0., 0., 0.75 * 2**16)

    # print("start test")

    for _ in range(300):
        global pos, vel
        print(pos, vel)
        pos_error = target_height - pos
        target_acc = (
            pos_gain * pos_error 
            + vel_gain * -vel 
            + 9.81
        )
        target_thrust = (target_acc * 0.04) / 0.15434568 / 4
        print("thrust", target_thrust)
        target_thrust = max(0., min(target_thrust, 0.9))
        cf.cmdVel(0., 0., 0., target_thrust * 2**16)
        rclpy.spin_once(subscriber)
        rclpy.spin_once(subscriber)
        timeHelper.sleep(0.01)
        now_time = time.time()
        print('time', now_time - last_time)
        last_time = now_time

    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # allcfs.
    goCircle(timeHelper, allcfs.crazyflies[0])
