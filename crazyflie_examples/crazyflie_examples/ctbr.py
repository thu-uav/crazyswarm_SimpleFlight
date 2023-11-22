#!/usr/bin/env python

import numpy as np
from crazyflie_py import Crazyswarm

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3
import time
Z = 1.0

global vx, vy, vz

vx = 0.
vy = 0.
vz = 0.
class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber_0')
        
        self.subscription_pos = self.create_subscription(
            TFMessage,
            '/tf',
            self.call_back_pos,
            10)
        
        self.last_time_stamp = 0
        self.last_x = 0
        self.last_y = 0
        self.last_z = 0

    def call_back_pos(self, msg):
        # print(msg)
        now_time_stamp = float(msg.transforms[0].header.stamp.sec) + float(msg.transforms[0].header.stamp.nanosec) / 1000000000
        time = now_time_stamp - self.last_time_stamp
        dx = msg.transforms[0].transform.translation.x - self.last_x
        dy = msg.transforms[0].transform.translation.y - self.last_y
        dz = msg.transforms[0].transform.translation.z - self.last_z
        self.last_time_stamp = now_time_stamp
        self.last_x = msg.transforms[0].transform.translation.x
        self.last_y = msg.transforms[0].transform.translation.y
        self.last_z = msg.transforms[0].transform.translation.z
        global vx, vy, vz
        vx = float(dx) / time
        vy = float(dy) / time
        vz = float(dz) / time
        # print(time, vx, vy, vz)

def goCircle(timeHelper, cf):
    print("starting")
    subscriber = Subscriber()
    # while True:
    publisher = subscriber.create_publisher(Vector3, '/test_vel', 10)
    cf.setParam("flightmode.stabModeRoll", 0)
    cf.setParam("flightmode.stabModePitch", 0)
    cf.setParam("flightmode.stabModeYaw", 0)

    def get_vel():
        vec = Vector3()
        global vx, vy, vz
        vec.x = vx
        vec.y = vy
        vec.z = vz
        return vec

    for _ in range(20):
        cf.cmdVel(0., 0., 0., 0.)
        msg = get_vel()
        publisher.publish(msg)
        # rclpy.spin_once(subscriber)

    print("start stage one")

    for _ in range(100):
        cf.cmdVel(0., 0., 0., 0.63 * 2**16)
        msg = get_vel()
        publisher.publish(msg)
        # rclpy.spin_once(subscriber)
        timeHelper.sleep(0.01)

    for _ in range(100):
        cf.cmdVel(30., 30., 0., 0.6 * 2**16)
        msg = get_vel()
        publisher.publish(msg)
        # rclpy.spin_once(subscriber)
        timeHelper.sleep(0.01)

    print('start stage two')

    for i in range(150):
        cf.cmdVel(0., 0., 0., 0.6*2**16)
        msg = get_vel()
        publisher.publish(msg)
        # rclpy.spin_once(subscriber)
        timeHelper.sleep(0.01)
    for _ in range(20):
        cf.cmdVel(0., 0., 0., 0.)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # allcfs.
    goCircle(timeHelper, allcfs.crazyflies[0])
