#!/usr/bin/env python

import numpy as np
from crazyflie_py import Crazyswarm

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3
import time
Z = 1.0

def goCircle(timeHelper, cf):
    cf.setParam("flightmode.stabModeRoll", 0)
    cf.setParam("flightmode.stabModePitch", 0)
    cf.setParam("flightmode.stabModeYaw", 0)
    # cf.takeoff(targetHeight=0.5, duration=2.5)
    # timeHelper.sleep(5.0)

    print("start stage one")

    for _ in range(100):
        cf.cmdVel(0., 0., 0., 0.63 * 2**16)
        timeHelper.sleep(0.01)

    for _ in range(100):
        cf.cmdVel(30., 30., 0., 0.6 * 2**16)
        timeHelper.sleep(0.01)

    print('start stage two')

    for i in range(150):
        cf.cmdVel(0., 0., 0., 0.6*2**16)
        timeHelper.sleep(0.01)
    for _ in range(20):
        cf.cmdVel(0., 0., 0., 0.)

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # allcfs.
    goCircle(timeHelper, allcfs.crazyflies[0])
