#!/usr/bin/env python

import numpy as np
from crazyflie_py import Crazyswarm

import time
Z = 1.0
sleepRate = 30


def goCircle(timeHelper, cf):
        # cf.takeoff(targetHeight=Z, duration=1.0+Z)
        # timeHelper.sleep(2 + Z)
        # startTime = timeHelper.time()
        # pos = cf.position()
        # startPos = cf.initialPosition + np.array([0, 0, Z])
        # center_circle = startPos - np.array([radius, 0, 0])
        # for _ in range(20):
        #     cf.cmdVel(0., 0., 0., 0.61)
        # for _ in range(50):
        #     cf.cmdVel(0., 0., 0., 0.75 * 2**16)
        #     timeHelper.sleep(0.01)
        for i in range(1000):
            # time = timeHelper.time() - startTime
            # omega = 2 * np.pi / totalTime
            # vx = -radius * omega * np.sin(omega * time)  
            # vy = radius * omega * np.cos(omega * time)
            # desiredPos = center_circle + radius * np.array(
            #     [np.cos(omega * time), np.sin(omega * time), 0])
            # errorX = desiredPos - cf.position() 
            # cf.cmdVelocityWorld(np.array([1, 1, 1]), yawRate=0)
            cf.cmdVel(0., 0., 1., 1.5*2**15)
            timeHelper.sleep(0.01)
            # timeHelper.sleepForRate(sleepRate)
        cf.land(targetHeight=0.03, duration=Z+1.0)
        timeHelper.sleep(Z+2.0)

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # allcfs.
    goCircle(timeHelper, allcfs.crazyflies[0])

if __name__ == "__main__":
    main()
