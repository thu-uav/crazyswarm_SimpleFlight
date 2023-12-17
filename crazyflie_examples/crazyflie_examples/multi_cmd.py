"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from crazyflie_py import Crazyswarm
import numpy as np
import time
TAKEOFF_DURATION = 2.5
HOVER_DURATION = 2.5
HEIGHT = 1.0
SIDE = 0.5
H_OFFSET = 0.
rate = 50.0


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper

    for i in range(250):
        for cf in swarm.allcfs.crazyflies:
            cf.cmdPosition([0.,1.,HEIGHT*i/250])
        timeHelper.sleep(rate)

    for i in range(250):
        for cf in swarm.allcfs.crazyflies:
            cf.cmdPosition([0.,-1.,HEIGHT*(1-i/250)])
        timeHelper.sleep(rate)

if __name__ == '__main__':
    main()
