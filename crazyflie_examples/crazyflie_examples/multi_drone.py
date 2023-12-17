
from crazyflie_py import Crazyswarm
import numpy as np
import time
TAKEOFF_DURATION = 2.5
HOVER_DURATION = 2.5
HEIGHT = 1.0
SIDE = 0.5
H_OFFSET = 0.


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper

    for cf in swarm.allcfs.crazyflies:
        cf.takeoff(targetHeight=HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)

    for cf in swarm.allcfs.crazyflies:
        cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == '__main__':
    main()
