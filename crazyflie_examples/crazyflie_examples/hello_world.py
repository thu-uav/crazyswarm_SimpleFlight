"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from crazyflie_py import Crazyswarm
import numpy as np
import time
TAKEOFF_DURATION = 2.5
HOVER_DURATION = 2.5
HEIGHT = 1.0
SIDE = 0.5
H_OFFSET = 0.

def main():
    # time.sleep(5)
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)
    # cf.goTo(np.array([0., SIDE, HEIGHT]), 360, HOVER_DURATION)
    # timeHelper.sleep(HOVER_DURATION)
    # cf.goTo(np.array([SIDE, SIDE, HEIGHT]), 360, HOVER_DURATION)
    # timeHelper.sleep(HOVER_DURATION)
    # cf.goTo(np.array([SIDE, 0., HEIGHT]), 360, HOVER_DURATION)
    # timeHelper.sleep(HOVER_DURATION)
    # cf.goTo(np.array([0., 0., HEIGHT]), 360, HOVER_DURATION)
    # timeHelper.sleep(HOVER_DURATION)
    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == '__main__':
    main()
