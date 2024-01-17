"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from crazyflie_py import Crazyswarm
import numpy as np
import time
TAKEOFF_DURATION = 2.5
HOVER_DURATION = 2.5
HEIGHT = 1.0
SIDE = 0.5
H_OFFSET = 0.

HZ = 20
time_scaling = 1/100
fx = lambda t: np.sin(t*time_scaling)
fy = lambda t: np.cos(t*time_scaling)-1

timesteps = np.arange(0, 2*np.pi/time_scaling, 1/HZ)
def main():
    # time.sleep(5)
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]
    cf.takeoff(HEIGHT, 2)
    timeHelper.sleep(4)

    for timestep in timesteps:
        x = fx(timestep)
        y = fy(timestep)
        z = HEIGHT
        yaw = 0.
        cf.cmdPosition((x,y,z), yaw)
        timeHelper.sleepForRate(HZ)

    cf.notifySetpointsStop()
    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == '__main__':
    main()
