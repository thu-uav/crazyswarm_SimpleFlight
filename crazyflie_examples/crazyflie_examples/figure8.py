#!/usr/bin/env python

from pathlib import Path

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import numpy as np


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    traj1 = Trajectory()
    traj1.loadcsv(Path(__file__).parent / 'data/figure8.csv')
    # traj1.loadcsv(Path(__file__).parent / 'data/turn/drone.csv')

    # TODO set

    TRIALS = 1
    TIMESCALE = 1.0
    for i in range(TRIALS):
        for cf in allcfs.crazyflies:

            cf.uploadTrajectory(0, 0, traj1)
            cf.takeoff(targetHeight=1.0, duration=2.0)
            timeHelper.sleep(2.5)
            # pos = np.array(cf.initialPosition) + np.array([0., 0., 1.0])
            pos = np.array([0., 0., 1.0])
            cf.goTo(pos, 0, 2.0)
            timeHelper.sleep(2.5)
            cf.startTrajectory(0, timescale=TIMESCALE)
            timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)

            cf.land(targetHeight=0.06, duration=2.0)
            timeHelper.sleep(3.0)


if __name__ == '__main__':
    main()
