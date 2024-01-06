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
    # traj1.loadcsv(Path(__file__).parent / 'data/figure8.csv')
    traj1.loadcsv(Path(__file__).parent / 'data/rotate.csv')


    TRIALS = 1
    TIMESCALE = 1.0
    for i in range(TRIALS):
        for cf in allcfs.crazyflies:

            for _ in range(10):
                cf.cmdVel(0.0, 0.0, 0.0, 0.0)
            # timeHelper.sleep(1.0)
            
            # give cmd
            
            for _ in range(200):
                # thrust = 41000.0
                thrust = 43000.0
                action = [0.0, 0.0, 0.0]
                cf.cmdVel(action[0], -action[1], action[2], thrust)
                timeHelper.sleep(0.01)

            # for _ in range(200):
            #     thrust = 40000.0
            #     action = [0.0, 0.0, 90.0]
            #     cf.cmdVel(action[0], -action[1], action[2], thrust)
            #     timeHelper.sleep(0.01)

            for _ in range(2000):
                # thrust = 39500.0
                thrust = 42000.0
                action = [0.0, 0.0, 30.0]
                cf.cmdVel(action[0], -action[1], action[2], thrust)
                # timeHelper.sleep(0.01)
            

            # timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)

            # cf.land(targetHeight=0.06, duration=2.0)
            # timeHelper.sleep(3.0)


if __name__ == '__main__':
    main()
