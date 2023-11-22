#!/usr/bin/env python

from crazyflie_py import Crazyswarm


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # disable LED (one by one)
    for cf in allcfs.crazyflies:
        cf.setParam("led.bitmask", 128)
        cf.setParam("flightmode.stabModeRoll", 0)
        cf.setParam("flightmode.stabModePitch", 0)
        cf.setParam("flightmode.stabModeYaw", 0)
        timeHelper.sleep(1.0)

    # timeHelper.sleep(2.0)
    # print(cf.getParam("led."))

    # enable LED (broadcast)
    allcfs.setParam("led.bitmask", 0)
    # timeHelper.sleep(5.0)


if __name__ == '__main__':
    main()
