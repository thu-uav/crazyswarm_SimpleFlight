"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from crazyflie_py import Crazyswarm


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0

trajectory = [
    {'goal': [0.5, 0.5, 0.4], 'yaw': 1.57},
    {'goal': [0.5, -0.5, 0.8], 'yaw': 3.14},
    {'goal': [-0.5, -0.5, 1.2], 'yaw': 4.71},
    {'goal': [-0.5, 0.5, 0.8], 'yaw': 0.0},
]


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies
    # cf = swarm.allcfs.crazyflies[0]
    n = len(cf)

    # takeoff
    for i in range(n):
        cf[i].takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    # # trajectory
    # for step in range(n):
    #     for i in range(4):
    #         s = (step + i) % 4
    #         cf[i].goTo(goal=trajectory[s]['goal'], yaw=trajectory[s]['yaw'], duration=HOVER_DURATION)
    #     timeHelper.sleep(HOVER_DURATION)

    # land
    for i in range(n):
        cf[i].land(targetHeight=0.04, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    # cf.takeoff(targetHeight=0.5, duration=TAKEOFF_DURATION)
    # timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    # cf.goTo(goal=[-0.2, -0.2, 0.2], yaw=0.0, duration=TAKEOFF_DURATION)
    # timeHelper.sleep(TAKEOFF_DURATION)
    # cf.goTo(goal=[0.2, -0.2, 0.4], yaw=1.57, duration=TAKEOFF_DURATION)
    # timeHelper.sleep(TAKEOFF_DURATION)
    # cf.goTo(goal=[0.2, 0.2, 0.6], yaw=3.14, duration=TAKEOFF_DURATION)
    # timeHelper.sleep(TAKEOFF_DURATION)
    # cf.goTo(goal=[-0.2, 0.2, 0.4], yaw=4.71, duration=TAKEOFF_DURATION)
    # timeHelper.sleep(TAKEOFF_DURATION)
    # cf.goTo(goal=[0.0, 0.0, 0.3], yaw=0.0, duration=TAKEOFF_DURATION)
    # timeHelper.sleep(TAKEOFF_DURATION)
    # cf.land(targetHeight=0.04, duration=2.5)
    # timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == '__main__':
    main()
