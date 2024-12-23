# Hardware Implementation of SimpleFlighe

## Training code: https://github.com/thu-uav/SimpleFlight

## Installation
A ROS 2-based stack for Bitcraze Crazyflie multirotor robots.

The documentation is available here: https://imrclab.github.io/crazyswarm2/.

## Crazyflie parameters
The cf[id] is the id you set for the crazyflie in the config file that is used in the launch file, i.e., crazyflies.yaml. Please set the correct cf[id] and uri for your flight.

## Deployment
We provide a stable RL policy for hovering at arbitrary points ``hover.pt`` and a policy for tracking trajectories ``deploy.pt``. You can also train your own RL policy by our training code.

In terminal 1, connect with the crazyflie
```
ros2 launch crazyflie launch.py backend:=cflib
```

In terminal 2, run the flight script.
```
cd crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples
```
For Figure-eight trajectories with different speeds,
```
python rl_track.py
```

For Polynomial, Pentagram and Zigzag trajectories,
```
python rl_arbitrary_track.py
```