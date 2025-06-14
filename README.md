# Hardware Implementation of SimpleFlight

## Training code: https://github.com/thu-uav/SimpleFlight

## Installation
First, install ROS 2 Galactic: https://docs.ros.org/en/galactic/Installation/Ubuntu-Install-Debians.html (for Ubuntu 20.04)

Then, set up your ROS 2 workspace using our crazyswarm_SimpleFlight
The documentation is available here: https://imrclab.github.io/crazyswarm2/installation.html. In step 3, replace the url of crazyswarm2 by this repository.

## SimpleFlight for deployment
The flight scripts require some of SimpleFlight's dependencies, so we have a simplified branch specifically for deployment, which can be downloaded from the repository at https://github.com/thu-uav/SimpleFlight.git. The branch is called ``deployment``. At the path /SimpleFlight, run
```
pip install -e .
```

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

## Modify crazyflie parameters
In ros2_ws/src/crazyswarm_SimpleFlight/crazyflie/config/crazyflies.yaml, the parameters are as follows:
```
cf1:
    enabled: true
    uri: radio://0/80/2M/E7E7E7E701
    initial_position: [0, 0, 0]
    type: cf21
```
Please set the correct cf[id] and uri for your flight.

## Modify crazyflie parameters
In ros2_ws/src/crazyswarm_SimpleFlight/crazyflie/config/motion_capture.yaml, the parameters are as follows:
```
/motion_capture_tracking:
  ros__parameters:
    type: "vrpn"
    hostname: "0.0.0.0" # IP address can be found on the motion capture computer
    mode: "motionCapture" 
```
Please set the correct hostname for your flight. We recommend a wired connection between the local computer used to send CTBR commands and the motion capture computer.