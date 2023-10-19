"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

# from pycrazyswarm import Crazyswarm
from crazyflie_py import Crazyswarm
import pandas as pd
import os


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 1.5

cf_name = ['cf1', 'cf2', 'cf3', 'cf4']

cf_mapping = {'cf1': '1', 
              'cf2': '2',
              'cf3': '3',
              'cf4': '4',}

formation_name = 'square'

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper    
    cfs = swarm.allcfs.crazyflies

    # Take off
    for i in range(len(cf_name)):
        cfs[i].takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)
    
    # read trajecory data
    csv_path = 'formation/%s.csv' % formation_name
    traj_path = os.path.join(os.path.dirname(__file__), csv_path)
    traj_df = pd.read_csv(traj_path)
    traj = traj_df.to_dict()
    
    # Fly trajectory
    for step in range(len(traj['duration'])):
        duration = traj['duration'][step]
        for i in range(len(cf_name)):
            assert len(traj[cf_mapping[cf_name[i]]][step]) >= 4, "Missing trajectory parameters!"
            pos = traj[cf_mapping[cf_name[i]]][step][:3]
            yaw = traj[cf_mapping[cf_name[i]]][step][3]
            
            cfs[i].goTo(pos, yaw, duration)
            
        timeHelper.sleep(duration)

    # Land
    for i in range(len(cf_name)):
        cfs[i].land(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == "__main__":
    main()
