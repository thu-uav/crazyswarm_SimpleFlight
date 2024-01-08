import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration

from motion_capture_tracking_interfaces.msg import NamedPoseArray

from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.cm import get_cmap

import csv
import time

CMAPS = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greys']

class SubscriberNode(Node):
    """
    A ROS2 node that subscribes to the motion capture lib's poses topic
    and plots the position of the agents in 3D online
    and saves the data to a csv file    
    """

    def __init__(self, name, num_agents=1, filename="data.csv"):
        super().__init__(name)

        csv_file = open(filename, 'w')
        self.csv_writer = csv.writer(csv_file)
        self.csv_writer.writerow(["time"] + [f"agent_{i}_{a}" for i in range(num_agents) 
                                             for a in ["name", "x", "y", "z", "qx", "qy", "qz", "qw"]])

        plt.ion()
        assert num_agents <= len(CMAPS), "Too many agents for color maps"
        self.color_maps = [get_cmap(cmap) for cmap in CMAPS]

        self.agent_names = []
        self.agents_pos = []
        for i in range(num_agents):
            self.agents_pos.append([np.array([]), np.array([]), np.array([])])

        self.start_time = time.time()
        self.times = np.array([])
        
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                    history=QoSHistoryPolicy.KEEP_LAST,
                                    depth=1,
                                    deadline=Duration(seconds=0, nanoseconds=1e9/100.0))
        self.create_subscription(
            NamedPoseArray, "/poses",
            self._poses_changed, qos_profile
        )


    def _poses_changed(self, msg):
        """
        Topic update callback to the motion capture lib's poses topic
        to plot the position of the agents in 3D online
        and saves the data to a csv file
        """
        plt.clf()
        fig = plt.gcf()
        ax = fig.gca(projection='3d')
        current_time = time.time() - self.start_time
        data_row = [current_time]

        poses = msg.poses
        for i in range(len(poses)):
            pose = poses[i]
            name = pose.name
            x = pose.pose.position.x
            y = pose.pose.position.y
            z = pose.pose.position.z
            qx = pose.pose.orientation.x
            qy = pose.pose.orientation.y
            qz = pose.pose.orientation.z
            qw = pose.pose.orientation.w 

            data_row.extend([name, x, y, z, qx, qy, qz, qw])

            self.times = np.append(self.times, current_time)
            self.agent_names.append(name)
            self.agents_pos[i][0] = np.append(self.agents_pos[i][0], x)
            self.agents_pos[i][1] = np.append(self.agents_pos[i][1], y)
            self.agents_pos[i][2] = np.append(self.agents_pos[i][2], z)

            ax.plot(xs=self.agents_pos[i][0], ys=self.agents_pos[i][1], zs=self.agents_pos[i][2], 
                    # c=self.color_maps[i](self.times / current_time)
                    )
        
        plt.pause(0.001)
        plt.ioff()
        self.csv_writer.writerow(data_row)


def csv2gif(filename="data.csv"):
    """
    Reads the csv file and plots the position of the agents in 3D
    and saves the data to a gif file
    """
    data = np.genfromtxt(filename, delimiter=',', names=True)
    times = data["time"]
    agent_names = [name for name in data.dtype.names if name != "time"]
    num_agents = len(agent_names) // 8
    agents_pos = []
    for i in range(num_agents):
        agents_pos.append([data[f"agent_{i}_x"], data[f"agent_{i}_y"], data[f"agent_{i}_z"]])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def update_graph(i):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        for j in range(num_agents):
            ax.plot(xs=agents_pos[j][0][:i], ys=agents_pos[j][1][:i], zs=agents_pos[j][2][:i], 
                    # c=get_cmap(CMAPS[j])(times[:i] / times[-1])
                    )
        return ax

    ani = animation.FuncAnimation(fig, update_graph, len(times), interval=1, blit=False)
    ani.save('data.gif', writer=animation.PillowWriter())


def main(args=None):
    rclpy.init(args=args)
    node = SubscriberNode("plot_optitrack")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    # main()
    csv2gif()
