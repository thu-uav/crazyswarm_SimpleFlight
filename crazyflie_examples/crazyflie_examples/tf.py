import rclpy
from rclpy.node import Node

from motion_capture_tracking_interfaces.msg._named_pose import NamedPose  # noqa: F401
from motion_capture_tracking_interfaces.msg._named_pose_array import NamedPoseArray  # noqa: F401
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped, TransformStamped

import tf_transformations
from tf2_ros import TransformBroadcaster


class publisher(Node):
    def __init__(self):
        super().__init__('tf_pub')
        self.publisher_ = self.create_publisher(NamedPoseArray, '/poses', 10)
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.tfbr = TransformBroadcaster(self)
        
    def timer_callback(self):
        msg = NamedPoseArray()
        pose = NamedPose()
        cf_pose = Pose()
        pose.name = "cf8"
        pose.pose = cf_pose
        msg.poses = [pose]
        # msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        self.publisher_.publish(msg)
        
        t_base = TransformStamped()
        t_base.header.stamp = self.get_clock().now().to_msg()
        t_base.header.frame_id = 'world'
        t_base.child_frame_id = 'cf8'
        t_base.transform.translation.x = 0.
        t_base.transform.translation.y = 0.
        t_base.transform.translation.z = 0.
        t_base.transform.rotation.x = 0.
        t_base.transform.rotation.y = 0.
        t_base.transform.rotation.z = 0.
        t_base.transform.rotation.w = 1.
        self.tfbr.sendTransform(t_base)

        
        
rclpy.init()
pub = publisher()
rclpy.spin(pub)