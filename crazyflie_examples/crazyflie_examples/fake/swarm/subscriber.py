import rclpy
from rclpy.node import Node
from crazyflie_interfaces.msg import LogDataGeneric
from tf2_msgs.msg import TFMessage

class Subscriber(Node):
    def __init__(self, name, call_back_pos, call_back_quat, call_back_vel):
        super().__init__('subscriber_'+name[1:])
        
        self.subscription_pos = self.create_subscription(
            LogDataGeneric,
            name + '/pos',
            call_back_pos,
            1)

        self.subscription_quat = self.create_subscription(
            LogDataGeneric,
            name + '/quat',
            call_back_quat,
            1)
        
        self.subscription_vel = self.create_subscription(
            LogDataGeneric,
            name + '/vel',
            call_back_vel,
            1)

class TFSubscriber(Node):
    def __init__(self, call_back):
        super().__init__('subscriber_tf')
        
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            call_back,
            1)