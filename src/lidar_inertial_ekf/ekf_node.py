import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from .pose_estimate import load_oxts, ekf_predict, DRIVE_DIR, IDX_YAW, IDX_VF, P0_DIAG


class EkfNode(Node):
    def __init__(self):
        super().__init__("ekf_node")
        frames, _ = load_oxts(DRIVE_DIR)                      # only to seed the start pose
        self.state = np.array([0.0, 0.0, frames[0, IDX_YAW], frames[0, IDX_VF]])
        self.P = np.diag(P0_DIAG ** 2)
        self.last_t = None

        self.create_subscription(Imu, "imu/data", self.on_imu, 10)
        self.odom_pub = self.create_publisher(Odometry, "ekf/odom", 10)
        self.tf = TransformBroadcaster(self)
        self.get_logger().info("ekf_node up: predicting from /imu/data")

    def on_imu(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_t is None:          # first message: no dt yet
            self.last_t = t
            return
        dt = t - self.last_t
        self.last_t = t
        if not (0.0 < dt < 1.0):         # skip the loop jump / bad gaps
            return

        self.state, self.P = ekf_predict(
            self.state, self.P, msg.linear_acceleration.x, msg.angular_velocity.z, dt)
        self.publish(msg.header.stamp)

    def publish(self, stamp):
        x, y, theta, v = self.state
        qz, qw = math.sin(theta / 2), math.cos(theta / 2)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link_ekf"
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link_ekf"
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf.sendTransform(tf)


def main():
    rclpy.init()
    rclpy.spin(EkfNode())
    rclpy.shutdown()