import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math
from nav_msgs.msg import Odometry
from .pose_estimate import latlon_to_xy, IDX_LAT, IDX_LON, IDX_YAW
from .pose_estimate import load_oxts, DRIVE_DIR, IDX_AF, IDX_WU

class SensorReplay(Node):
    def __init__(self):
        super().__init__("sensor_replay")
        self.frames, self.t = load_oxts(DRIVE_DIR)
        self.imu_pub = self.create_publisher(Imu, "imu/data", 10)
        self.k = 0
        self.create_timer(0.1, self.tick)# tick 10 times a second
        self.get_logger().info(f"loaded {len(self.frames)} frames")
        self.gt_x, self.gt_y = latlon_to_xy(self.frames[:, IDX_LAT], self.frames[:, IDX_LON])
        self.gt_pub = self.create_publisher(Odometry, "ground_truth/odom", 10)

    def tick(self):
        f = self.frames[self.k]
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.linear_acceleration.x = float(f[IDX_AF])          # forward acceleration
        msg.angular_velocity.z    = float(f[IDX_WU])          # yaw rate
        self.imu_pub.publish(msg)
    

        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = "odom"          # the fixed world frame
        odom.child_frame_id = "base_link"      # the car
        odom.pose.pose.position.x = float(self.gt_x[self.k])
        odom.pose.pose.position.y = float(self.gt_y[self.k])
        yaw = float(f[IDX_YAW])
        odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.gt_pub.publish(odom)

        self.k = (self.k + 1) % len(self.frames) 

def main():
    rclpy.init()
    rclpy.spin(SensorReplay())
    rclpy.shutdown()