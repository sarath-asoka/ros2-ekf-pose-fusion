import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

from .pose_estimate import load_oxts, DRIVE_DIR, IDX_AF, IDX_WU

class SensorReplay(Node):
    def __init__(self):
        super().__init__("sensor_replay")
        self.frames, self.t = load_oxts(DRIVE_DIR)
        self.imu_pub = self.create_publisher(Imu, "imu/data", 10)
        self.k = 0
        self.create_timer(0.1, self.tick)# tick 10 times a second
        self.get_logger().info(f"loaded {len(self.frames)} frames")

    def tick(self):
        # if self.k >= len(self.frames):
        #     return
        f = self.frames[self.k]
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.linear_acceleration.x = float(f[IDX_AF])          # forward acceleration
        msg.angular_velocity.z    = float(f[IDX_WU])          # yaw rate
        self.imu_pub.publish(msg)
        # self.k += 1
        self.k = (self.k + 1) % len(self.frames) 



def main():
    rclpy.init()
    rclpy.spin(SensorReplay())
    rclpy.shutdown()