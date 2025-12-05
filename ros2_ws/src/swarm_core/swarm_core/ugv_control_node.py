import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist


class UgvControlNode(Node):
    def __init__(self) -> None:
        super().__init__("ugv_control")
        self.create_subscription(PoseArray, "/ugv/goal_points", self._goal_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/robot_0/cmd_vel", 10)
        self.state_pub = self.create_publisher(PoseArray, "/ugv/state_array", 10)
        self.timer = self.create_timer(1.0, self._publish_placeholder_state)

    def _goal_cb(self, msg: PoseArray) -> None:
        self.get_logger().debug(f"Received {len(msg.poses)} UGV goals")

    def _publish_placeholder_state(self) -> None:
        self.state_pub.publish(PoseArray())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UgvControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
