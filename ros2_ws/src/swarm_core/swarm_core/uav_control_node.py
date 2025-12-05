import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist
from custom_msgs.msg import UavCommandArray


class UavControlNode(Node):
    def __init__(self) -> None:
        super().__init__("uav_control")
        self.create_subscription(UavCommandArray, "/uav/high_level_cmd", self._cmd_cb, 10)
        self.state_pub = self.create_publisher(PoseArray, "/uav/state_array", 10)
        self.cmd_pub = self.create_publisher(Twist, "/airsim_node/cmd_vel", 10)
        self.timer = self.create_timer(1.0, self._publish_placeholder_state)

    def _cmd_cb(self, msg: UavCommandArray) -> None:
        self.get_logger().debug(f"Received {len(msg.commands)} UAV commands")

    def _publish_placeholder_state(self) -> None:
        self.state_pub.publish(PoseArray())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UavControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
