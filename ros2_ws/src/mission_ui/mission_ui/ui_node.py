import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PolygonStamped
from custom_msgs.msg import BiasField


class MissionUiNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_ui")
        self.create_subscription(PoseArray, "/world/ugv_poses", self._ugv_cb, 10)
        self.create_subscription(PoseArray, "/world/uav_poses", self._uav_cb, 10)
        self.goal_pub = self.create_publisher(PolygonStamped, "/mission/goal_area", 10)
        self.bias_pub = self.create_publisher(BiasField, "/mission/bias_field", 10)

    def _ugv_cb(self, msg: PoseArray) -> None:
        self.get_logger().debug(f"UGV poses: {len(msg.poses)}")

    def _uav_cb(self, msg: PoseArray) -> None:
        self.get_logger().debug(f"UAV poses: {len(msg.poses)}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionUiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
