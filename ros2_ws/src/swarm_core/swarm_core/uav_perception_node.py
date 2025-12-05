import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped


class UavPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("uav_perception")
        self.detection_pub = self.create_publisher(PointStamped, "/uav/detections", 10)
        self.timer = self.create_timer(2.0, self._publish_placeholder_detection)

    def _publish_placeholder_detection(self) -> None:
        msg = PointStamped()
        msg.header.frame_id = "world"
        self.detection_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UavPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
