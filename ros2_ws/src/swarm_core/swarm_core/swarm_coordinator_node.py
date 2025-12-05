import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from custom_msgs.msg import BiasField, UavCommandArray
from swarm_core.world_model import WorldModel


class SwarmCoordinatorNode(Node):
    def __init__(self) -> None:
        super().__init__("swarm_coordinator")
        self.world = WorldModel()
        self.create_subscription(BiasField, "/mission/bias_field", self._bias_cb, 10)
        self.create_subscription(PoseArray, "/uav/state_array", self._uav_state_cb, 10)
        self.create_subscription(PoseArray, "/ugv/state_array", self._ugv_state_cb, 10)
        self.cmd_pub = self.create_publisher(UavCommandArray, "/uav/high_level_cmd", 10)
        self.timer = self.create_timer(1.0, self._tick)

    def _bias_cb(self, msg: BiasField) -> None:
        self.world.bias = msg

    def _uav_state_cb(self, msg: PoseArray) -> None:
        self.world.uav_poses = msg

    def _ugv_state_cb(self, msg: PoseArray) -> None:
        self.world.ugv_poses = msg

    def _tick(self) -> None:
        if not self.world.ready():
            return
        cmd = self.world.build_idle_uav_command()
        self.cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SwarmCoordinatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
