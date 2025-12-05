from custom_msgs.msg import BiasField, UavCommandArray, UavCommand
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Header


class WorldModel:
    def __init__(self) -> None:
        self.bias: BiasField | None = None
        self.uav_poses: PoseArray | None = None
        self.ugv_poses: PoseArray | None = None

    def ready(self) -> bool:
        return self.uav_poses is not None

    def build_idle_uav_command(self) -> UavCommandArray:
        cmd = UavCommandArray()
        cmd.header = Header()
        cmd.commands = []
        if self.uav_poses:
            for idx, _pose in enumerate(self.uav_poses.poses):
                single = UavCommand()
                single.uav_id = f"uav_{idx}"
                single.mode = "idle"
                cmd.commands.append(single)
        return cmd
