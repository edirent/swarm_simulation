# Swarm Simulation Phase 1 (ROS 2, AirSim + Stage)

Single-host simulation scaffold for a mixed UAV/UGV swarm:
- UAV in AirSim (ROS wrapper `airsim_ros_pkgs`)
- UGVs in Stage 2D via `stage_ros2`
- Centralized Python swarm logic (explore + followBias + target handoff)
- Operator clicks on a 2D map to set goal/bias

## Layout
```
ros2_ws/
  src/
    airsim_ros_pkgs/    # clone from microsoft/AirSim (ros/src/airsim_ros_pkgs)
    stage_ros2/         # clone from tuw-robotics/stage_ros2
    custom_msgs/        # message package (BiasField, UavCommand*)
    mission_ui/         # pygame UI node
    swarm_core/         # coordinator + control + perception nodes
sim_configs/
  airsim/               # UE project/configs (placeholder)
  stage/
    worlds/             # Stage world files (placeholder)
```

## Repos to clone (Phase 1)
- AirSim: `https://github.com/microsoft/AirSim` (copy `ros/src/airsim_ros_pkgs` into `ros2_ws/src/airsim_ros_pkgs`)
- Stage bridge: `https://github.com/tuw-robotics/stage_ros2`
- Optional fallback 2D sim: `https://github.com/stdr-simulator-ros-pkg/stdr_simulator`

## ROS 2 packages (Python unless noted)
- `custom_msgs`: `BiasField`, `UavCommand`, `UavCommandArray`
- `mission_ui`: subscribes `/world/{uav,ugv}_poses`, publishes `/mission/goal_area`, `/mission/bias_field`
- `swarm_core`:
  - `swarm_coordinator_node.py`: explore + bias blend; assigns UGV to UAV detections
  - `uav_control_node.py`: translate high-level waypoints to AirSim commands
  - `uav_perception_node.py`: simple detection stub (replace with vision)
  - `ugv_control_node.py`: go-to-goal for Stage robots
  - `world_model.py`: tracks poses/bias/detections

## Build
```
cd ros2_ws
colcon build
source install/setup.bash
```

## Next steps
1) Clone `airsim_ros_pkgs` and `stage_ros2` into `ros2_ws/src/`.
2) Author a Stage world in `sim_configs/stage/worlds/` and launch `stage_ros2`.
3) Launch AirSim + `airsim_node` with at least one drone.
4) Implement real control/detection logic in `swarm_core` and UI interactions in `mission_ui`.
