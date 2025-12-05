from setuptools import setup

package_name = "swarm_core"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "opencv-python"],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="todo@example.com",
    description="Swarm coordinator, control, and perception nodes for Phase 1 simulation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "swarm_coordinator = swarm_core.swarm_coordinator_node:main",
            "uav_control = swarm_core.uav_control_node:main",
            "uav_perception = swarm_core.uav_perception_node:main",
            "ugv_control = swarm_core.ugv_control_node:main",
        ],
    },
)
