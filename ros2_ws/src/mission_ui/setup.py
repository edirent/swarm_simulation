from setuptools import setup

package_name = "mission_ui"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "pygame", "numpy", "shapely"],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="todo@example.com",
    description="Operator UI for Phase 1 swarm simulation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mission_ui = mission_ui.ui_node:main",
        ],
    },
)
