from launch import LaunchDescription
from launch_ros.actions import Node
from launch_pal import get_pal_configuration


def generate_launch_description():

    pkg = 'hri_visualization'
    node = 'hri_visualization'
    ld = LaunchDescription()

    config = get_pal_configuration(pkg=pkg, node=node, ld=ld)

    hri_visualization_node = Node(
        package=pkg,
        executable='visualization',
        name=node,
        output='screen',
        parameters=config["parameters"],
        remappings=config["remappings"],
        arguments=config["arguments"],

    )

    ld.add_action(hri_visualization_node)

    return ld
