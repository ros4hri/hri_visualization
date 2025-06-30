# Copyright 2024 PAL Robotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from ament_index_python.packages import get_package_share_directory
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

    diagnostics_analyzer = Node(
        package='diagnostic_aggregator',
        executable='add_analyzer',
        namespace=pkg,
        output='screen',
        emulate_tty=True,
        parameters=[
            os.path.join(get_package_share_directory(pkg), 'config', f'{pkg}_analyzers.yaml')],
    )

    ld.add_action(hri_visualization_node)
    ld.add_action(diagnostics_analyzer)
    return ld
