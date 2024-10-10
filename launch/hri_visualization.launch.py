from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'funny_names',
            default_value='false',
            description='If true, the visualizer will display a random adjective + Pokémon name as a person identifier.'
        ),
        DeclareLaunchArgument(
            'compressed_output',
            default_value='false',
            description='If true, the output will be published as a compressed image.'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/image',
            description='The input image topic to subscribe to.'
        ),
        DeclareLaunchArgument(
            'body_topic',
            default_value='/humans/bodies/tracked',
            description='The input body topic to subscribe to.'
        ),

        # Node to launch the HRI Visualizer
        Node(
            package='hri_visualization',
            executable='visualization',  # This should match the entry point in your setup.py
            name='hri_visualization',
            output='screen',
            parameters=[
                {'funny_names': LaunchConfiguration('funny_names')},
                {'compressed_output': LaunchConfiguration(
                    'compressed_output')},
            ],
            remappings=[
                ('/image', LaunchConfiguration('image_topic')),
                ('/humans/bodies/tracked', LaunchConfiguration('body_topic'))
            ]
        ),
    ])
