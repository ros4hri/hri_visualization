from setuptools import setup
import os
from glob import glob

package_name = 'hri_visualization'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=[],
    zip_safe=True,
    maintainer='Lorenzo Ferrini',
    maintainer_email='lorenzo.ferrini@pal-robotics.com',
    description='Nodes to visualize body and face detection results',
    license='BSD',
    entry_points={
        'console_scripts': [
            'visualization = hri_visualization.visualization:main',
        ],
    },
    data_files=[
        ('share/' + package_name + '/fonts/Montserrat', glob('fonts/Montserrat/*')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/images', glob('images/*')),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/ament_index/resource_index/pal_system_module',
         ['module/' + package_name]),
        ('share/' + package_name + '/module',
         ['module/hri_visualization_module.yaml']),

    ],
)
