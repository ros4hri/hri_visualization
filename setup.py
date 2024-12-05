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

from setuptools import find_packages, setup
from glob import glob

package_name = 'hri_visualization'

setup(
    name=package_name,
    version='2.2.0',
    packages=find_packages(exclude=['test']),
    install_requires=[],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Lorenzo Ferrini',
    maintainer_email='lorenzo.ferrini@pal-robotics.com',
    description='ROS4HRI-compatible node to visualize body and face detection results',
    license='Apache 2.0',
    entry_points={
        'console_scripts': [
            'visualization = hri_visualization.visualization:main',
        ],
    },
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/fonts/Montserrat', glob('fonts/Montserrat/*')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/images', glob('images/*')),
        ('share/ament_index/resource_index/pal_configuration.' + package_name,
            ['config/' + package_name]),
        ('share/' + package_name + '/config', ['config/00-defaults.yml']),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/ament_index/resource_index/pal_system_module',
         ['module/' + package_name]),
        ('share/' + package_name + '/module',
         ['module/hri_visualization_module.yaml']),

    ],
)
