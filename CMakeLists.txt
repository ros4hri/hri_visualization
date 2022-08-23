cmake_minimum_required(VERSION 3.0.2)
project(hri_visualization)

find_package(catkin REQUIRED)

catkin_package(
    CATKIN_DEPENDS 
    sensor_msgs
    std_msgs
    hri_msgs)

catkin_install_python(PROGRAMS nodes/visualization
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


