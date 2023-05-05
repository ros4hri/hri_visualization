cmake_minimum_required(VERSION 3.0.2)
project(hri_visualization)

find_package(catkin REQUIRED)

catkin_package( CATKIN_DEPENDS 
  rospy
  pyhri
  hri_msgs
  cv_bridge
  sensor_msgs
)

catkin_install_python(PROGRAMS nodes/visualization
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY
  fonts
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)