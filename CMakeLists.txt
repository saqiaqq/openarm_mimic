cmake_minimum_required(VERSION 3.8)
project(openarm_mimic)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(openarm_teleop REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(urdf REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenArmCAN REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MimicFrame.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Python module
# ament_python_install_package(${PROJECT_NAME})

# C++ Executable
add_executable(mimic_control src/mimic_control.cpp src/robot_kdl.cpp)
target_include_directories(mimic_control PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(mimic_control
  openarm_teleop::openarm_teleop_lib
  OpenArmCAN::openarm_can
  ${orocos_kdl_LIBRARIES}
  kdl_parser::kdl_parser
)
ament_target_dependencies(mimic_control
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  urdf
  tf2
  tf2_geometry_msgs
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(mimic_control "${cpp_typesupport_target}")

# Install
install(TARGETS mimic_control
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  openarm_mimic/mimic_vision.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mimic_vision
)

ament_package()
