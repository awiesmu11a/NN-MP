
cmake_minimum_required(VERSION 2.8)
project(ROS_NN-MP)

find_package(NN-MP REQUIRED)
find_package(jetson-utils REQUIRED)

find_package(CUDA REQUIRED)

find_package(catkin QUIET)

string(TOUPPER $ENV{ROS_DISTRO} ROS_DISTRO)
set(ROS_DISTRO "ROS_${ROS_DISTRO}")
message(STATUS "ROS_DISTRO: ${ROS_DISTRO}")

find_package(VPI 2.0)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/package.ros1.xml
    ${CMAKE_CURRENT_SOURCE_DIR}/package.xml
    COPYONLY
)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    image_transport
    cv_bridge
    tf2
    tf2_ros
    tf2_geometry_msgs
    tf2_sensor_msgs
    tf2_msgs
    tf2_eigen
    message_filters
    message_generation
    std_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    dynamic_reconfigure
    ${ROS_DISTRO}
)

catkin_packages(
    LIBRARIES NN-MP nodelets
    CATKIN_DEPENDS nodelets roscpp sensor_msgs image_transport cv_bridge tf2 tf2_ros tf2_geometry_msgs tf2_sensor_msgs tf2_msgs tf2_eigen message_filters message_generation std_msgs geometry_msgs nav_msgs visualization_msgs dynamic_reconfigure ${ROS_DISTRO}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGES} -std=c++11")
include_directories(${catkin_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
add_definition(-DROS1 -DROS_DISTRO=${ROS_DISTRO})

set(common_src src/image_converter.cpp)

add_executable(image_feature ${common_src} src/node_img_feat.cpp)
target_link_libraries(image_feature ${catkin_LIBRARIES} ${CUDA_LIBRARIES} NN-MP jetson-utils)

add_executable(actor ${common_src} src/node_actor.cpp)
target_link_libraries(actor ${catkin_LIBRARIES} ${CUDA_LIBRARIES} NN-MP jetson-utils image_feature)

# Think about nodelets