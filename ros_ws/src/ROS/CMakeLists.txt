
cmake_minimum_required(VERSION 2.8)
project(nn-mp)

find_package(NN-MP REQUIRED)
find_package(jetson-utils REQUIRED)

find_package(CUDA REQUIRED)

find_package(catkin QUIET)

string(TOUPPER $ENV{ROS_DISTRO} ROS_DISTRO)
set(ROS_DISTRO "ROS_${ROS_DISTRO}")
message(STATUS "ROS_DISTRO: ${ROS_DISTRO}")

find_package(OpenCV COMPONENTS core calib3d)
#find_package(OpenCV REQUIRED)
find_package(VPI 2.0)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# Check this
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/package.xml
    COPYONLY
)

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    roscpp
    std_msgs
    sensor_msgs
    nav_msgs
    ackermann_msgs
)

catkin_package(
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp sensor_msgs std_msgs nav_msgs ackermann_msgs cv_bridge
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGES} -std=c++11")
include_directories(${catkin_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} /usr/include/eigen3 /home/nvidia/NN-MP/networks)

add_definitions(-DROS1 -DROS_DISTRO=${ROS_DISTRO})

cuda_add_executable(image_feature src/node_img_feat.cpp src/imageConverter.cpp)
target_link_libraries(image_feature ${catkin_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES} NN-MP jetson-utils opencv_core opencv_calib3d opencv_imgcodecs /opt/ros/melodic/include/cv_bridge)


cuda_add_executable(state_feature src/node_state_feat.cpp)
target_link_libraries(state_feature ${catkin_LIBRARIES} ${CUDA_LIBRARIES} NN-MP jetson-utils)

cuda_add_executable(actor src/node_actor.cpp)
target_link_libraries(actor ${catkin_LIBRARIES} ${CUDA_LIBRARIES} NN-MP jetson-utils)
