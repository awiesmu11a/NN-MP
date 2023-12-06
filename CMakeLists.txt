
cmake_minimum_required(VERSION 2.8)
project(NN-MP)

# For now BUILD_EXPERIMENTAL not set

# setup build flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-write-strings -Wno-deprecated-declarations")	# -std=c++14 
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")
set(BUILD_INTERACTIVE "YES" CACHE BOOL "If NO, will download/install the default DNN models without prompting the user, and skip installation of PyTorch.")
set(BUILD_EXPERIMENTAL "NO" CACHE BOOL "If YES, will enable support for experimental DNNs, examples, and plugins")

# detect distro version
find_program(LSB_RELEASE_EXEC lsb_release)

execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --id OUTPUT_VARIABLE LSB_RELEASE_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --release OUTPUT_VARIABLE LSB_RELEASE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --codename OUTPUT_VARIABLE LSB_RELEASE_CODENAME OUTPUT_STRIP_TRAILING_WHITESPACE)

message("-- distro ID:       ${LSB_RELEASE_ID}")
message("-- distro version:  ${LSB_RELEASE_VERSION}")
message("-- distro codename: ${LSB_RELEASE_CODENAME}")

# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
if( ${BUILD_DEPS} )
	message("-- Launching pre-build dependency installer script...")
	message("-- Build interactive:  ${BUILD_INTERACTIVE}")

	execute_process(COMMAND sh ../CMakePreBuild.sh ${BUILD_INTERACTIVE}
				WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
				RESULT_VARIABLE PREBUILD_SCRIPT_RESULT)

	set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed." FORCE)
	message("-- Finished installing dependencies")
endif()

# setup CUDA
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/jetson-utils/cuda" )
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
	message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_53 SM_62")
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62)

	if(CUDA_VERSION_MAJOR GREATER 9)
		message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_72")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_72,code=sm_72)
	endif()

	if(CUDA_VERSION_MAJOR GREATER 10)
		message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_87")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
	endif()
endif()

# OpenCV used for findHomography() and decomposeHomography()
# OpenCV version >= 3.0.0 required for decomposeHomography()
find_package(OpenCV COMPONENTS core calib3d)

if( NOT OpenCV_FOUND )
	message("-- didn't find OpenCV on system, disabling OpenCV")
else()
	message("-- OpenCV version:  " ${OpenCV_VERSION})

	if( ${OpenCV_VERSION_MAJOR} LESS 3 )
		message("-- OpenCV version less than 3.0, disabling OpenCV")
	else()
		message("-- OpenCV version >= 3.0.0, enabling OpenCV")
		set(HAS_OPENCV 1)
		add_definitions(-DHAS_OPENCV)
	endif()	
endif()

# check for VPI (TODO: VPI 1.0 support for JetPack 4.x)
find_package(VPI 2.0)

if( NOT VPI_FOUND )
	message("-- didn't find VPI on system, disabling VPI")
else()
	message("-- VPI version:  " ${VPI_VERSION})
	set(HAS_VPI 1)
	add_definitions(-DHAS_VPI)
endif()

# setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

# build C/C++ library
# --------------------------------------------Add the jetson-utils path here------------------------------------------
include_directories(${PROJECT_INCLUDE_DIR} ${PROJECT_INCLUDE_DIR}/jetson-utils)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/${CMAKE_SYSTEM_PROCESSOR}/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/${CMAKE_SYSTEM_PROCESSOR}/glib-2.0/include/)

#-----------------------------------------------Change the paths accordingly------------------------------------------
# file(GLOB inferenceSources networks/*.cpp)
# file(GLOB inferenceIncludes networks/*.h)

file(GLOB inferenceSources networks/CNN.cpp networks/imageConverter_test.cpp networks/randInt8Calibrator.cpp networks/tensorNet.cpp networks/test.cpp)
file(GLOB inferenceIncludes networks/CNN.h networks/imageConverter_test.h networks/randInt8Calibrator.h networks/tensorNet.h)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
	link_directories(/usr/lib/aarch64-linux-gnu/tegra)
endif()


#----------------------------------Not sure what this is-----------------------------------------------------------
cuda_add_library(NN-MP SHARED ${inferenceSources})

# transfer all headers to the include directory--------------------------------------------------------------------
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/NN-MP)

foreach(include ${inferenceIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR}/NN-MP COPYONLY)
endforeach()

execute_process( COMMAND "$CMAKE_COMMAND" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/networks/models" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" )

# set linker options--------------------------------------------------------------
target_link_libraries(NN-MP jetson-utils nvinfer nvinfer_plugin nvcaffe_parser)

#---------------------------------------
if(CUDA_VERSION_MAJOR GREATER 9)
	target_link_libraries(NN-MP nvonnxparser)
endif()

if(HAS_OPENCV) 
   message("-- linking CNN-test with OpenCV " ${OpenCV_VERSION})
   target_link_libraries(NN-MP opencv_core opencv_calib3d)
endif()

if(HAS_VPI) 
   message("-- linking CNN-test with VPI " ${VPI_VERSION})
   target_link_libraries(NN-MP vpi)
endif()

# install includes
foreach(include ${inferenceIncludes})
    install(FILES "${include}" DESTINATION include/NN-MP)
endforeach()

# install symlink to networks and images
# install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/cnn/models ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})" )
# install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/data/images ${CMAKE_INSTALL_PREFIX}/bin/images )" )

# install the shared library
install(TARGETS NN-MP DESTINATION lib EXPORT NN-MPConfig)

cuda_add_executable(test_model ${PROJECT_SOURCE_DIR}/NN-MP/test.cpp)
target_link_libraries(test_model NN-MP)

install(TARGETS test_model DESTINATION bin)

# install the cmake project, for importing
install(EXPORT NN-MPConfig DESTINATION share/NN-MP/cmake)

# run ldconfig after installing
install(CODE "execute_process( COMMAND ldconfig )")
