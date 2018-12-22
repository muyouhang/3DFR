CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(Led3DFR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#find_package(PCL 1.2 REQUIRED)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

SET(OpenCV_DIR /data/irip_rpt/3rdparty/opencv-4.0.0/build)
find_package(OpenCV 4.0 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(/data/irip_rpt/3rdparty/mxnet/include)
link_directories(/data/irip_rpt/3rdparty/mxnet/lib)

include_directories(${PROJECT_SOURCE_DIR}/include/Led3DFR)

add_executable(test_kinfu src/BasicFuncation.cpp src/FaceLandmark.cpp src/FaceRecognition.cpp src/FeatureExtractor.cpp src/ImageProcess.cpp src/KinectFusion.cpp examples/test_kinfu.cpp)

target_link_libraries(test_kinfu  ${OpenCV_LIBS} libmxnet.so)