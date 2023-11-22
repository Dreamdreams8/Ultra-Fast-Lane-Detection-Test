cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(laneDetectApi)
set(CMAKE_CXX_STANDARD 14)
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -Wl,-rpath=/apollo/third_party/libtorch1.7/libtorch/lib") #-Wl,-rpath=./

#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wl,-rpath=/apollo/third_party/libtorch1.7/libtorch/lib") 
#set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/")
set(Torch_DIR "/apollo/third_party/libtorch1.7/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

aux_source_directory(${PROJECT_SOURCE_DIR}/src/. DIR_SRCS)

#link_directories(${TORCH_LIBRARIES} ${OpenCV_LIBS})

message(STATUS "include path: ${Torch_INCLUDE_DIRS}")
message(STATUS "include path: ${TORCH_LIBRARIES}")

#INCLUDE_DIRECTORIES(${Torch_INCLUDE_DIRS})

ADD_LIBRARY(${CMAKE_PROJECT_NAME} SHARED ${DIR_SRCS})

target_link_libraries(${CMAKE_PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})


