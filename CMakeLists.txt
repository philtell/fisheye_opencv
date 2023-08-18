cmake_minimum_required(VERSION 3.10)
project(fisheye)

set(CMAKE_CXX_STANDARD 17)

add_executable(fisheye main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


set(SOURCE_FILES main.cpp)

target_link_libraries(fisheye ${OpenCV_LIBS})