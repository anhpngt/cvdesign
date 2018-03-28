cmake_minimum_required(VERSION 2.8.3)
project(cannyEdgeDesign)

set(CMAKE_CXX_FLAGS "-std=c++0x -O2 -Wall ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

add_executable(CannyEdgeDetector src/CannyEdgeDetector.cpp)
add_executable(LoGEdgeDetector src/LoGEdgeDetector.cpp)
target_link_libraries(CannyEdgeDetector ${OpenCV_LIBS})
target_link_libraries(LoGEdgeDetector ${OpenCV_LIBS})