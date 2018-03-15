cmake_minimum_required(VERSION 2.8.3)
project(FaceRecognition)

set(CMAKE_CXX_FLAGS "-std=c++0x -O2 -Wall ${CMAKE_CXX_FLAGS}")

find_package(OpenCV 3.4 REQUIRED)

add_executable(faceRecognition faceRecognition.cpp)
add_executable(trainEigenfaces trainEigenfaces.cpp)

target_link_libraries(faceRecognition ${OpenCV_LIBS})
target_link_libraries(trainEigenfaces ${OpenCV_LIBS})