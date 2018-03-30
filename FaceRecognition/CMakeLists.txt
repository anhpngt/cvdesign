cmake_minimum_required(VERSION 2.8.3)
project(FaceRecognition)

set(CMAKE_CXX_FLAGS "-std=c++0x -O2 -Wall ${CMAKE_CXX_FLAGS}")

find_package(OpenCV 3.4.0 REQUIRED)

add_executable(trainEigenfaces trainEigenfaces.cpp)
add_executable(evalEigenfaces evalEigenfaces.cpp)
add_executable(faceRecognition faceRecognition.cpp)
target_link_libraries(trainEigenfaces ${OpenCV_LIBS})
target_link_libraries(evalEigenfaces ${OpenCV_LIBS})
target_link_libraries(faceRecognition ${OpenCV_LIBS})

add_executable(opencvTrainEigenfaces opencvTrainEigenfaces.cpp)
add_executable(opencvFaceRecognition opencvFaceRecognition.cpp)
target_link_libraries(opencvTrainEigenfaces ${OpenCV_LIBS})
target_link_libraries(opencvFaceRecognition ${OpenCV_LIBS})