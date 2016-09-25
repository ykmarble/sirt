cmake_minimum_required(VERSION 2.6)

project(SIRT)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
