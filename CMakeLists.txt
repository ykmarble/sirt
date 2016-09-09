cmake_minimum_required(VERSION 2.6)

project(SIRT)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main main.cpp)
