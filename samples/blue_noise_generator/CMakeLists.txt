cmake_minimum_required(VERSION 3.24)

project(BlueNoiseGenerator)
add_executable(BlueNoiseGenerator main.cpp)
target_include_directories(
    BlueNoiseGenerator
    PUBLIC
    ../../
    ../common/
)
set_property(TARGET BlueNoiseGenerator PROPERTY CXX_STANDARD 20)
set_property(TARGET BlueNoiseGenerator PROPERTY CXX_STANDARD_REQUIRED On)
set_property(TARGET BlueNoiseGenerator PROPERTY CXX_EXTENSIONS Off)