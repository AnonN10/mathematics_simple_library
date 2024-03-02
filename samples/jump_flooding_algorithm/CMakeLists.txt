cmake_minimum_required(VERSION 3.24)

project(JumpFloodingAlgorithm)
add_executable(JumpFloodingAlgorithm main.cpp)
target_include_directories(
    JumpFloodingAlgorithm
    PUBLIC
    ../../
    ../common/
)
set_property(TARGET JumpFloodingAlgorithm PROPERTY CXX_STANDARD 23)
set_property(TARGET JumpFloodingAlgorithm PROPERTY CXX_STANDARD_REQUIRED On)
set_property(TARGET JumpFloodingAlgorithm PROPERTY CXX_EXTENSIONS Off)