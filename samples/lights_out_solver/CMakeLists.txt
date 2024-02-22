cmake_minimum_required(VERSION 3.24)

project(LightsOutSolver)
add_executable(LightsOutSolver main.cpp)
target_include_directories(
    LightsOutSolver
    PUBLIC
    ../../
    ../common/
)
set_property(TARGET LightsOutSolver PROPERTY CXX_STANDARD 20)
set_property(TARGET LightsOutSolver PROPERTY CXX_STANDARD_REQUIRED On)
set_property(TARGET LightsOutSolver PROPERTY CXX_EXTENSIONS Off)