cmake_minimum_required(VERSION 3.24)

# set the project name
project(Application)

# add the executable
add_executable(Application main.cpp)

set_property(TARGET Application PROPERTY CXX_STANDARD 20)
set_property(TARGET Application PROPERTY CXX_STANDARD_REQUIRED On)
set_property(TARGET Application PROPERTY CXX_EXTENSIONS Off)

add_subdirectory(samples/blue_noise_generator/)
add_subdirectory(samples/lights_out_solver/)