cmake_minimum_required(VERSION 3.24)

# set the project name
project(MainDemo)

# add the executable
add_executable(MainDemo main.cpp)

set_property(TARGET MainDemo PROPERTY CXX_STANDARD 23)
set_property(TARGET MainDemo PROPERTY CXX_STANDARD_REQUIRED On)
set_property(TARGET MainDemo PROPERTY CXX_EXTENSIONS Off)

option(USE_SDL2_AS_PACKAGE "Use SDL2 package instead of submodule" OFF)
option(MSL_BUILD_SAMPLES "Build samples" ON)

if(MSL_BUILD_SAMPLES)
    add_subdirectory(samples/blue_noise_generator/)
    add_subdirectory(samples/lights_out_solver/)
    add_subdirectory(samples/jump_flooding_algorithm/)
    add_subdirectory(samples/ray_tracing/)
endif()