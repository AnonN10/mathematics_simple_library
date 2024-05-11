cmake_minimum_required(VERSION 3.24)

project(RayTracing)
add_executable(RayTracing main.cpp)
target_include_directories(
    RayTracing
    PUBLIC
    ../../
    ../common/
)
set_property(TARGET RayTracing PROPERTY CXX_STANDARD 23)
set_property(TARGET RayTracing PROPERTY CXX_STANDARD_REQUIRED On)
set_property(TARGET RayTracing PROPERTY CXX_EXTENSIONS Off)

add_subdirectory(contrib/sdl)

target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2 SDL2::SDL2main)

if (WIN32)
# copy the .dll file to the same folder as the executable
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:SDL2::SDL2>
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
    VERBATIM)
endif()