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

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        message(STATUS "clang-cl detected")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX /arch:AVX2")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX /arch:AVX2")
    elseif (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
        message(STATUS "clang(GNU CLI) detected")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx2")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    message(STATUS "GCC detected")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx2")
else()
    message(WARNING "Couldn't figure out the compiler to set architecture.")
endif()

add_subdirectory(contrib/sdl)

target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2main SDL2::SDL2)

if (WIN32)
# copy the .dll file to the same folder as the executable
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:SDL2::SDL2>
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
    VERBATIM)
endif()