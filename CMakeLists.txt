cmake_minimum_required(VERSION 3.12)

project(subpix)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} 
    main.cpp 
    edgesubpix.h 
    edgesubpix.cpp
    gradient.h
    gradient.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_compile_options(${PROJECT_NAME} PRIVATE
        #$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /external:W0>
        $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>: -fPIC -fvisibility=hidden -Wl,--exclude-libs,ALL -Wall -Wextra -Wpedantic -Wmisleading-indentation -Wunused -Wuninitialized -Wshadow -Werror>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>>:/W4 /WX /external:W0>
        $<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},loongarch64>:-mlsx>
)
target_compile_definitions(${PROJECT_NAME} PRIVATE
        $<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},loongarch64>:CV_LSX>
)

message("Arch:${CMAKE_SYSTEM_PROCESSOR}")
