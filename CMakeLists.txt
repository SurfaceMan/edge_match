cmake_minimum_required(VERSION 3.12)

project(match)
find_package(OpenCV REQUIRED)
add_executable(match 
main.cpp 
edgesubpix3.h
edgesubpix3.cpp
)

target_include_directories(match PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(match ${OpenCV_LIBRARIES})