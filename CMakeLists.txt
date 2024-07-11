cmake_minimum_required(VERSION 3.12)

project(subpix)

find_package(OpenCV REQUIRED)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

add_executable(${PROJECT_NAME} 
    main.cpp 
    edgesubpix.h 
    edgesubpix.cpp
    gradient.h
    gradient.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)