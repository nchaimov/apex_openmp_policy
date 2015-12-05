cmake_minimum_required (VERSION 2.8.12 FATAL_ERROR)
cmake_policy(SET CMP0042 NEW) 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(apex_openmp_policy CXX)

find_package(APEX)
if(APEX_FOUND)
    include_directories(${APEX_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${APEX_LIBRARIES})
else()
    message(FATAL_ERROR " APEX not found.")
endif()

add_definitions(-std=c++14)

add_subdirectory(src)

