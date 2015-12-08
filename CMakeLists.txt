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

find_package(Boost 1.54.0 REQUIRED COMPONENTS system thread timer chrono regex)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${Boost_LIBRARIES})
else()
    message(FATAL_ERROR " Boost not found.")
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR " OpenMP not found.")
endif()

add_definitions(-std=c++14)

add_subdirectory(src)

