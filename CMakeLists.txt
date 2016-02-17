# Copyright (c) 2015 University of Oregon
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required (VERSION 2.8.12 FATAL_ERROR)
cmake_policy(SET CMP0042 NEW) 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(apex_openmp_policy CXX)

include(GitExternal)

# add_definitions(-std=c++11)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    message(INFO " Compiler ${CMAKE_CXX_COMPILER} has C++14 support.")
else()
    CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
    if(COMPILER_SUPPORTS_CXX1Y)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
        message(INFO " Compiler ${CMAKE_CXX_COMPILER} has early C++14 support.")
    else()
        message(FATAL_ERROR " Compiler ${CMAKE_CXX_COMPILER} does not have any C++14 support; please use a newer compiler.")
    endif()
endif()

find_package(APEX)
if(APEX_FOUND)
    message(INFO " Found APEX: ${APEX_LIBRARIES}")
    include_directories(${APEX_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${APEX_LIBRARIES})
else()
    message(FATAL_ERROR " APEX not found. Please make sure APEX_ROOT is set to the APEX install path.")
endif()

find_package(Boost 1.54.0 REQUIRED COMPONENTS system thread timer chrono regex)
if(Boost_FOUND)
    message(INFO " Found Boost: ${Boost_INCLUDE_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${Boost_LIBRARIES})
else()
    message(FATAL_ERROR " Boost 1.54.0 or newer, or a required component, not found.")
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
    message(INFO " Found OpenMP: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR " OpenMP not found.")
endif()

find_library(
    GOMP_LIBRARY
    NAMES gomp libgomp 
    PATHS ${GOMP_ROOT} ${GOMP_ROOT}/build-gcc ${GOMP_ROOT}/build-gcc/lib ${GOMP_ROOT}/lib ${OMPT_ROOT} ${OMPT_ROOT}/build-gcc ${OMPT_ROOT}/build-gcc/lib ${OMPT_ROOT}/lib
)

if(GOMP_LIBRARY)
    message(INFO " Found GOMP at ${GOMP_LIBRARY}")
else()
    message(INFO " No GOMP")
endif()

git_external(rapidjson
    https://github.com/miloyip/rapidjson.git
    master
    VERBOSE)

find_path(
    RAPIDJSON_INCLUDE_DIR
    NAMES rapidjson
    PATHS ${PROJECT_SOURCE_DIR}/rapidjson/include)

if(RAPIDJSON_INCLUDE_DIR)
    message(INFO " Found rapidjson at ${RAPIDJSON_INCLUDE_DIR}")
    include_directories(${RAPIDJSON_INCLUDE_DIR})
else()
    message(FATAL_ERROR " rapidjson not found. This should have been checked out automatically. "
        "Try manually check out https://github.com/miloyip/rapidjson.git to ${PROJECT_SOURCE_DIR}")
endif()

add_subdirectory(src)

