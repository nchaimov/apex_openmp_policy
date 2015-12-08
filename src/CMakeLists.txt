# Copyright (c) 2015 University of Oregon
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_library(apex_openmp_policy SHARED apex_openmp_policy.cpp)
target_include_directories(apex_openmp_policy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(apex_openmp_policy ${LIBS})
set_target_properties(apex_openmp_policy PROPERTIES OUTPUT_NAME apex_openmp_policy)

add_executable (policy_test policy_test.cpp)

if(GOMP_LIBRARY)
    target_link_libraries(policy_test PUBLIC ${GOMP_LIBRARY})
endif()

INSTALL(TARGETS apex_openmp_policy policy_test
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

