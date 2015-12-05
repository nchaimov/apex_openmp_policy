add_library(apex_openmp_policy SHARED apex_openmp_policy.cpp)
target_include_directories(apex_openmp_policy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(apex_openmp_policy ${LIBS})
set_target_properties(apex_openmp_policy PROPERTIES OUTPUT_NAME apex_openmp_policy)

INSTALL(TARGETS apex_openmp_policy
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

