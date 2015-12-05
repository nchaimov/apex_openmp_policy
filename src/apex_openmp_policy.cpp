#include <iostream>

#include "apex_api.hpp"

extern "C" {
    int apex_plugin_init() {
        std::cerr << "apex_openmp_policy init" << std::endl;
        return 0;
    }

    int apex_plugin_finalize() {
        std::cerr << "apex_openmp_policy finalize" << std::endl;
        return 0;
    }
}
