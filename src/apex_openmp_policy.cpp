//  Copyright (c) 2015 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>
#include <set>
#include <utility>
#include <cstdlib>
#include <stdexcept>
#include <stdio.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

#include <omp.h>

#include "apex_api.hpp"
#include "apex_policies.hpp"


static int window = 1;
static std::unordered_map<std::string, std::shared_ptr<apex_tuning_request>> requests;

static void set_omp_params(std::shared_ptr<apex_tuning_request> request) {
        std::shared_ptr<apex_param_enum> thread_param = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_num_threads"));
        const int num_threads = boost::lexical_cast<int>(thread_param->get_value());

        std::shared_ptr<apex_param_enum> schedule_param = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_schedule"));
        const std::string schedule_value = schedule_param->get_value();
        omp_sched_t schedule = omp_sched_auto;
        if(schedule_value == "static") {
            schedule = omp_sched_static;
        } else if(schedule_value == "dynamic") {
            schedule = omp_sched_dynamic;
        } else if(schedule_value == "guided") {
            schedule = omp_sched_guided;
        } else {
            throw std::invalid_argument("omp_schedule");
        }

        std::shared_ptr<apex_param_enum> chunk_param = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_chunk_size"));
        const int chunk_size = boost::lexical_cast<int>(chunk_param->get_value());

        //fprintf(stderr, "num_threads: %d, schedule %d, chunk_size %d\n", num_threads, schedule, chunk_size);

        omp_set_num_threads(num_threads);
        omp_set_schedule(schedule, chunk_size);
}

void handle_start(const std::string & name) {
    auto search = requests.find(name);
    if(search == requests.end()) {
        // Start a new tuning session.
        std::shared_ptr<apex_tuning_request> request{std::make_shared<apex_tuning_request>(name)};
        requests.insert(std::make_pair(name, request));

        // Create an event to trigger this tuning session.
        apex_event_type trigger = apex::register_custom_event(name);
        request->set_trigger(trigger);

        // Create a metric
        std::function<double(void)> metric = [=]()->double{
            apex_profile * profile = apex::get_profile(name);
            if(profile == nullptr) {
                std::cerr << "ERROR: no profile for " << name << std::endl;
                return 0.0;
            } 
            if(profile->calls == 0.0) {
                std::cerr << "ERROR: calls = 0 for " << name << std::endl;
                return 0.0;
            }
            return (profile->accumulated/profile->calls);
        };
        request->set_metric(metric);

        // Set strategy
        request->set_strategy(apex_ah_tuning_strategy::NELDER_MEAD);

        int max_threads = omp_get_num_procs();

        // Create a parameter for number of threads.
        std::shared_ptr<apex_param_enum> threads_param = request->add_param_enum("omp_num_threads", "16", {"2", "4", "8", "16", "24", "32"});

        // Create a parameter for scheduling policy.
        std::shared_ptr<apex_param_enum> schedule_param = request->add_param_enum("omp_schedule", "static", {"static", "dynamic", "guided"});

        // Create a parameter for chunk size.
        std::shared_ptr<apex_param_enum> chunk_param = request->add_param_enum("omp_chunk_size", "64", {"1", "8", "32", "64", "128", "256", "512"});

        // Set OpenMP runtime parameters to initial values.
        set_omp_params(request);

        // Start the tuning session.
        apex_tuning_session_handle session = apex::setup_custom_tuning(*request);
    } else {
        // We've seen this region before.
        std::shared_ptr<apex_tuning_request> request = search->second;
        set_omp_params(request);
    }
}

void handle_stop(const std::string & name) {
    auto search = requests.find(name);
    if(search == requests.end()) {
        std::cerr << "ERROR: Stop received on \"" << name << "\" but we've never seen a start for it." << std::endl;
    } else {
        apex_profile * profile = apex::get_profile(name);
        if(window == 1 || (profile != nullptr && profile->calls >= window)) {
            std::shared_ptr<apex_tuning_request> request = search->second;
            // Evaluate the results
            apex::custom_event(request->get_trigger(), NULL);
            // Reset counter so each measurement is fresh.
            apex::reset(name);
        }
    }
};

int policy(const apex_context context) {
    if(context.event_type == APEX_START_EVENT) {
        std::string name{*((std::string *) context.data)};
        if(boost::starts_with(name, "OpenMP_PARALLEL_REGION")) {
            handle_start(name);
        }
    } else if(context.event_type == APEX_STOP_EVENT) {
        std::string name{*((std::string *) context.data)};
        if(boost::starts_with(name, "OpenMP_PARALLEL_REGION")) {
            handle_stop(name);
        }
    }        
    return APEX_NOERROR;
}


static apex_policy_handle * start_policy;
static apex_policy_handle * stop_policy;

int register_policy() {
    const char * option = std::getenv("APEX_OPENMP_WINDOW");
    if(option != nullptr) {
        window = boost::lexical_cast<int>(option);        
    }
    std::function<int(apex_context const&)> policy_fn{policy};
    start_policy = apex::register_policy(APEX_START_EVENT, policy_fn);    
    stop_policy  = apex::register_policy(APEX_STOP_EVENT,  policy_fn);    
    if(start_policy == nullptr || stop_policy == nullptr) {
        return APEX_ERROR;
    } else {
        return APEX_NOERROR;
    }
}
 
extern "C" {

    int apex_plugin_init() {
        int status =  register_policy();
        return status;
    }

    int apex_plugin_finalize() {
        apex::deregister_policy(start_policy);
        apex::deregister_policy(stop_policy);
        return APEX_NOERROR;
    }

}

