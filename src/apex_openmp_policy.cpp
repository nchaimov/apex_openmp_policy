//  Copyright (c) 2015 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <memory>
#include <set>
#include <utility>
#include <cstdlib>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <stdio.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
            

#include <omp.h>

#include "apex_api.hpp"
#include "apex_policies.hpp"


static int apex_openmp_policy_tuning_window = 3;
static apex_ah_tuning_strategy apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::NELDER_MEAD;
static std::unordered_map<std::string, std::shared_ptr<apex_tuning_request>> apex_openmp_policy_tuning_requests;
//static boost::shared_mutex request_mutex;
static bool apex_openmp_policy_verbose = false;
static bool apex_openmp_policy_use_history = false;
static std::string apex_openmp_policy_history_file = "";

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

        const char * name = request->get_name().c_str();

        if(apex_openmp_policy_verbose) {
            fprintf(stderr, "name: %s, num_threads: %d, schedule %d, chunk_size %d\n", name, num_threads, schedule, chunk_size);
        }

        omp_set_num_threads(num_threads);
        omp_set_schedule(schedule, chunk_size);
}


void handle_start(const std::string & name) {
    //boost::upgrade_lock<boost::shared_mutex> lock(request_mutex);
    auto search = apex_openmp_policy_tuning_requests.find(name);
    if(search == apex_openmp_policy_tuning_requests.end()) {
        // Start a new tuning session.
        if(apex_openmp_policy_verbose) {
            fprintf(stderr, "Starting tuning session for %s\n", name.c_str());
        }
        std::shared_ptr<apex_tuning_request> request{std::make_shared<apex_tuning_request>(name)};
        //boost::upgrade_to_unique_lock<boost::shared_mutex> unique_lock(lock);
        apex_openmp_policy_tuning_requests.insert(std::make_pair(name, request));

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
            double result = profile->accumulated/profile->calls;
            if(apex_openmp_policy_verbose) {
                fprintf(stderr, "time per call: %f\n", result);
            }
            return result;
        };
        request->set_metric(metric);

        // Set apex_openmp_policy_tuning_strategy
        request->set_strategy(apex_openmp_policy_tuning_strategy);

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
    //boost::shared_lock<boost::shared_mutex> lock(request_mutex);
    auto search = apex_openmp_policy_tuning_requests.find(name);
    if(search == apex_openmp_policy_tuning_requests.end()) {
        std::cerr << "ERROR: Stop received on \"" << name << "\" but we've never seen a start for it." << std::endl;
    } else {
        apex_profile * profile = apex::get_profile(name);
        if(apex_openmp_policy_tuning_window == 1 || (profile != nullptr && profile->calls >= apex_openmp_policy_tuning_window)) {
            std::shared_ptr<apex_tuning_request> request = search->second;
            // Evaluate the results
            apex::custom_event(request->get_trigger(), NULL);
            // Reset counter so each measurement is fresh.
            apex::reset(name);
        }
    }
};

int policy(const apex_context context) {
    if(context.data == nullptr) {
        // This is an address-identified timer.
        // Skip it.
        // FIXME apex_context.data should be a timer_event_data
        // referencing a task_identifier
        return APEX_NOERROR;
    }
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

void read_results(const std::string & filename) {
    std::ifstream results_file(filename, std::ifstream::in);
    if(!results_file.good()) {
        std::cerr << "Unable to open results file " << filename << std::endl;
        assert(false);
    } else {
        std::string line;
        std::getline(results_file, line); // ignore first line (header)
        while(!results_file.eof()) {
            std::getline(results_file, line);
            std::vector<std::string> parts;
            boost::split(parts, line, boost::is_any_of(","));
            if(parts.size() == 5) {
                std::string & name = parts[0];
                std::string & threads = parts[1];
                std::string & schedule = parts[2];
                std::string & chunk_size = parts[3];
                std::string & converged = parts[4];
                // Remove quotes from strings
                boost::remove_erase_if(name, boost::is_any_of("\""));
                boost::remove_erase_if(threads, boost::is_any_of("\""));
                boost::remove_erase_if(schedule, boost::is_any_of("\""));
                boost::remove_erase_if(chunk_size, boost::is_any_of("\""));
                boost::remove_erase_if(converged, boost::is_any_of("\""));
                // Create a dummy tuning request with the values from the results file.
                std::shared_ptr<apex_tuning_request> request{std::make_shared<apex_tuning_request>(name)};
                apex_openmp_policy_tuning_requests.insert(std::make_pair(name, request));
                std::shared_ptr<apex_param_enum> threads_param = request->add_param_enum("omp_num_threads", threads, {threads});
                std::shared_ptr<apex_param_enum> schedule_param = request->add_param_enum("omp_schedule", schedule, {schedule});
                std::shared_ptr<apex_param_enum> chunk_param = request->add_param_enum("omp_chunk_size", chunk_size, {chunk_size});
                
                if(apex_openmp_policy_verbose) {
                   fprintf(stderr, "Added %s -> (%s, %s, %s) from history.\n", name.c_str(), threads.c_str(), schedule.c_str(), chunk_size.c_str());
                }
            }
        }
    }
}

void print_summary() {
    std::time_t time = std::time(NULL);
    char time_str[128];
    std::strftime(time_str, 128, "results-%F-%H-%M-%S.csv", std::localtime(&time));
    std::ofstream results_file(time_str, std::ofstream::out);
    results_file << "\"name\",\"num_threads\",\"schedule\",\"chunk_size\",\"converged\"" << std::endl;
    std::cout << std::endl << "OpenMP final settings: " << std::endl;
    //boost::shared_lock<boost::shared_mutex> lock(request_mutex);
    for(auto request_pair : apex_openmp_policy_tuning_requests) {
        auto request = request_pair.second;
        const std::string & name = request->get_name();
        const std::string & threads = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_num_threads"))->get_value();
        const std::string & schedule = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_schedule"))->get_value();
        const std::string & chunk = std::static_pointer_cast<apex_param_enum>(request->get_param("omp_chunk_size"))->get_value();
        const std::string converged = request->has_converged() ? "CONVERGED" : "NOT CONVERGED";
        std::cout << "name: " << name << ", num_threads: " << threads << ", schedule: " << schedule
            << ", chunk_size: " << chunk << " " << converged << std::endl;
        results_file << "\"" << name << "\"," << threads << ",\"" << schedule << "\"," << chunk << ",\"" << converged << "\"" << std::endl;
    }
    std::cout << std::endl;
}

static apex_policy_handle * start_policy;
static apex_policy_handle * stop_policy;

int register_policy() {
    // Process environment variables
    
    // APEX_OPENMP_VERBOSE
    const char * apex_openmp_policy_verbose_option = std::getenv("APEX_OPENMP_VERBOSE");
    if(apex_openmp_policy_verbose_option != nullptr) {
        apex_openmp_policy_verbose = 1;
    }

    // APEX_OPENMP_WINDOW
    const char * option = std::getenv("APEX_OPENMP_WINDOW");
    if(option != nullptr) {
        apex_openmp_policy_tuning_window = boost::lexical_cast<int>(option);        
    }
    if(apex_openmp_policy_verbose) {
        std::cerr << "apex_openmp_policy_tuning_window = " << apex_openmp_policy_tuning_window << std::endl;
    }

    // APEX_OPENMP_STRATEGY
    const char * apex_openmp_policy_tuning_strategy_option = std::getenv("APEX_OPENMP_STRATEGY");
    std::string apex_openmp_policy_tuning_strategy_str = (apex_openmp_policy_tuning_strategy_option == nullptr) ? std::string() : std::string(apex_openmp_policy_tuning_strategy_option);
    boost::algorithm::to_upper(apex_openmp_policy_tuning_strategy_str);
    if(apex_openmp_policy_tuning_strategy_str.empty()) {
        // default
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::NELDER_MEAD;
        std::cerr << "Using default tuning strategy (NELDER_MEAD)" << std::endl;
    } else if(apex_openmp_policy_tuning_strategy_str == "EXHAUSTIVE") {
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::EXHAUSTIVE;
        std::cerr << "Using EXHAUSTIVE tuning strategy." << std::endl;
    } else if(apex_openmp_policy_tuning_strategy_str == "RANDOM") {
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::RANDOM;
        std::cerr << "Using RANDOM tuning strategy." << std::endl;
    } else if(apex_openmp_policy_tuning_strategy_str == "NELDER_MEAD") {
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::NELDER_MEAD;
        std::cerr << "Using NELDER_MEAD tuning strategy." << std::endl;
    } else if(apex_openmp_policy_tuning_strategy_str == "PARALLEL_RANK_ORDER") {
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::PARALLEL_RANK_ORDER;
        std::cerr << "Using PARALLEL_RANK_ORDER tuning strategy." << std::endl;
    } else {
        std::cerr << "Invalid setting for APEX_OPENMP_STRATEGY: " << apex_openmp_policy_tuning_strategy_str << std::endl;
        std::cerr << "Will use default of NELDER_MEAD." << std::endl;
        apex_openmp_policy_tuning_strategy = apex_ah_tuning_strategy::NELDER_MEAD;
    }

    // APEX_OPENMP_HISTORY
    const char * apex_openmp_policy_history_file_option = std::getenv("APEX_OPENMP_HISTORY");
    if(apex_openmp_policy_history_file_option != nullptr) {
        apex_openmp_policy_history_file = std::string(apex_openmp_policy_history_file_option);
        if(!apex_openmp_policy_history_file.empty()) {
            apex_openmp_policy_use_history = true;
        }
    }

    if(apex_openmp_policy_use_history) {
        read_results(apex_openmp_policy_history_file);
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
        fprintf(stderr, "apex_openmp_policy init\n");
        int status =  register_policy();
        return status;
    }

    int apex_plugin_finalize() {
        fprintf(stderr, "apex_openmp_policy finalize\n");
        apex::deregister_policy(start_policy);
        apex::deregister_policy(stop_policy);
        print_summary();
        return APEX_NOERROR;
    }

}

