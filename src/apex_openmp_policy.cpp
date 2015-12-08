#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>
#include <set>
#include <utility>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <omp.h>

#include "apex_api.hpp"
#include "apex_policies.hpp"


static std::unordered_map<std::string, std::shared_ptr<apex_tuning_request>> requests;

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
            }
            return profile->accumulated;
        };
        request->set_metric(metric);

        // Create a parameter for number of threads.
        std::shared_ptr<apex_param_long> param = request->add_param_long("omp_num_threads", 12, 0, 24, 1);

        omp_set_num_threads((int)param->get_value());

        // Start the tuning session.
        apex_tuning_session_handle session = apex::setup_custom_tuning(*request);
    } else {
        // We've seen this region before.
        std::shared_ptr<apex_tuning_request> request = search->second;
        std::shared_ptr<apex_param_long> param = std::static_pointer_cast<apex_param_long>(request->get_param("omp_num_threads"));
        omp_set_num_threads((int)param->get_value());
    }
}

void handle_stop(const std::string & name) {
    auto search = requests.find(name);
    if(search == requests.end()) {
        std::cerr << "ERROR: Stop received on \"" << name << "\" but we've never seen a start for it." << std::endl;
    } else {
        std::shared_ptr<apex_tuning_request> request = search->second;
        // Evaluate the results
        apex::custom_event(request->get_trigger(), NULL);
        // Reset counter so each measurement is fresh.
        apex::reset(name);
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

