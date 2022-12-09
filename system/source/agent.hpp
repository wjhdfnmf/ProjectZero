// William Gregory (c) Copyright 2017 - All rights reserved

#ifndef _AGENT_
#define _AGENT_

#include <vector>
#include "agent/network_trainer.hpp"	// AGENT

/*
*	agent class - "port" between agent and system
*/
class Agent {
    private:
    Trainer::Trainer agent;	 // AGENT
    double best_fitness;
    Network::Network best_network;
    //
    public:
    Agent() {
        agent.setup();
    }
    // send state to agent
    void state(std::vector <double> in) {
        agent.get_state(in);
    }
    // get action from agent
    std::vector <double> action() {
        std::vector <double> out;
        out = agent.give_action();
        return out;
    }
    // send fitness to agent
    void fitness(std::vector <double> in) {
        agent.get_fitness(in);
    }
    // data from system; update agent
    void update_in(std::vector <double> in) {
        agent.update_in(in);
    }
    // data from agent; update system
    std::vector <double>  update_out() {
        std::vector <double> u;
        u = agent.update_out();
        return u;
    }
};

#endif