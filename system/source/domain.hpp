// William Gregory (c) Copyright 2017 - All rights reserved

#ifndef _DOMAIN_
#define _DOMAIN_

#include <vector>
#include "domain/pendulum.hpp"	 // DOMAIN

/*
*	domain class - "port" between domain and system
*/
class Domain {
    private:
    InvPend::Pendulum domain; // DOMAIN

    public:
    Domain() {
    unsigned int test_count = 100;
        domain.setup(test_count, 89.0);	 // FIX - need test count from agent
    }
    // return domain state
    std::vector <double> state() {
        std::vector <double> out;
        out = domain.give_state();
        return out;
    }
    // set domain action
    void action(std::vector <double> in) {
        domain.get_action(in);
    }
    // return domain fitness // just double?
    std::vector <double> fitness() {
        std::vector <double> out;
        out = domain.give_fitness();
        return out;
    }
    // data from system; update domain
    void update_in(std::vector <double> in) {
        domain.update_in(in);
    }
    // data from domain; update system
    std::vector <double> update_out() {
        std::vector <double> u;
        u = domain.update_out();
        return u;
    }
};

#endif