/*

System 1.1
agent <> domain

William Gregory
copyright 2017 (c) all rights reserved

date: 12/20/2016 to Present

---------------------------------

info:
- Agent
- Domain
- SystemSingle
- SystemTrainer
- System

about:
- multi agent; single domain
- 'updates' allow communication between agent, domain, and system
- each agent is treated the same besides an optional 'action' weight

cycle:
- domain state > agent(s)
- compile agent(s) actions using weights
- action > domain
- domain fitness > agent(s)
- posible configuration change or end of program

setup:
- single agent; single objective

inline note format:
- DOMAIN : domain specific
- FIX : needs work
- ERROR : current error point

---------------------------------

status: keep updated log here
-- serate SystemSingle and SystemTrainer
-- SystemTrainer uses SystemSingle

- next step is merging the trainer into the system.
- - no seperate trainer class.  manage the population of agents from the system.
- - posible Evolutionary Algorithm seperate from system for easy modification.
- - consider alternitives to currention 'update_in/out' method.
- - testing seperate trainer; no mutation; only running population at first index.
- - note: single run agents are wrong.  both should be agents, no trainer.

---------------------------------

todo:
- add data output to system
- add round to system - consider trainer
- check update in/out for trainer and cart_balance
- add needed local values (consider GUI) updated with 'agent_update()'
- add test best network function to trainer
- add trainer domain.exort_all_states() control
- efnorce namespace and class standards to prevent code modification need
- how to handle which network is being tested in agent
- - instead of local loop, checkin and increment with report
- - add new domain option from agent
- update Makefile so only modified files are compiled

ideas:
- system has two modes (single and train (multiple)) alternate main run code
- - make network_trainer a general purpose 'agent trainer'
- - incorperate trainer into the system
- add class for iostream use (error, debug, ...)
- create individual test files for agent, domain and system
- - see _testing folder

---------------------------------

//double progress = (double)in_cr/in_rm;

*/

#ifndef _SYSTEM_
#define _SYSTEM_

#include "utilities.hpp"

#include "agent.hpp"
#include "domain.hpp"

namespace System {

	/*
	 *	State: system state container
	 */
	struct StateSingle {
		std::vector <double> state;  // from domain
		std::vector <std::vector <double> > action_all;  // from agent
        std::vector <double> action_sum;
		std::vector <double> fitness_all;
        double fitness; // sum of fitness_all (cycle specific)
		//
		std::vector <double> domain_update;	 // from domain
		unsigned int cycle_current;
	};

    /*
     *	SystemSingle
     */
	class SystemSingle {
        std::vector <StateSingle> states;  // collection of states
        std::vector <Agent> agents;
        Domain domain;
        std::vector <double> action_weights;
        unsigned int round_max;
        double fitness; // sum of all states fitness - result
	public:
		// in: agents, domain, cycles, action weights
        void setup_single(std::vector<Agent> in_a, Domain in_d, unsigned int in_c, std::vector <double> in_w) {
#ifdef S_DEBUG
			std::cout << "DEBUG:SYSTEM:SINGLE:SETUP" << std::endl;
#endif
            agents = in_a;
            domain = in_d;
            cycle_max = in_c;
            action_weights = in_w;
        }
        // return fitness
		double run() {
#ifdef S_DEBUG
			std::cout << "DEBUG:SYSTEM:SINGLE:RUN" << std::endl;
#endif
			for (std::size_t i=0; i<round_max; ++i) {
				// new system state
				State s;
				state_last = s;
				state_last.round_current = round_current;
				// get domain state
				state_last.state = domain.state();
				// manage agents
				for (std::size_t i=0; i<agents.size(); ++i) {
					// give state
					agents.at(i).state(state_last.state);
					// get action
					state_last.action_all.push_back(agents.at(i).action());
				}
				// compile actions
				for (std::size_t i=0; i<state_last.action_all.size(); ++i) {
					for (std::size_t j=0; j<state_last.action_all.at(i).size(); ++j) {
						if (i == 0) state_last.action_sum.push_back(0.0);
						// sum actions from each agent with agent action weight
						state_last.action_sum.at(j) += state_last.action_all.at(i).at(j) * action_weights.at(i);
					}
				}
				// domain action
				domain.action(state_last.action_sum);
				// domain fitness
				state_last.fitness = domain.fitness();
				// round cleanup
				state_all.push_back(state_last);
			}
			return fitness;
		}
	};

    /*
     *	SystemTrainer
     */
	class SystemTrainer {
    private:
        std::vector <std::vector <Agent> > population;   // groups of agents
        std::vector <double> fitness; // double per group
        //
        unsigned int pop_size;  // population size
        unsigned int agent_count;  // agents in a group
        unsigned int cycle_max;
	public:
		// in: pop size, agent_count, cycles
        void setup(unsigned int in_pop_size, unsigned int in_agent_count, unsigned int in_cycle) {
            // parameters
            pop_size = in_pop_size;
            agent_count = in_agent_count;
            cycle_max = in_cycle;
            cycle_current = 0;
            // create population
            population.clear();
            for (std::size_t p=0; p<pop_size; ++p) {
                std::vector <Agent> t;
                for (std::size_t i=0; i<agent_count; ++i) {
                    Agent a;
                    t.push_back(a);
                }
                population.push_back(t);
            }
        }
		void run() {
#ifdef S_DEBUG
			std::cout << "S_DEBUG: run trainer" << std::endl;
#endif
			for (std::size_t i=0; i<cycle_max; ++i) {
				for (std::size_t p=0; p<population.size(); ++p) {
					SystemSingle s;
					std::vector <double> tw;
					for (std::size_t w=0; w<agent_count; ++w) {
						tw.push_back(1.0); // FIX - pre-define
					}
					Domain td;
					// in: agents, domain, cycles, action weights
					s.setup(population.at(p), td, 100, tw);
					fitness.push_back(s.run());
				}
				// improvement algorithm (EA) - given fitness and population
				IA ia;
				population = ia.cycle(population, fitness);
			}
		}
	};

	/*
	 *	System
	 */
	class System {
    private:
		unsigned int type_run;  // run type (0:single, 1:trainer)
		unsigned int agent_count; // total number of agents
		//
		double best_fitness;
        std::vector <Agent> best_agents;
    public:
		void setup(unsigned int in_rt) {
#ifdef S_DEBUG
			std::cout << "DEBUG:SYSTEM:SETUP" << std::endl;
#endif
			run_type = in_rt;
		}
		void run_single() {
			SystemSingle s;
			// in: agents, domain, cycles, action weights
			//s.setup();
			std::cout << "NOTE:SYSTEM:RUN_SINGLE - not implemented" << std::endl;
		}
		void run_trainer() {
			SystemTrainer t;
			// in: pop size, agent_count, cycles
			t.setup(100, 2, 100);
			t.run();
		}
		void run() {
#ifdef S_DEBUG
			std::cout << "DEBUG:SYSTEM:RUN - start" << std::endl;
#endif
			if (run_type == 1) run_single();
			else if (run_type == 2) run_trainer();
			else std::cout << "ERROR:SYSTEM:RUN - unknown run_type" << std::endl;
#ifdef S_DEBUG
			std::cout << "DEBUG:SYSTEM:RUN - complete" << std::endl;
#endif
		}
	};

}

#endif
