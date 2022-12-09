
/*

Network Trainer
Version 4.0

Created by William Gregory on December 24th to present.
Copyright (c) William Gregory.	All rights reserved.

Migrate to system

*/

#ifndef _NETWORK_TRAINER_
#define _NETWORK_TRAINER_

// manage in system
//#define NT_DEBUG
//#define NT_VERBOSE

#ifndef PI
#define PI 3.14159265
#endif

#include "neural_network.hpp"

namespace Trainer
{
class Trainer
{
private:
	//
	std::vector <Network::Network> population;
	std::vector <double> pop_fitness;
	std::vector <std::vector <double> > fitness_history;  // .at(round).at(population)
	//
	std::vector <double> max_output;
	unsigned int current_round;
	unsigned int network_test_count;  // current
	unsigned int domain_test_current;
	unsigned int ID_next;
	bool runtime_error;
	double delta_time;
	// state
	std::vector <double> last_state;
	std::vector <double> last_action;
	double last_fitness;
	//
	unsigned int round_max;
	unsigned int population_size;
	std::vector <unsigned int> nodes_per_layer;
	double mutate_mod;
	double mutate_chance;
	unsigned int domain_test_max;  // domain tests per sub-round
	unsigned int input_layer_size;
	unsigned int hidden_layer_size;
	unsigned int output_layer_size;
	//
	Network::Network best_network;
	double best_fitness;
	//-----------------------------

	// generate_network
	Network::Network generate_network()
	{
#ifdef NT_DEBUG
		std::cout << "debug: generate_network() start" << std::endl;
#endif
		Network::Network net;
		net.ID_value = ID_next;
		net.setup(nodes_per_layer, mutate_mod, mutate_chance);
		ID_next++;
		return net;
	}

	// generate_population
	std::vector <Network::Network> generate_population()
	{
		std::vector <Network::Network> t;
		for (std::size_t i=0; i<population_size; ++i) {
			t.push_back(generate_network());
		}
		return t;
	}

	// reset pop_fitness variable
	void reset_pop_fitness() {
		pop_fitness.clear();
		for (std::size_t i=0; i<population.size(); ++i) {
			pop_fitness.push_back(0.0);
		}
	}
	//-----------------------------

	// cycle network with given inputs and return outputs
	std::vector <double> cycle_network(std::vector <double>& in_val, Network::Network& in_net) {
#ifdef NT_DEBUG
		std::cout << "debug: cycle_network() start" << std::endl;
#endif
#ifdef NT_TEST
		if (in_val.size() != input_layer_size) {
			std::cout << "error: input to network wrong size" << std::endl;
		}
#endif
		// cycle network
		std::vector <double> t_out;
		t_out = in_net.cycle(in_val);
		return t_out;
	}

	// prune population
	std::vector <Network::Network> prune(std::vector <Network::Network>& in_pop, std::vector <double>& in_pop_error) {
#ifdef NT_DEBUG
		std::cout << "debug: prune() start" << std::endl;
#endif
		unsigned int try_1;
		unsigned int try_2;
		std::vector <Network::Network> temp_new_pop;
		for (std::size_t i = 0; i < in_pop.size() / 2; ++i)
		{
			try_1 = rand() % in_pop.size();
			try_2 = rand() % in_pop.size();
			while (try_1 == try_2)
				try_2 = rand() % in_pop.size();
			if (in_pop_error.at(try_1) < in_pop_error.at(try_2))
				temp_new_pop.push_back(in_pop.at(try_1));
			else
				temp_new_pop.push_back(in_pop.at(try_2));
		}
		return temp_new_pop;
	}

	// fill population
	std::vector <Network::Network> populate(std::vector <Network::Network>& in_pop, unsigned int& in_pop_size) {
#ifdef NT_DEBUG
		std::cout << "debug: populate() start" << std::endl;
#endif
		int copy_index;
		Network::Network net;
		std::vector <Network::Network> temp_new_pop = in_pop;
		while (temp_new_pop.size() < in_pop_size)
		{
			copy_index = rand() % in_pop.size();
			// generate a new network
			net = generate_network();
			// set to same weight as random
			net.import_weights(in_pop.at(copy_index).export_weights());
			net.mutate();
			temp_new_pop.push_back(net);
		}
		return temp_new_pop;
	}
	//----------------------------

	// input: pop_fitness vector
	// find average fitness for entire round
	// find average fitness for each network
	void fitness_manager(std::vector <double>& in_fitness) {
		// average fitness for each
		std::vector <double> temp_avg_fitness_all;
		double temp_best_fitness = HUGE_VAL; // for this round
		for (std::size_t i=0; i<in_fitness.size(); ++i) {
			temp_avg_fitness_all.push_back(in_fitness.at(i) / domain_test_max);
			// best fitness (overall)
			if (temp_avg_fitness_all.at(i) < best_fitness) {
				best_fitness = temp_avg_fitness_all.at(i);
				best_network = population.at(i);
			}
			// round best fitness
			if (temp_avg_fitness_all.at(i) < temp_best_fitness) {
				temp_best_fitness = temp_avg_fitness_all.at(i);
			}
		}
		// add to fitness history
		if (network_test_count == population_size-1) fitness_history.push_back(temp_avg_fitness_all);
		// total average fitness
		double temp_avg_fitness = 0.0;
		for (std::size_t i=0; i<temp_avg_fitness_all.size(); ++i) {
			temp_avg_fitness += temp_avg_fitness_all.at(i);
		} temp_avg_fitness = temp_avg_fitness / temp_avg_fitness_all.size();
		reset_pop_fitness();
	}

	// add last fitness to pop_fitness
	void log_reward(double& in_val, unsigned int& in_count) {
#ifdef NT_DEBUG
		std::cout << "debug: log_reward() start" << std::endl;
#endif
		pop_fitness.at(in_count) = pop_fitness.at(in_count) + in_val;
	}
	//-----------------------------

	// reset everything
	void reset_agent() {
		generate_population();
		network_test_count = 0;
	}

	// check state each population test cycle
	// advance counts as needed
	// true for continue
	bool check_state() {
		if (current_round > round_max) {
			//export_fitness_history();
			// run_best_network();
		}
		else if (!runtime_error) {
			//domain = generate_domain(); // new domain
			++network_test_count;
			// next round
			if (network_test_count == population_size) {
				fitness_manager(pop_fitness);
				population = prune(population, pop_fitness);
				population = populate(population, population_size);
				network_test_count = 0;
				++current_round;
			}
			return true;
		}
		return false;
	}
	//-----------------------------

public:

	Trainer() {
#ifdef NT_DEBUG
		std::cout << "debug: Trainer() start" << std::endl;
#endif
		setup();
	}

	void setup() {
		// settings
		domain_test_max = 400; // network/domain cycles
		round_max = 100;  // population evolutions
		population_size = 100;
		hidden_layer_size = 8;
		mutate_mod = 0.1;
		mutate_chance = 0.3;
		double max_torque = 1.0;  // DOMAIN SPECIFIC  // REMOVE - auto max 1
		// do not modify
		ID_next = 1;
		nodes_per_layer.clear();
		nodes_per_layer.push_back(3);  // input
		nodes_per_layer.push_back(hidden_layer_size);
		nodes_per_layer.push_back(1);  // output
		domain_test_current = 0;
		runtime_error = false;
		current_round = 0;
		max_output.push_back(max_torque);  // DOMAIN SPECIFIC
		best_fitness = HUGE_VAL;
		population = generate_population();
		population.reserve(population_size);
		pop_fitness.reserve(population_size);
	}

	// get state from domain
	// input: <theta, theta_dot>
	void get_state(std::vector <double> in) {
#ifdef NT_DEBUG
		std::cout << "debug: get_state() start" << std::endl;
#endif
		last_state.clear();
		last_state.push_back(cos(in.at(0)));  // x
		last_state.push_back(sin(in.at(0)));  // y
		last_state.push_back(in.at(1));	 // theta dot
		// cycle
		last_action = cycle_network(last_state, population.at(0));  // FIX - single network?????

	}

	// give action to domain
	std::vector <double> give_action() {
#ifdef NT_DEBUG
		std::cout << "debug: give_action() start" << std::endl;
#endif
		return last_action;
	}

	// get fitness
	void get_fitness(std::vector <double> in) {
#ifdef NT_DEBUG
		std::cout << "debug: get_reward() start" << std::endl;
#endif
		last_fitness = in.at(0);
		check_state();
	}
	//-----------------------------

	std::vector <double> update_out() {
		std::vector <double> t;
		// current fitness
		// generate new domain - new for new net
		if (true) t.push_back(1.0);	 // check count
		else t.push_back(0.0);
		return t;
	}


	void update_in(std::vector <double> in) {
		// first index - domain stop flag
		//// below horizontal
		if (in.at(0) == 1.0) { // stop
			last_fitness += domain_test_max-domain_test_current-1 * 1000000.0;	// penalty for remaining tests
			log_reward(last_fitness, network_test_count);
			domain_test_current = domain_test_max;
		}
		// second index - last round
		//if (in.at(1) == 1.0) {
			// export values
			// test best network
		//}
		// third index - next domain test?
		//if (in.at(2) == 1.0) {

		//}
	}
};
}

#endif
