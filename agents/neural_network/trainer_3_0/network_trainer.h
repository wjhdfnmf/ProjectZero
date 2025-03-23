
/*

Network Trainer
Version 3.0

Created by William Gregory on October 27th 2016 to present.
Copyright (c) William Gregory.  All rights reserved.

*/

#ifndef _NETWORK_TRAINER_
#define _NETWORK_TRAINER_

//#define NT_DEBUG
//#define NT_VERBOSE
//#define NT_TEST

#include <sstream>

#include "../network/neural_network.h"  // agent
#include "../../../domains/_not_mine/cart_balance/cart_balance.h"  // DOMAIN SPECIFIC

namespace Trainer
{
	struct DomainState {
		double theta;			// inital theta 			[degrees]
		double mass_p;			// mass of pendulum 		[kg]
		double length;			// length of pendulum		[m]
		double mass_c;			// mass of cart 			[kg]
		double track_length;	// length of track 			[m]
	};

	class Trainer
	{
	private:
        std::vector <Network::Network> population;
		std::vector <double> pop_fitness; // .at(population)
        double best_fitness;
        Network::Network best_network;
		double first_best_fitness;
		double last_best_fitness;
		double last_avg_fitness;
		std::vector <std::vector <double> > fitness_history;  // .at(round).at(population)  // reserve?
        unsigned int current_round;
		unsigned int ID_next;
		unsigned int current_network;  // current
		CB::Pendulum domain;  // domain // DOMAIN SPECIFIC
		bool runtime_error;
		double delta_time;
		std::vector <double> max_output;
		std::vector <double> last_state;
		std::vector <double> last_action;
		double last_fitness;

        void print_intro();
        void print_end();

        void get_state();
        void cycle_domain();

		void progress();
		void export_fitness_history();
		CB::Pendulum generate_domain();  // DOMAIN SPECIFIC
		Network::Network generate_network();
		void generate_population();
		std::vector <double> cycle_network(std::vector <double>&, Network::Network&);
		void cycle();
		void error_manager();
		void run_best_network();

        void prune();
        void populate();
    public:
        unsigned int round_max;
        unsigned int population_size;
        std::vector <unsigned int> nodes_per_layer;
		double mutate_mod;
		double mutate_chance;
		double max_torque;
		double max_force;
		unsigned int test_count;  // domain cycles
		unsigned int input_layer_size;
		unsigned int hidden_layer_size;
		unsigned int output_layer_size;
		DomainState init_d;	// initial domain state

        Trainer();
        void train();
    };

    Trainer::Trainer() : domain(1000) {
#ifdef NT_DEBUG
		std::cout << "debug: Trainer()" << std::endl;
#endif

		// NETWORK SETTINGS
		test_count = 200;			// network itterations
		round_max = 1000;			// rounds of simulation 	[dt]
		population_size = 100;
		mutate_mod = 0.2;
		mutate_chance = 0.1;

		max_torque = 0.0;			// 							[N*m]
		max_force = 40.0;  			// 							[N]


		DomainState domain;
		// DOMAIN SETTINGS
		domain.theta = 95;			// inital theta 			[degrees]
		domain.mass_p = 1;			// mass of pendulum 		[kg]
		domain.length = 1;			// length of pendulum		[m]
		domain.mass_c = 2;			// mass of cart 			[kg]
		domain.track_length = 10;	// length of track 			[m]

		init_d = domain;

		// do not modify
		population.reserve(population_size);
		pop_fitness.reserve(population_size);
		srand(time(0));
		runtime_error = false;
        current_round = 0;
		ID_next = 1;
		best_fitness = HUGE_VAL;
		input_layer_size = 3;
		hidden_layer_size = 4; // ADJUSTABLE
		output_layer_size = 2;
		nodes_per_layer.clear();
		nodes_per_layer.push_back(input_layer_size);
		nodes_per_layer.push_back(hidden_layer_size);
		nodes_per_layer.push_back(output_layer_size);
		max_output.push_back(max_torque);
		max_output.push_back(max_force);
    }

    void Trainer::print_intro() {
#ifdef NT_DEBUG
		std::cout << "debug: print_intro()" << std::endl;
#endif
        std::cout << std::endl;
        std::cout << "Network Trainer" << std::endl;
        std::cout << "version 3.0" << std::endl;
		std::cout << std::endl;
		std::cout << "Domain: Cart Balance by sierra121314" << std::endl;
        std::cout << "--------------------" << std::endl;
        std::cout << std::endl;
    }

    void Trainer::print_end() {
#ifdef NT_DEBUG
		std::cout << "debug: print_end()" << std::endl;
#endif
        std::cout << std::endl;
		std::cout << std::endl;
		if (delta_time > 60) {
			unsigned int t_min = (unsigned int)(delta_time/60);
			std::cout << "run time: " << t_min << "m " << delta_time-60*t_min << "s" << std::endl;
		} else {
			std::cout << "run time: " << delta_time << "s" << std::endl;
		}
        std::cout << "best fitness: " << best_fitness << std::endl;
		std::cout << "inital best fitness: " << first_best_fitness << std::endl;
		std::cout << "last best fitness: " << last_best_fitness << std::endl;
    }

    //----------------------------
    // get state from domain
    void Trainer::get_state() {
#ifdef NT_DEBUG
		std::cout << "debug: get_state()" << std::endl;
#endif
		std::vector <double> t_state = domain.give_state();  // theta, theta dot
		last_state.clear();
		last_state.push_back(cos(t_state.at(0)));  // x
		last_state.push_back(sin(t_state.at(0)));  // y
		last_state.push_back(t_state.at(1));  // theta dot
    }

    // give action to domain
    void Trainer::cycle_domain() {
#ifdef NT_DEBUG
		std::cout << "debug: cycle_domain()" << std::endl;
#endif
		last_fitness = domain.get_action(last_action).at(0);
		pop_fitness.at(current_network) = pop_fitness.at(current_network) + last_fitness;
    }

    //-----------------------------

	//
	// progress: sub
	//
	void Trainer::progress()
	{
		unsigned int update_interval = 100.0;  // setting
		double progress = (double)current_round/round_max;
		unsigned int t_progress = int(progress*10000);
		if (t_progress % update_interval == 0) {
			
			unsigned int bar_width = 40;  // setting
			std::cout << "[";
			unsigned int bar_pos = bar_width * progress;
			unsigned int bar_pos_em = bar_width - bar_pos;
			for (std::size_t i=1; i<=bar_pos; ++i) {
				if (i < bar_pos) std::cout << "=";
				else std::cout << ">";
			}
			for (std::size_t i=0; i<bar_pos_em; ++i) {
				std::cout << " ";
			}
			std::cout << "] " << int(progress * 100) << "%" << "\r";
			std::cout.flush();
		}
	}

	// export all fitness history to file
	void Trainer::export_fitness_history() {
#ifdef NT_DEBUG
		std::cout << "debug: export_fitness_history()" << std::endl;
#endif
		std::ofstream file;
		file.open("./outputs/fitness_history.csv",
			std::ofstream::out | std::ofstream::trunc);
		if (!file.is_open()) {
			std::cerr << "ERROR: could not open file for writing" << std::endl;
			return;
		}
		for (std::size_t i=0; i<fitness_history.size(); ++i) {
			for (std::size_t j=0; j<fitness_history.at(i).size(); ++j) {
				if (j!=0) file << ", ";
				file << fitness_history.at(i).at(j);
			}
			if (i!=fitness_history.size()) file << "\n";
		}
		file.close();
	}

	////////////////////////////////////////
	// generate single domain and return
	CB::Pendulum Trainer::generate_domain()
	{
#ifdef NT_DEBUG
		std::cout << "debug: generate_domain()" << std::endl;
#endif
		// variables
		CB::Pendulum pend(test_count, init_d.theta, init_d.mass_p, init_d.length,
			max_torque, max_force, init_d.mass_c, init_d.track_length);
		return pend;
	}

	////////////////////////////////////////
	// generate a network, setup and return
	Network::Network Trainer::generate_network()
	{
#ifdef NT_DEBUG
		std::cout << "debug: generate_network()" << std::endl;
#endif
		Network::Network net;
		net.ID_value = ID_next;
		net.run_type = 1;
		net.setup(nodes_per_layer, mutate_mod, mutate_chance);
		ID_next++;
		return net;
	}

	////////////////////////////////////////
	// generate starting population with zero fitness
	void Trainer::generate_population()
	{
#ifdef NT_DEBUG
		std::cout << "debug: generate_population()" << std::endl;
#endif
		population.clear();
		pop_fitness.clear();
		for (std::size_t i=0; i<population_size; ++i) {
			population.push_back(generate_network());
			pop_fitness.push_back(0.0);
		}
	}

	////////////////////////////////////////
	// Processes an input vector through a neural network and returns the scaled output.
	// This function takes an input vector and a network, cycles the input through the network
	// to produce an output, and scales the output values based on predefined maximums.
	// Includes debug and test checks for input/output size consistency.
	std::vector <double> Trainer::cycle_network(
		std::vector <double>& in_val,
		Network::Network& in_net
	) {
#ifdef NT_DEBUG
		std::cout << "debug: cycle_network()" << std::endl;
#endif
#ifdef NT_TEST
		if (in_val.size() != input_layer_size) runtime_error = true; // ERROR - input size mismatch
#endif
		// cycle network
		std::vector <double> t_out;
		t_out = in_net.cycle(in_val);
#ifdef NT_TEST
		if (t_out.size() != max_output.size()) runtime_error = true; // ERROR - output size mismatch
#endif
		// scale output
		for (std::size_t i=0; i<t_out.size(); ++i) {
			t_out.at(i) = t_out.at(i) * max_output.at(i);
		}
		return t_out;
	}

	////////////////////////////////////////
	// cycle agent - domain for 'test_count' iterations
	void Trainer::cycle() {
#ifdef NT_DEBUG
		std::cout << "debug: cycle()" << std::endl;
#endif
		for (std::size_t i = 0; i<test_count; ++i) {
			get_state();
			last_action = cycle_network(
				last_state,
				population.at(current_network));
			cycle_domain();
#ifdef NT_VERBOSE
			// info
			std::cout << "current round: " << current_round << std::endl;
			std::cout << "current network: " << current_network << std::endl;
			std::cout << "current test: " << i << std::endl;
			// states
			std::cout << "states: " << " ";
			for (std::size_t i=0; i<last_state.size(); ++i)
			{
				std::cout << last_state.at(i) << " ";
			}
			std::cout << std::endl;
			// actions
			std::cout << "actions: " << " ";
			for (std::size_t i=0; i<last_action.size(); ++i)
			{
				std::cout << last_action.at(i) << " ";
			}
			std::cout << std::endl;
#endif
		}
	}

	////////////////////////////////////////
	// input: pop_fitness vector
	// find average fitness for all domain rounds
	// find average fitness for each network
	void Trainer::error_manager() {
#ifdef NT_DEBUG
		std::cout << "debug: error_manager() start" << std::endl;
#endif
		// average fitness for each
		std::vector <double> temp_avg_fitness_all;
		double temp_best_fitness = HUGE_VAL; // for this round
		for (std::size_t i=0; i<pop_fitness.size(); ++i) {
			temp_avg_fitness_all.push_back(pop_fitness.at(i) / test_count);
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
		// log best fitness
		last_best_fitness = temp_best_fitness;
		if (current_round == 0) first_best_fitness = temp_best_fitness;
		// add to fitness history
		if (current_network == population_size-1) fitness_history.push_back(temp_avg_fitness_all);
		// total average fitness
		double temp_avg_fitness = 0.0;
		for (std::size_t i=0; i<temp_avg_fitness_all.size(); ++i) {
			temp_avg_fitness += temp_avg_fitness_all.at(i);
		} temp_avg_fitness = temp_avg_fitness / temp_avg_fitness_all.size();
		// reset 'pop_fitness'
		pop_fitness.clear();
		for (std::size_t i=0; i<population_size; ++i) {
			pop_fitness.push_back(0.0);
		}
	}

	////////////////////////////////////////
	// run best network and export states
	void Trainer::run_best_network() {
#ifdef NT_DEBUG
		std::cout << "debug: run_best_network() start" << std::endl;
#endif
		domain = generate_domain();
		for (std::size_t i = 0; i<test_count; ++i) {
			get_state();
			last_action = cycle_network(last_state, best_network);
			cycle_domain();
		}
		domain.export_all_states();

		std::vector <std::vector <std::vector <double> > > data = best_network.export_weights();

		/*
		std::ofstream out("best_network.bin", std::ios_base::binary);
		size_t X = data.size(), Y = X ? data[0].size() : 0, Z = Y ? data[0][0].size() : 0;
		out.write(reinterpret_cast<const char*>(&X), sizeof(X));
		out.write(reinterpret_cast<const char*>(&Y), sizeof(Y));
		out.write(reinterpret_cast<const char*>(&Z), sizeof(Z));
		for (const auto& outer : data) {
			for (const auto& middle : outer) {
				out.write(reinterpret_cast<const char*>(middle.data()), Z * sizeof(double));
			}
		}
		*/
	}

    ////////////////////////////////////////
	// Prunes the population by selecting the fittest individuals.
	// This function reduces the population size by half, comparing pairs of randomly
	// selected networks based on their error values and keeping the one with the lower
	// error, effectively implementing a tournament-style selection process.
    void Trainer::prune() {
#ifdef NT_DEBUG
		std::cout << "debug: prune()" << std::endl;
#endif
		int try_1;
		int try_2;
		std::vector <Network::Network> temp_new_pop;
		for (std::size_t i = 0; i < population.size() / 2; ++i)
		{
			try_1 = rand() % population.size();
			try_2 = rand() % population.size();
			while (try_1 == try_2)
				try_2 = rand() % population.size();
			if (pop_fitness.at(try_1) < pop_fitness.at(try_2))
				temp_new_pop.push_back(population.at(try_1));
			else
				temp_new_pop.push_back(population.at(try_2));
		}
		population = temp_new_pop;
    }

    ////////////////////////////////////////
	// Repopulates the given population to reach the desired size.
	// This function takes an existing population of networks and adds new networks
	// by duplicating random individuals, importing their weights, and applying mutations,
	// ensuring the population size matches the specified target.
    void Trainer::populate() {
#ifdef NT_DEBUG
		std::cout << "debug: populate()" << std::endl;
#endif
		int copy_index;
		Network::Network net;
		std::vector <Network::Network> temp_new_pop = population;
		while (temp_new_pop.size() < population_size)
		{
			copy_index = rand() % population.size();
			// generate a new network
			net = generate_network();
			// set to same weight as random
			net.import_weights(population.at(copy_index).export_weights());
			net.mutate();
			temp_new_pop.push_back(net);
		}
		population = temp_new_pop;
    }

	////////////////////////////////////////
	// Trains the population of neural networks over multiple rounds.
    // This function implements an evolutionary training loop, generating a population,
    // evaluating fitness, pruning weak performers, repopulating, and tracking progress.
    // It also measures execution time and exports results for analysis.
    void Trainer::train() {
#ifdef NT_DEBUG
		std::cout << "debug: train()" << std::endl;
#endif
		print_intro();
		generate_population();
		clock_t time_start = clock();
		while (current_round <= round_max && !runtime_error)
		{
#ifdef NT_VERBOSE
			std::cout << "debug: round " << current_round << " start" << std::endl;
#endif
			for (std::size_t i=0; i<population_size; ++i)
			{
				current_network = i;
				domain = generate_domain();
				cycle();
			}
			// finish training round
			error_manager();
			prune();
            populate();
			// progress
			++current_round;
			progress();
        }
		export_fitness_history();
		run_best_network();
        delta_time = (clock() - time_start) / CLOCKS_PER_SEC;
		print_end();
    }

}

#endif
