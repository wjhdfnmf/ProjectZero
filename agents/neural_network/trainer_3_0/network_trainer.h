
/*

Network Trainer
Version 3.0

Created by William Gregory on October 27th to present.
Copyright (c) William Gregory.  All rights reserved.

*/

#ifndef _NETWORK_TRAINER_
#define _NETWORK_TRAINER_

//#define NT_DEBUG
//#define NT_VERBOSE

#define PI 3.14159265

#include "../network/neural_network.h"  // agent
#include "../../../domains/_not_mine/cart_balance/cart_balance.h"  // DOMAIN SPECIFIC

namespace Trainer
{
	class Trainer
	{
	private:
        std::vector <Network::Network> population;
		std::vector <double> pop_fitness;
        double best_fitness;
        Network::Network best_network;
		double first_best_fitness;
		double last_best_fitness;
		double last_avg_fitness;
		std::vector <std::vector <double> > fitness_history;  // .at(round).at(population)  // reserve?
        unsigned int current_round;
		unsigned int ID_next;
		unsigned int network_test_count;  // current
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
        void give_action();
        bool get_reward();

		void progress();
		void export_fitness_history();
		CB::Pendulum generate_domain();  // DOMAIN SPECIFIC
		Network::Network generate_network();
		void generate_population();
		std::vector <double> cycle_network(std::vector <double>&, Network::Network&);
		void log_reward(double&, unsigned int&);
		void cycle();
		void error_manager(std::vector <double>&);
		void run_best_network();

        std::vector <Network::Network> prune(std::vector <Network::Network>&, std::vector <double>&);
        std::vector <Network::Network> populate(std::vector <Network::Network>&, unsigned int&);
    public:
        unsigned int round_max;
        unsigned int population_size;
        std::vector <unsigned int> nodes_per_layer;
		double mutate_mod;
		double mutate_chance;
		unsigned int test_count;  // domain cycles
		unsigned int input_layer_size;
		unsigned int hidden_layer_size;
		unsigned int output_layer_size;

        Trainer();
        void train();
    };

    Trainer::Trainer() {
#ifdef NT_DEBUG
		std::cout << "debug: Trainer() start" << std::endl;
#endif
		// settings
		test_count = 400; // network/domain cycles
		round_max = 100;
		population_size = 100;
	    hidden_layer_size = 8;
	    mutate_mod = 0.1;
	    mutate_chance = 0.3;
		double max_torque = 1.0;  // DOMAIN SPECIFIC
		// end settings
		// do not modify
		population.reserve(population_size);
		pop_fitness.reserve(population_size);
		srand(time(0));
		runtime_error = false;
        current_round = 0;
		ID_next = 1;
		best_fitness = HUGE_VAL;
		input_layer_size = 3;  // fixed
		output_layer_size = 1; // fixed
		nodes_per_layer.clear();
		nodes_per_layer.push_back(input_layer_size);
		nodes_per_layer.push_back(hidden_layer_size);
		nodes_per_layer.push_back(output_layer_size);
		max_output.push_back(max_torque);  // DOMAIN SPECIFIC
    }

    void Trainer::print_intro() {
#ifdef NT_DEBUG
		std::cout << "debug: print_intro() start" << std::endl;
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
		std::cout << "debug: print_end() start" << std::endl;
#endif
        std::cout << std::endl;
        std::cout << "--------------------" << std::endl;
		if (delta_time > 60) {
			unsigned int t_min = (unsigned int)(delta_time/60);
			std::cout << "run time: " << t_min << "m " << delta_time-60*t_min << "s" << std::endl;
		} else
			std::cout << "run time: " << delta_time << " s" << std::endl;
		std::cout << "training complete" << std::endl;
        std::cout << "best fitness: " << best_fitness << std::endl;
		std::cout << "inital best fitness: " << first_best_fitness << std::endl;
		std::cout << "last best fitness: " << last_best_fitness << std::endl;
        std::cout << std::endl;
    }

    //----------------------------
    // get state from domain
    void Trainer::get_state() {
#ifdef NT_DEBUG
		std::cout << "debug: get_state() start" << std::endl;
#endif
		std::vector <double> t_state = domain.give_state();  // theta, theta dot
		last_state.clear();
		last_state.push_back(cos(t_state.at(0)));  // x
		last_state.push_back(sin(t_state.at(0)));  // y
		last_state.push_back(t_state.at(1));  // theta dot
    }

    // give action to domain
    void Trainer::give_action() {
#ifdef NT_DEBUG
		std::cout << "debug: give_action() start" << std::endl;
#endif
		domain.get_action(last_action);
    }

    bool Trainer::get_reward() {
#ifdef NT_DEBUG
		std::cout << "debug: get_reward() start" << std::endl;
#endif
		last_fitness = domain.give_reward().at(0);
		return domain.return_below_horizontal();
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
		std::ofstream file;
		file.open("fitness_history.csv", std::ofstream::out | std::ofstream::trunc);
		for (std::size_t i=0; i<fitness_history.size(); ++i) {
			for (std::size_t j=0; j<fitness_history.at(i).size(); ++j) {
				if (j!=0) file << ", ";
				file << fitness_history.at(i).at(j);
			}
			if (i!=fitness_history.size()) file << "\n";
		}
		file.close();
	}

	// generate single domain and return
	CB::Pendulum Trainer::generate_domain()
	{
#ifdef NT_DEBUG
		std::cout << "debug: generate_domain() start" << std::endl;
#endif
		CB::Pendulum pend(test_count);  // destructor needed?
		return pend;
	}

	//
	// generate_network: sub
	// generate a network, setup and return
	//
	// output:
	// - output generally configured network
	//
	Network::Network Trainer::generate_network()
	{
#ifdef NT_DEBUG
		std::cout << "debug: generate_network() start" << std::endl;
#endif
		Network::Network net;
		net.ID_value = ID_next;
		net.run_type = 1;
		net.setup(nodes_per_layer, mutate_mod, mutate_chance);
		ID_next++;
		return net;
	}

	//
	// generate_population: sub
	// generate starting population
	// fill pop_fitness with zeros
	//
	void Trainer::generate_population()
	{
#ifdef NT_DEBUG
		std::cout << "debug: generate_population() start" << std::endl;
#endif
		population.clear();
		pop_fitness.clear();
		for (std::size_t i=0; i<population_size; ++i) {
			population.push_back(generate_network());
			pop_fitness.push_back(0.0);
		}
	}

	// cycle network with given inputs and return outputs
	std::vector <double> Trainer::cycle_network(std::vector <double>& in_val, Network::Network& in_net) {
#ifdef NT_DEBUG
		std::cout << "debug: cycle_network() start" << std::endl;
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

	// add last fitness to pop_fitness
	void Trainer::log_reward(double& in_val, unsigned int& in_count) {
#ifdef NT_DEBUG
		std::cout << "debug: log_reward() start" << std::endl;
#endif
		pop_fitness.at(in_count) = pop_fitness.at(in_count) + in_val;
	}

	// cycle agent - domain for 'test_count' iterations
	void Trainer::cycle() {
#ifdef NT_DEBUG
		std::cout << "debug: cycle() start" << std::endl;
#endif
		bool t = false; // domain specific
		for (std::size_t i = 0; i<test_count; ++i) {
			get_state();
			last_action = cycle_network(last_state, population.at(network_test_count));
			give_action();
			t = get_reward();  // domain specific
			if(t) last_fitness += test_count-i-1 * 1000000.0;  // penalty for remaining tests
			log_reward(last_fitness, network_test_count);
			if (t) i = test_count;  // quit
		}
	}

	// input: pop_fitness vector
	// find average fitness for entire round
	// find average fitness for each network
	void Trainer::error_manager(std::vector <double>& in_fitness) {
		// average fitness for each
		std::vector <double> temp_avg_fitness_all;
		double temp_best_fitness = HUGE_VAL; // for this round
		for (std::size_t i=0; i<in_fitness.size(); ++i) {
			temp_avg_fitness_all.push_back(in_fitness.at(i) / test_count);
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
		if (network_test_count == population_size-1) fitness_history.push_back(temp_avg_fitness_all);
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

	void Trainer::run_best_network() {
		domain = generate_domain();
		for (std::size_t i = 0; i<test_count; ++i) {
			get_state();
			last_action = cycle_network(last_state, population.at(network_test_count));
			give_action();
		}
		domain.export_all_states();
	}

    //
	// prune: main
	// prune a population using binary selection
	//
	// input:
	// - population to prune
	// - population errors
	// output:
	// - pruned population
	//
    std::vector <Network::Network> Trainer::prune(std::vector <Network::Network>& in_pop, std::vector <double>& in_pop_error) {
#ifdef NT_DEBUG
		std::cout << "debug: prune() start" << std::endl;
#endif
		int try_1;
		int try_2;
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

    //
	// populate: main
	// populate a population with mutations of the existing
	//
	// note: seperate network generation
	// input:
	// - population to populate
	// - desired population size
	// output:
	// - populated population
	//
    std::vector <Network::Network> Trainer::populate(std::vector <Network::Network>& in_pop, unsigned int& in_pop_size) {
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

	//
	// train: main
	// main training loop
	//
    void Trainer::train() {
#ifdef NT_DEBUG
		std::cout << "debug: train() start" << std::endl;
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
				network_test_count = i;
				domain = generate_domain();
				cycle();
			}
			error_manager(pop_fitness);
			population = prune(population, pop_fitness);
            population = populate(population, population_size);
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
