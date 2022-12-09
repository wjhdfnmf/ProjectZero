/*
* Neural Network Trainer
*
* Version 2.1
*
* Creaded by William on July 8th, 2016 to Present
* Copyright (C) 2016 William.  All rights reserved
*
*/


/*
* To use:
* call setup()
* change params
* call train()
*
* Node:
* - train functions:
*	(0:x^2+1, 1:step, 2:exp(x), 3:sin(x), 4:cos(x), 5:const, 6:sqrt(x)+1, 7:-x)
*
* ToDo:
*	- add sweeping function to determine best values for...
*		- hidden_layer_size, mutate_mod, mutate_chance, sub_round_max, population_size
*	- add setting for activation type
*	- add setting for function to train
*	- add progress info (an updating bar would be cool)
*
*/

#ifndef _NETWORK_TRAINER_
#define _NETWORK_TRAINER_

//#define NT_DEBUG
//#define NT_VERBOSE
//#define NT_TEST

#define PI 3.14159265

//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <time.h>
//#include <string>
//#include <fstream> // file io
//#include <stdlib.h> // system

#include "../network/neural_network.h"

namespace Trainer
{
	class Trainer
	{
	private:
		// - variables
		// will be true if there is an error
		bool runtime_error;
		// collection of networks being trained
		std::vector <Network::Network> population;
		// nodes in each layer
		std::vector <unsigned int> nodes_per_layer;
		// inputs for the network this round
		std::vector <double> net_inputs;
		// expected outputs this round
		std::vector <double> net_expected_outputs;
		// actual outputs from the network this round
		std::vector <std::vector <double> > net_outputs;
		// unscaled net_inputs
		std::vector <double> net_inputs_raw;
		// unscaled net_expected_outputs
		std::vector <double> net_expected_outputs_raw;
		// sum of sub round errors for each network
		std::vector <double> sub_pop_round_errors;
		// avg of sub round errors for each network in population
		std::vector <double> round_pop_errors;
		// round average error of whole population
		double round_pop_avg_error;
		// first round average error
		double starting_avg_error;
		// current round	
		unsigned int round;
		// current sub round
		unsigned int sub_round;
		// current best error
		double best_error;
		// current best network
		Network::Network best_network;
		// next unique ID for network
		unsigned int ID_next;
		// script start time
		clock_t time_start;
		// script run time
		double time_delta;

		// - functions
		void error_call(std::string);
		void print_intro();
		void print_round();
		void print_end();
		void export_error(double, bool);
		void progress();
		void generate_population();
		Network::Network generate_network();
		double train_function(double&);
		double generate_value();
		double scale_value(double, bool, bool);
		void debug_check_network(Network::Network, std::string, bool);
		void generate_round_values();
		std::vector <std::vector <double> > cycle_network(std::vector <double>&, std::vector <Network::Network>&);
		double fitness_function(double, double);
		double check_network_error(std::vector <double>&, std::vector <double>&);
		void check_population_error(bool);
		std::vector <Network::Network> prune(std::vector <Network::Network>&, std::vector <double>&);
		std::vector <Network::Network> populate(std::vector <Network::Network>&, unsigned int&);

	public:
		// - variables
		// network input min value
		double input_min;
		// network input max value
		double input_max;
		// round max
		unsigned int round_max;
		// sub round max
		unsigned int sub_round_max;
		// max population size
		unsigned int population_size;
		// nodes in hidden layer
		unsigned int hidden_layer_size;
		// network mutation coefficient 
		double mutate_mod;
		// chance for mutating
		double mutate_chance;
		// train_func_type
		unsigned int train_func_type;

		// - functions
		Trainer();
		void train();
	};


	Trainer::Trainer()
	{
		// settings
		train_func_type = 0;
		population_size = 100;  // must be even and positive
		round_max = 400;  // must greator than zero
		sub_round_max = 60;  // must be greater than zero
		input_min = 1;
		input_max = 5;
		hidden_layer_size = 4;  // must be greater than zero
		mutate_mod = 0.1; // between zero and one
		mutate_chance = 0.5; // between zero and one
		// settings end
		srand(time(NULL));
		// do not modify
		runtime_error = false;
		round = 0;
		sub_round = 0;
		nodes_per_layer = { 1, hidden_layer_size, 1 };
		best_error = HUGE_VAL;
		ID_next = 1;
	}

	void Trainer::error_call(std::string in_message)
	{
		runtime_error = true;
		std::cout << "error: " << in_message << std::endl;
	}

	// general training info at start
	void Trainer::print_intro()
	{
		bool fancy = false;
		if (fancy) system("clear");
		std::cout << std::endl;
		std::cout << "------------------------" << std::endl;
		std::cout << "Network Trainer" << std::endl;
		std::cout << "by " << "WG" << std::endl;
		std::cout << std::endl;
		std::cout << "population size: " << population_size << std::endl;
		std::cout << "max round: " << round_max << std::endl;
		std::cout << "hidden nodes: " << hidden_layer_size << std::endl;
		std::cout << "mutate mod: " << mutate_mod << std::endl;
		std::cout << "input min: " << input_min << std::endl;
		std::cout << "input max: " << input_max << std::endl;
		std::cout << std::endl;
	}

	// current training info after each round
	void Trainer::print_round()
	{
		std::cout << "round: " << round << std::endl;
		//std::cout << "sub round: " << sub_round << std::endl;
		std::cout << "best_error: " << best_error << std::endl;
		std::cout << "average error: " << round_pop_avg_error << std::endl;
		std::cout << std::endl;
	}

	// training info after complete
	void Trainer::print_end()
	{
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "Training complete" << std::endl;
		if (time_delta > 60)
		{
			// convert to min and seconds
			int temp_min = (int)(time_delta / 60);
			int temp_sec = (int)(time_delta - temp_min * 60);
			std::cout << "time running: " << temp_min << " min " << temp_sec << " sec" << std::endl;
		}
		else {
			std::cout << "time running: " << time_delta << " seconds " << std::endl;
		}
		std::cout << "ending best error: " << best_error << std::endl;
		std::cout << "starting avgerage error: " << starting_avg_error << std::endl;
		std::cout << "ending average error: " << round_pop_avg_error << std::endl;
		std::cout << "------------------------" << std::endl;
		std::cout << std::endl;
	}
 
	void Trainer::export_error(double in, bool in_clear)
	{
		std::ofstream file;
		std::string filename = "output_results/avg_error.txt";
		if (in_clear) file.open(filename);
		else file.open(filename, std::ofstream::app);
		file << in << "\n";
	}

	//
	// progress: sub
	//
	void Trainer::progress()
	{
		unsigned int update_interval = 100.0;  // setting
		double progress = (double)round/round_max;
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

	//
	// generate_population: sub
	// generate starting population
	//
	void Trainer::generate_population()
	{
		population.clear();
		for (std::size_t i=0; i<population_size; ++i) {
			population.push_back(generate_network());
		}
		// output first network info
		std::string name = "first";
		debug_check_network(population.at(0), name, false);
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
		Network::Network net;

		net.ID_value = ID_next;
		net.run_type = 1;
		net.setup(nodes_per_layer, mutate_mod, mutate_chance);
		
		ID_next++;
		return net;
	}

	//
	// train_function: sub
	// returns network expected output given input value
	//
	// input mapped to output (x^2+1)
	// input:
	// - input to network
	// output:
	// - expected output of network
	//
	double Trainer::train_function(double& val)
	{
#ifdef NT_DEBUG
		std::cout << "train_function()" << std::endl;
#endif
		double t_out = 0;
		double t_half = input_min+(input_max-input_min)*0.5;
		// (0:x^2+1, 1:step, 2:exp(x), 3:sin(x), 4:cos(x), 5:const, 6:sqrt(x)+1, 7:-x)
		switch (train_func_type) {
			case 0:  // x^2+1
				t_out = (pow(val, 2) + 1.0);
				break;
			case 1:  // x>=0.5 return 1, x<0.5 return 0
				if (val > t_half) t_out = 2.0;
				else t_out = 1.0;
				break;
			case 2:  // exponential
				t_out = exp(val);
				break;
			case 3:  // sin(x)
				t_out = sin(val);
				break;
			case 4:  // cos(x)
				t_out = cos(val);
				break;
			case 5:  // constant
				t_out = 10.0;
				break;
			case 6:  // sqrt(x)+1
				t_out = std::sqrt(val)+1;
				break;
			case 7:  // (-x)
				t_out = -val;
				break;
			default:
				t_out = 0;
		}
		return t_out;
	}

	//
	// generate_value: sub
	// generate a single network input value within allowed range
	//
	// uses: input_min and input_max
	// output:
	// - network input value
	double Trainer::generate_value()
	{
#ifdef NT_DEBUG
		std::cout << "generate_value()" << std::endl;
#endif
		return (double)input_min + LYRAND*(input_max - input_min);
	}

	//
	// scale_value: sub
	// scales raw values to allowed network values (ie <= 1)
	// descales scaled values
	//
	// note: this may not be the best method
	// input:
	// - value to scale
	// - option: (true for input, false for output)
	// - option: (true for scale, false for descale)
	// output:
	// - scaled value
	//
	double Trainer::scale_value(double in_val, bool in_op, bool in_scale)
	{
#ifdef NT_DEBUG
		std::cout << "scale_value()" << std::endl;
#endif
		if (in_scale)
		{
			if (in_op)
			{
				return (in_val / input_max);
			}
			else {
				return (in_val / train_function(input_max));
			}
		}
		else {
			if (in_op)
			{
				return (in_val*input_max);
			}
			else {
				return (in_val*train_function(input_max));
			}
		}
	}

	//
	// debug_check_network: debug
	// test given network over range of values
	//
	void Trainer::debug_check_network(Network::Network in_pop, std::string in_name, bool print_weights)
	{
#ifdef NT_DEBUG
		std::cout << "debug_check_network() start" << std::endl;
#endif
		unsigned int test_points = 100;
		std::vector <double> temp_test_errors;
		std::vector <double> temp_inputs_log;
		std::vector <double> temp_outputs_log;
		std::vector <double> temp_ex_outputs_log;
		// network to vector of networks
		std::vector <Network::Network> temp_net_vec;
		temp_net_vec.push_back(in_pop);
		for (std::size_t i = 0; i < test_points; ++i)
		{
			std::vector <double> temp_inputs;
			std::vector <std::vector <double> > temp_outputs;
			std::vector <double> temp_ex_outputs;
			// generate values
			double temp_val = (double)input_min + (input_max - input_min) / test_points*i;  // even distribution
			temp_inputs.push_back(scale_value(temp_val, true, true));
			temp_ex_outputs.push_back(scale_value(train_function(temp_val), false, true));
			// cycle network
			temp_outputs = cycle_network(temp_inputs, temp_net_vec);
			// save for log
			temp_inputs_log.push_back(scale_value(temp_inputs.at(0), true, false));
			temp_outputs_log.push_back(scale_value(temp_outputs.at(0).at(0), false, false));
			temp_ex_outputs_log.push_back(scale_value(temp_ex_outputs.at(0), false, false));
			// check error
			temp_test_errors.push_back(check_network_error(temp_outputs.at(0), temp_ex_outputs));
		}
		// output for gnuplot (network_output_)
		std::ofstream file;
		std::string name = "output_results/network_output_" + in_name + ".txt";
		file.open(name);
		file << "#\tX\tY\n";
		for (std::size_t i = 0; i < test_points; ++i)
		{
			file << "\t" << temp_inputs_log.at(i); // inputs
			file << "\t" << temp_outputs_log.at(i); // outputs
			file << "\n";
		}
		file << "\n\n#\tX\tY\n";
		for (std::size_t i = 0; i < test_points; ++i)
		{
			file << "\t" << temp_inputs_log.at(i);  // inputs
			file << "\t" << temp_ex_outputs_log.at(i);  // expected outputs
			file << "\n";
		}
		file.close();
		// output general (network_info_)
		name = "output_results/network_info_" + in_name + ".csv";
		file.open(name);
		file << "inputs, outputs, expected, error" << "\n";
		for (std::size_t i = 0; i < test_points; ++i)
		{
			file << temp_inputs_log.at(i) << ", "; // inputs
			file << temp_outputs_log.at(i) << ", "; // outputs
			file << temp_ex_outputs_log.at(i) << ", "; // expected outputs
			file << temp_test_errors.at(i); // error
			file << "\n";
		}
		// output weights
		std::vector <std::vector <std::vector <double> > > temp_raw_weights = in_pop.export_weights();
		if (print_weights)
		{
			file << "\nweights:\n";
			for (std::size_t l = 0; l < temp_raw_weights.size(); ++l)
			{
				file << "layer :" << l << "\n";
				for (std::size_t n = 0; n < temp_raw_weights.at(l).size(); ++n)
				{
					file << "\tnode: " << n << "\n";
					for (std::size_t i = 0; i < temp_raw_weights.at(l).at(n).size(); ++i)
					{
						file << "\t\tid: " << i << "\n";
						file << "\t\t\t" << temp_raw_weights.at(l).at(n).at(i) << "\n";
					}
				}
			}
		}
		file.close();
	}

	//
	// generate_round_values: main
	// generate input and expected outputs for a round
	//
	// note: currently input and output layers must have same number of nodes
	// uses: nodes_per_layer, input_min and input_max
	// modifies: net_inputs and net_expected_outputs
	//
	void Trainer::generate_round_values()
	{
#ifdef NT_DEBUG
		std::cout << "debug: generate_round_values() start" << std::endl;
#endif
		// clear last rounds values
		net_inputs.clear();
		net_expected_outputs.clear();
		net_inputs_raw.clear();
		net_expected_outputs_raw.clear();

		// generate input/expected_output pair for each input node
		for (std::size_t i = 0; i < nodes_per_layer.at(0); ++i)
		{
			double t_input = generate_value();
			double t_output = train_function(t_input);
			net_inputs.push_back(scale_value(t_input, true, true));
			net_expected_outputs.push_back(scale_value(t_output, false, true));
			net_inputs_raw.push_back(t_input);
			net_expected_outputs_raw.push_back(t_output);
		}

#ifdef NT_TEST
		if (net_inputs.size() != nodes_per_layer.at(0))
		{
			error_call("generate_round_values() input size off");
		}
		if (net_expected_outputs.size() != nodes_per_layer.at(0))
		{
			error_call("generate_round_values() output size off");
		}
#endif
		/*
		 // error if using ex sin(x) because end points may be lower than somewhere else
		 for (int i=0; i<net_inputs.size(); i++)
		 {
		 if (net_inputs.at(i) > scale_value(input_max, true, true) ||
		 net_inputs.at(i) < scale_value(input_min, true, true))
		 {
		 error_call("generate_round_values() input value outside limits");
		 }
		 double temp_max_val = scale_value(train_function(input_max), false, true);
		 double temp_min_val = scale_value(train_function(input_min), false, true);
		 if (net_expected_outputs.at(i) > temp_max_val || net_expected_outputs.at(i) < temp_min_val)
		 {
		 error_call("generate_round_values() expected output value outside limits");
		 }
		 }*/
	}

	//
	// cycle_network: main, sub
	// cycle given networks with given inputs and return outputs
	//
	// note: seperate function for cycling one network
	// input:
	// - vector of inputs, one per input node
	// - vector of networks to cycle
	// output: vector (size of given population) vector (size of output node count)
	// - vector (size of given population) <vector (size of output node count) <double> >
	// - - vector of vector outputs
	// - - each network returns a vector of outputs, one per node
	//
	std::vector <std::vector <double> > Trainer::cycle_network(std::vector <double>& in_net_inputs, std::vector <Network::Network>& in_population)
	{
#ifdef NT_DEBUG
		std::cout << "debug: cycle_network() start" << std::endl;
#endif
		// set inputs and cycle
		std::vector <std::vector <double> > temp_net_outputs;
		for (std::size_t i = 0; i < in_population.size(); ++i)
		{
			temp_net_outputs.push_back(in_population.at(i).cycle(in_net_inputs));
		}
		return temp_net_outputs;
	}

	//
	// fitness function: sub
	// return fitness based on actual and expected network output
	//
	// input:
	// - actual network output
	// - expected network output
	// output:
	// - fitness value
	//
	double Trainer::fitness_function(double in_ac_out, double in_ex_out)
	{
		// (actual - expected)^2
		// lower is better
		//
		// ideas:
		// give more credit too end points
		// 20% reduction for farthest points
		// 50% to 100% mapped to 0% to 20%
		// 

		double fitness = pow(in_ac_out - in_ex_out, 2);
		return fitness;
	}

	//
	// check_network_error: sub
	// check single network output error
	// 
	// input:
	// - network output
	// - expected output
	// return:
	// - error for given network
	//
	double Trainer::check_network_error(std::vector <double>& in_real_out, std::vector <double>& in_ex_out)
	{

#ifdef NT_DEBUG
		std::cout << "debug: check_network_error() start" << std::endl;
#endif
#ifdef NT_TEST
		if (in_real_out.size() != in_ex_out.size())
		{
			error_call("check_network_error() size mismatch");
		}
#endif
		// check each output nodes error
		double temp_error = 0.0;
		for (std::size_t i = 0; i < in_real_out.size(); ++i)
		{
			temp_error += fitness_function(in_real_out.at(i), in_ex_out.at(i));
		}
		// return average error
		return temp_error / in_real_out.size();
	}

	//
	// check_population_error: main
	// check all networks error in population
	//
	// note: add population input and return error vector
	// input:
	// - option: (true for subround, false for round)
	// output:
	//  
	void Trainer::check_population_error(bool is_sub_round)
	{
#ifdef NT_DEBUG
		std::cout << "debug: check_population_error() start" << std::endl;
#endif
		if (is_sub_round)
		{
			// first sub round setup
			if (sub_round == 0)
			{
				sub_pop_round_errors.clear();
				for (std::size_t i = 0; i < population_size; ++i)
				{
					sub_pop_round_errors.push_back(0.0);
				}
			}
			// sub round error
			for (std::size_t i = 0; i < population_size; ++i)
			{
				sub_pop_round_errors.at(i) += \
					check_network_error(net_outputs.at(i), net_expected_outputs);
			}
			// round error
		}
		else {
			// reset round variables
			round_pop_errors.clear();
			round_pop_avg_error = 0.0;
			// take average of sub_round error
			double temp_round_pop_avg_error = 0.0;
			for (std::size_t i = 0; i < sub_pop_round_errors.size(); ++i)
			{
				double temp_error = sub_pop_round_errors.at(i) / sub_round_max;
				round_pop_errors.push_back(temp_error);
				temp_round_pop_avg_error += temp_error;
				// network best error
				if (temp_error < best_error)
				{
					// add: save network weights
					best_error = temp_error;
					best_network = population.at(i);
				}
			}
			round_pop_avg_error = temp_round_pop_avg_error / population_size;
			switch (round) {
				case 0:
					starting_avg_error = round_pop_avg_error;
			}
		}
	}

	//
	// prune: main
	// prune a population using binary selection
	//
	// idea: check average error after and it should be better than the starting average error
	// input:
	// - population to prune
	// - population errors
	// output:
	// - pruned population
	//
	std::vector <Network::Network> Trainer::prune(std::vector <Network::Network>& in_pop, std::vector <double>& in_pop_errors)
	{
#ifdef NT_DEBUG
		std::cout << "debug: prune() start" << std::endl;
#endif
		std::vector <Network::Network> temp_new_pop;
		double temp_avg_error = 0.0; // average error of new pop
		for (std::size_t i = 0; i < in_pop.size() / 2; ++i)
		{
			int try_1 = rand() % in_pop.size();
			int try_2 = rand() % in_pop.size();
			while (try_1 == try_2)
			{
				try_2 = rand() % in_pop.size();
			}
			if (in_pop_errors.at(try_1) < in_pop_errors.at(try_2)) {
				temp_new_pop.push_back(in_pop.at(try_1));
				temp_avg_error += round_pop_errors.at(try_1);
			}
			else {
				temp_new_pop.push_back(in_pop.at(try_2));
				temp_avg_error += round_pop_errors.at(try_2);
			}
		}
#ifdef NT_TEST
		if (temp_new_pop.size() != in_pop.size() / 2)
		{
			error_call("prune() output size wrong");
		}
#endif
#ifdef NT_VERBOSE
			temp_avg_error = temp_avg_error / temp_new_pop.size();
			std::cout << "prune results: " << std::endl;
			std::cout << " - start: " << round_pop_avg_error << std::endl;
			std::cout << " - end: " << temp_avg_error << std::endl;
			double prune_improvement = round_pop_avg_error - temp_avg_error;
			std::cout << " - improvement: " << prune_improvement << std::endl;
#endif
		//if (temp_avg_error > round_pop_avg_error)
		//{
			//error_call("prune() average error got worse");
		//}
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
	std::vector <Network::Network> Trainer::populate(std::vector <Network::Network>& in_pop, unsigned int& in_pop_size)
	{
#ifdef NT_DEBUG
		std::cout << "debug: populate() start" << std::endl;
#endif
#ifdef NT_TEST
		if (in_pop.size() < 2)
		{
			error_call("populate() input pop too small");
		}
		if (in_pop_size <= in_pop.size())
		{
			error_call("populate() input pop and desired size mismatch");
		}
#endif
		// populate until size is 'in_pop_size'
		std::vector <Network::Network> temp_new_pop = in_pop;
		while (temp_new_pop.size() < in_pop_size)
		{
			int copy_index = rand() % in_pop.size();
			// generate a new network
			Network::Network net = generate_network();
			// set to same weight as random
			net.import_weights(in_pop.at(copy_index).export_weights());
			net.mutate();
			temp_new_pop.push_back(net);
		}
#ifdef NT_TEST
		if (temp_new_pop.size() != in_pop_size)
		{
			error_call("populate() output population size wrong");
		}
#endif
		return temp_new_pop;
	}

	//
	// train: main
	// main training loop
	//
	void Trainer::train()
	{
#ifdef NT_DEBUG
		std::cout << "debug: train() start" << std::endl;
#endif
		print_intro();
		generate_population();
		time_start = clock();
		while (round <= round_max && runtime_error == false)
		{
#ifdef NT_DEBUG
			std::cout << "debug: round " << round << "; sub " << sub_round << " start" << std::endl;
#endif
			// generate values for this round
			generate_round_values();
			// cycle network
			net_outputs.clear();
			net_outputs = cycle_network(net_inputs, population);
			// check error
			check_population_error(true); // true: subround
										  // sub round complete
			sub_round++;
			if (sub_round == sub_round_max)
			{
				progress();
				check_population_error(false); // false: round
				population = prune(population, round_pop_errors);
				population = populate(population, population_size);
				//
				if (round == 0) export_error(round_pop_avg_error, true);
				else export_error(round_pop_avg_error, false);
				// round complete
				round++;
				sub_round = 0;
			}
#ifdef NT_VERBOSE
			print_round();
#endif
#ifdef NT_DEBUG
			std::cout << "debug: round " << round << ";sub " << sub_round << " complete" << std::endl;
#endif
		}
		// training complete
		time_delta = double(clock() - time_start) / CLOCKS_PER_SEC;
#ifdef NT_DEBUG
		std::cout << "debug: train() end" << std::endl;
#endif
		if (runtime_error) std::cout << "error: runtime error triggered" << std::endl;
		else
		{
			print_end();
			//std::cout << "----------------------------------" << std::endl;
			debug_check_network(best_network, "best", true);
		}
	}

}

#endif
