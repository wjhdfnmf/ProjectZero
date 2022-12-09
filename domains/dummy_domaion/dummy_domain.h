/*
* Dummy Domain
*
* Version 2.1
*
* Creaded by William on July 8th, 2016 to Present
* Copyright (C) 2016 William.  All rights reserved
*
*/
// incorperate sub_rounds


#ifndef _DUMMY_DOMAIN_
#define _DUMMY_DOMAIN_


class DummyDomain {


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
	double DummyDomain::train_function(double& val)
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
	double DummyDomain::generate_value()
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
	double DummyDomain::scale_value(double in_val, bool in_op, bool in_scale)
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
	void DummyDomain::debug_check_network(Network::Network in_pop, std::string in_name, bool print_weights)
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
	void DummyDomain::generate_round_values()
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
	double DummyDomain::fitness_function(double in_ac_out, double in_ex_out)
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
	double DummyDomain::check_network_error(std::vector <double>& in_real_out, std::vector <double>& in_ex_out)
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
	void DummyDomain::check_population_error(bool is_sub_round)
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
};

#endif
