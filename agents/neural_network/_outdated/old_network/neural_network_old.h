/*
* Neural Network
*
* Version 2.1
*
* Created by William Gregory on March 1st, 2016
* Copyright © 2016 William.  All rights reserved.
*
*/

#ifndef _Neural_Network_
#define _Neural_Network_

//#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iomanip>
#include <time.h>

#define LYRAND (double)rand()/RAND_MAX


// 
namespace Network {

	//----------------------------------------
	class node
	{
		friend class layer;
		friend class network;

		double weight_min;
		double weight_max;

	public:
		node();
		std::vector <double> inputs;
		std::vector <double> weights;

		void mutate(double, double);
		//void crossover(std::vector <double>, std::vector <double>);
	};

	node::node()
	{
		weight_min = -1.0;
		weight_max = 1.0;
	}

	// note: weights not constrained to min/max
	void node::mutate(double m, double c)
	{
		for (std::size_t a = 0; a<weights.size(); a++) {

			// MUTATE CODE START
			// ------------------

			// negate
			//if (LYRAND < 0.2) weights.at(a) = weights.at(a) * -1;

			// weights + LYRAND*mod - LYRAND*mod
			// mutate chance
			if (LYRAND < c)
				weights.at(a) = weights.at(a) + (LYRAND * m) - (LYRAND * m);

			// check range
			/*
			while (weights.at(a) > weight_max) {
				weights.at(a) = weights.at(a) - (LYRAND * m);
			}
			while (weights.at(a) < weight_min) {
				weights.at(a) = weights.at(a) + (LYRAND * m);
			}
			*/

			// ------------------
			// MUTATE CODE END
		}
	}
	/*
	// swap with parent weights
	void node::crossover(std::vector <double> p1_weights, std::vector <double> p2_weights)
	{
		std::vector <double> new_weights;
		// pick one net randomly to replace equivalent value in new weight
		for (std::size_t i = 0; i<p1_weights.size(); ++i)
		{
			if (LYRAND >= 0.5) {
				new_weights.push_back(p1_weights.at(i));
			}
			else {
				new_weights.push_back(p2_weights.at(i));
			}
		}
	}
	*/
	//----------------------------------------
	class layer
	{
		friend class node;
		friend class network;

	public:
		std::vector <node> nodes;
	};

	//----------------------------------------
	class network
	{
		friend class node;
		friend class layer;

		bool debug;
		int cycle;

		std::vector <double> generate_weights(unsigned int);
		void create_network();

	public:
		unsigned int ID_value;
		unsigned int activation_type;

		std::vector <double> inputs;
		std::vector <double> outputs;
		std::vector <layer> layers;
		std::vector <unsigned int> node_count;

		network();
		void setup(std::vector <unsigned int>);

		void print_info();
		void weights_info(std::string);

		double activation_function(double);

		std::vector <std::vector <std::vector <double> > > parse_input_file(std::string);
		std::vector <std::vector <std::vector <double> > > export_weights();
		void import_weights(std::vector <std::vector <std::vector <double> > >);

		void export_weights_to_file(std::string);
		void import_weights_from_file(std::string);

		void set_values(std::vector <double>);
		void mutate(double, double);
		void cycle_network();
	};

	network::network()
	{
		
	}

	void network::setup(std::vector <unsigned int> nc)
	{
		debug = false; // setting for debug
		activation_type = 1;  // 0:sigmoid, 1:tanh

		outputs.clear();
		inputs.clear();
		layers.clear();
		node_count.clear();

		node_count = nc;
		cycle = 1;
		create_network();
	}

	void network::print_info()
	{
		std::cout << std::endl;
		std::cout << "------------------------------" << std::endl;
		std::cout << std::endl;
		std::cout << "Neural Network" << std::endl;
		std::cout << "version " << 2.1 << std::endl;
		std::cout << std::endl;
		std::cout << "Input nodes: " << node_count.at(0) << std::endl;
		std::cout << "Output nodes: " << node_count.at(node_count.size() - 1) << std::endl;
		std::cout << "Hidden layers: " << node_count.size() - 2 << std::endl;
		std::cout << std::endl;
		std::cout << "------------------------------" << std::endl;
		std::cout << std::endl;
	}

	std::vector <std::vector <std::vector <double> > > network::parse_input_file(std::string file_name)
	{
		if (file_name == "") file_name = "weights.txt";
		std::ifstream file;
		file.open(file_name, std::ios::in);
		std::string line;
		std::vector <std::vector <std::vector <double> > > temp;
		std::vector <std::vector <double> > t2;
		int l = 0;
		int n = 0;
		while (getline(file, line)) {
			int pos = 0;
			std::string token;
			if (line.empty()) {
				l++;
				temp.push_back(t2);
				t2.clear();
			}
			else {
				std::vector <double> t1;
				while ((pos = line.find(",")) != std::string::npos) {
					token = line.substr(0, pos);
					line.erase(0, pos + 1);
					t1.push_back(stod(token));
				}
				t2.push_back(t1);
				n++;
			}
		}
		temp.push_back(t2);
		file.close();
		return temp;
	}

	void network::weights_info(std::string file_name)
	{
		std::cout << std::endl;
		std::cout << "weight file info" << std::endl;
		std::vector <std::vector <std::vector <double> > > temp = parse_input_file(file_name);
		for (std::size_t l = 0; l<temp.size(); l++) {
			std::cout << "nodes: " << temp.at(l).size() << std::endl;
		}
	}

	std::vector <std::vector <std::vector <double> > > network::export_weights()
	{
		std::vector <std::vector <std::vector <double> > > temp;
		for (std::size_t l = 0; l<layers.size() - 1; l++) { // no weights on output
			std::vector <std::vector <double> > t1;
			for (std::size_t n = 0; n<layers.at(l).nodes.size(); n++) {
				t1.push_back(layers.at(l).nodes.at(n).weights);
			} temp.push_back(t1);
		}
		return temp;
	}


	void network::import_weights(std::vector <std::vector <std::vector <double> > > given_weights)
	{
		for (std::size_t l = 0; l<layers.size() - 1; l++) {
			for (std::size_t n = 0; n<layers.at(l).nodes.size(); n++) {
				layers.at(l).nodes.at(n).weights = given_weights.at(l).at(n);
			}
		}
	}

	void network::export_weights_to_file(std::string file_name)
	{
		std::vector <std::vector <std::vector <double> > > temp = export_weights();
		std::ofstream file;
		file.open(file_name, std::ios::out | std::ios::trunc);
		for (std::size_t l = 0; l<temp.size(); l++) {
			for (std::size_t n = 0; n<temp.at(l).size(); n++) {
				for (std::size_t w = 0; w<temp.at(l).at(n).size(); w++) {
					file << std::setprecision(15) << temp.at(l).at(n).at(w);
					file << ',';
				}
				if (n != temp.at(l).size() - 1) file << '\n';
			}
			if (l != temp.size() - 1) file << "\n\n";
		}
		file.close();
	}

	void network::import_weights_from_file(std::string file_name)
	{
		std::vector <std::vector <std::vector <double> > > temp = parse_input_file(file_name);
		import_weights(temp);
	}

	// --------------------------------------------------- 

	double network::activation_function(double in_val)
	{
		// sigmoid
		if (activation_type == 0) return (double)(1 / (1 + exp(in_val*-1)));
		// hyperbolic tangent
		else if (activation_type == 1) return (double)tanh(in_val);
		// unknown
		else return 0.0;
	}

	std::vector <double> network::generate_weights(unsigned int nodes_next)
	{
		std::vector <double> t;
		for (std::size_t a = 0; a<nodes_next; a++) {
			if (LYRAND>0.5)	t.push_back(LYRAND);
			else t.push_back(-LYRAND);
		}
		return t;
	}

	void network::create_network()
	{
		// create layers
		for (std::size_t l = 0; l<node_count.size(); l++) {
			layer lr;
			for (std::size_t h = 0; h<node_count.at(l); h++) {
				node n;
				lr.nodes.push_back(n);
			} layers.push_back(lr);
		}

		// create weights
		for (std::size_t l = 0; l<layers.size() - 1; l++) {
			for (std::size_t n = 0; n<layers.at(l).nodes.size(); n++) {
				layers.at(l).nodes.at(n).weights = generate_weights(node_count.at(l + 1));
			}
		}
	}

	void network::set_values(std::vector <double> in)
	{
		inputs.clear();
		inputs = in;
	}

	void network::mutate(double m, double c)
	{
		for (std::size_t l = 0; l<layers.size() - 1; l++) {
			for (std::size_t n = 0; n<layers.at(l).nodes.size(); n++) {
				layers.at(l).nodes.at(n).mutate(m, c);
			}
		}
	}

	void network::cycle_network()
	{
		if (debug) std::cout << "cycle starting: " << cycle << std::endl;
		std::vector <std::vector <double> > temp_inputs;
		std::vector <std::vector <double> > temp_outputs;
		std::vector <double> temp_input;
		std::vector <double> temp_output;
		std::vector <std::vector <double> > temp_weights;
		// Iterate through each layer
		for (std::size_t l = 0; l<layers.size(); l++) {
			// - get inputs from current layer's nodes except for input layer which gets its inputs from 'inputs'
			if (l == 0) {
				temp_input = inputs;
			}
			else {
				for (std::size_t n = 0; n<layers.at(l).nodes.size(); n++) {
					temp_inputs.push_back(layers.at(l).nodes.at(n).inputs);
				}
			}
			if (debug) std::cout << "got intputs" << std::endl;

			if (l != layers.size() - 1) {
				for (std::size_t n = 0; n<layers.at(l).nodes.size(); n++) {
					temp_weights.push_back(layers.at(l).nodes.at(n).weights);
				}
			}
			if (debug) std::cout << "got weights" << std::endl;

			// - sum inputs except for input layer, its inputs are doubles, the rest are vectors of doubles
			if (l != 0) {
				for (std::size_t n = 0; n<temp_inputs.size(); n++) {
					temp_input.push_back(0.0);
					for (std::size_t i = 0; i<temp_inputs.at(n).size(); i++) {
						temp_input.at(n) += temp_inputs.at(n).at(i);
					}
				}
			}
			if (debug) std::cout << "inputs summed" << std::endl;

			// - hidden layer and output apply sigmoid
			if (l == 0) temp_output = temp_input;
			if (l != 0 && l != layers.size()) {
				for (std::size_t a = 0; a<temp_input.size(); a++) {
					temp_output.push_back(activation_function(temp_input.at(a)));
				}
			}
			if (debug) std::cout << "sigmoid applied" << std::endl;

			// - output layer returns outputs here, last layer
			if (l == layers.size() - 1) {
				outputs = temp_output;
			}

			// - create output vectors and apply weights
			if (l != layers.size() - 1) {
				for (std::size_t a = 0; a<node_count.at(l + 1); a++) {
					// push back every output from every node int the last layer
					std::vector <double> t;
					for (std::size_t b = 0; b<temp_output.size(); b++) {
						t.push_back(temp_output.at(b)*temp_weights.at(b).at(a));
					} temp_outputs.push_back(t);
				}
			}
			if (debug) std::cout << "output vectors with weights created" << std::endl;

			// - fill next layer nodes with inputs
			if (l != layers.size() - 1) {
				for (std::size_t a = 0; a<node_count.at(l + 1); a++) {
					layers.at(l + 1).nodes.at(a).inputs.clear();
					layers.at(l + 1).nodes.at(a).inputs = temp_outputs.at(a);
				}
			}
			if (debug) std::cout << "- layer done" << std::endl;

			temp_inputs.clear();
			temp_outputs.clear();
			temp_input.clear();
			temp_output.clear();
			temp_weights.clear();
		}
		if (debug) std::cout << "cycle complete\n" << std::endl;
		cycle++;
	}
}

#endif
