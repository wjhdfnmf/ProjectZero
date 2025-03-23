
/*

Neural Network

Version 3.0

Created by William Gregory on October 1st, 2016 to Present.
Copyright (c) 2016 William Gregory.  All rights reserved.

*/

#ifndef _NEURAL_NETWORK_
#define _NEURAL_NETWORK_

//#define NN_DEBUG
//#define NN_VERBOSE
//#define NN_TEST

#include <iostream>  // cout, endl
#include <vector>  // .. vectors
#include <stdlib.h> // srand, rand, exit
#include <cstdlib> // RAND_MAX
#include <time.h> // time for srand and delta times
#include <math.h> // exp - sigmoid
#include <string>  // messages
#include <ctime>  // clock
#include <fstream>  // file IO

#define LYRAND (double)rand()/RAND_MAX

/*

Ideas:
	- create (error, debug, info)/message class or remove functions
	- create testing class
	- add scale function for input and output to meet network requirements given min and max inputs

ToDo:
	- add weight import/export to file (save best network after training)

Info:
	- settings
		- debug: provide function start end info to find where error occurs.
		- verbose: provide info of internal network states while running

*/

namespace Network {

	class Node
	{
	private:
		/// variables
		bool runtime_error;
		// weight for each connection
		std::vector <double> weights;
		// generated weight allowed min and max values
		double weight_gen_min;
		double weight_gen_max;
		// mutate mod and chance
		double mutate_mod;
		double mutate_chance;
		//std::vector <unsigned int> connections;  // for selectively connected
		bool bias;  // true for bias node (output 1 before weight)
		unsigned int layer_type;  // 0:input, 1:hidden, 2:output (redundant with layer_num)
		unsigned int layer_num;  // layer 1 for input, N for output, else hidden

		/// functions

		// call to report an error
		void error_call(std::string in) {
			std::cout << "ERROR:Node: " << in << std::endl;
			runtime_error = true;
			exit(0);
		}
		// 
		void debug_call(std::string in) {
			std::cout << "DEBUG:Node: " << in << std::endl;
		}

		// generate weights randomly at start (sub)
		// input: number of weights to generate
		// output: populates weight vector
		void generate_weights(unsigned int& in_size) {
#ifdef NN_DEBUG
			debug_call("generate_weights start");
#endif
			weights.clear();
			// not output node - randomly generate weights
			for (std::size_t i = 0; i<in_size; ++i) {
				// generate weights within weight_gen_min:weight_gen_max
				if(LYRAND > 0.5) weights.push_back(LYRAND);
				else weights.push_back(-LYRAND);
			}
		}

		// sum inputs (sub)
		// input: all inputs to node
		// output: summation of inputs
		double sum_inputs(std::vector <double>& in) {
			double t_avg = 0.0;
			for (std::size_t i=0; i<in.size(); ++i) {
				t_avg += in.at(i);
			}
			return t_avg;
			//return t_avg / (double) in.size();
		}

		// activation function (sub)
		// input: scaler input
		// output: input after activation function
		double activation_function(double& in) {
			//return 1/(1+exp(-in));  // sigmoid
			return tanh(in);  // hyperbolic tangent
		}

		// generate outputs and apply weights (sub)
		// input: scaler network input after activation function
		// output: vector, scaler times connection weight
		std::vector <double> generate_outputs(double& in) {
#ifdef NN_DEBUG
			debug_call("generate_outputs start");
#endif
			std::vector <double> t_out;
			switch (layer_type) {
				case 0:
				case 1:
					// output = out * weight(i)
					for (std::size_t i=0; i<weights.size(); ++i) {
						t_out.push_back(in*weights.at(i));
					}
					break;
				default: // output layer - no weights
					t_out.push_back(in);
			}
			return t_out;
		}

	public:
		/// functions

		// setup (main)
		// input: node settings
		void setup(	unsigned int& in_w_count,  // in weight count
					bool& in_bias,
					double& in_mod,
					double& in_chance,
					unsigned int& in_type,
					unsigned int& in_num ) {
#ifdef NN_DEBUG
			debug_call("setup start");
#endif
			// settings
			weight_gen_min = (-1);
			weight_gen_max = ( 1);
			// settings end
			// set input options
			runtime_error = false;
			bias = in_bias;
			mutate_mod = in_mod;
			mutate_chance = in_chance;
			layer_type = in_type;
			layer_num = in_num;
			// generate weights except for output layer
			if (layer_type != 2) generate_weights(in_w_count);
#ifdef NN_VERBOSE
			std::string ti = "weight count: " + std::to_string(in_w_count) +
				"; bias: " + std::to_string(in_bias) + "; mod: " + std::to_string(in_mod) +
				"; chance: " + std::to_string(in_chance) + "; type: " + std::to_string(in_type) +
				"; num: " + std::to_string(in_num);
			debug_call(ti);
#endif
		}

		// cycle (main)
		// input: vector of inputs to node
		// output: outputs, one for each connection, with weights applied
		// note: how to deal with cycle of bias node
		std::vector <double> cycle(std::vector <double>& in) {
#ifdef NN_DEBUG
			debug_call("cycle start");
#endif
			// test inputs to node (info not available)
			// cycle
			double t_out;  // node output scaler
			if (!bias) {
				switch (layer_type) {
					case 0:
						t_out = in.at(0); // input layer
						break;
					default:
						double t_summed = sum_inputs(in);
						t_out = activation_function(t_summed);  // not input layer
				}
			}
			else t_out = 1.0;  // bias node (output 1.0)
			std::vector <double> t_final_out = generate_outputs(t_out);
			return t_final_out;
		}

		// mutate (main)
		// mutates the node using mod and chance
		void mutate() {
			for (std::size_t i=0; i<weights.size(); ++i) {
				if (LYRAND < mutate_chance) {
					// weight = weight + LYRAND*mod - LYRAND*mod
					weights.at(i) = weights.at(i) + LYRAND*mutate_mod \
									- LYRAND*mutate_mod;
				}
			}
		}

		// export weights (side)
		std::vector <double> export_weights() {
			return weights;
		}

		// import weights (side)
		void import_weights(std::vector <double>& in) {
			weights = in;
		}
	};


	class Layer
	{
	private:
		/// variables
		// vector of all nodes in this layer
		std::vector <Node> nodes;
		//
		unsigned int layer_type;
		unsigned int layer_num;
		//
		bool runtime_error;
		/// functions
		// call to report an error
		void error_call(std::string in) {
			std::cout << "ERROR:Layer: " << in << std::endl;
			runtime_error = true;
			exit(0);
		}
		// 
		void debug_call(std::string in) {
			std::cout << "DEBUG:Layer: " << in << std::endl;
		}

	public:
		/// functions

		// setup layer (main)
		void setup( unsigned int& in_n_count,  // nodes in layer
					unsigned int& in_next_n_count,  // nodes in next layer
					double& in_mod,
					double& in_chance,
					unsigned int& in_type,  // layer type
					unsigned int& in_num ) {
			// setup
#ifdef NN_DEBUG
			debug_call("setup start");
#endif
			runtime_error = false;
			nodes.clear();
			layer_type = in_type;
			layer_num = in_num;
			// fill node vector with in_n_count nodes
			for (std::size_t i=0; i<in_n_count; ++i) {
				// note:
				// - weight size = nodes in next layer
				// - in_n_count includes bias node
				// - in_next_n_count is 0 when output layer

				// last node bias if not an output layer
				bool t_bias = true;
				if (i == (in_n_count-1) && in_type != 2) {  // bias
					Node n;
					n.setup(in_next_n_count, t_bias, in_mod, in_chance, layer_type, layer_num);
					nodes.push_back(n);
				} else {  // normal
					t_bias = false;
					Node n;
					n.setup(in_next_n_count, t_bias, in_mod, in_chance, layer_type, layer_num);
					nodes.push_back(n);
				}
			}
		}

		// cycle layer (main)
		// input: all inputs for nodes in layer
		// output: all outputs from all nodes in layer
		// format note: < <node1>, <node2>, ..., <nodeN> >
		std::vector <std::vector <double> > cycle(std::vector <std::vector <double> >& in) {
#ifdef NN_DEBUG
			debug_call("cycle start");
#endif
#ifdef NN_TEST
			if (in.at(0).size() != nodes.size()) {
				std::string tm = "cycle - input size not equal to nodes size.  nodes: " +
					std::to_string(nodes.size()) + "; inputs: " +
					std::to_string(in.size()) + ";";
				error_call(tm);
			}
#endif
			// cycle
			std::vector <std::vector <double> > t_in;  // formatted input vector
			// format input FIX
			for (std::size_t i=0; i<nodes.size(); ++i) {
				std::vector <double> tt;
				for (std::size_t j=0; j<in.size(); ++j) {
					tt.push_back(in.at(j).at(i));
				}
				t_in.push_back(tt);
			}
			// input and cycle
			std::vector <std::vector <double> > t_out;  // outputs from all nodes
			for (std::size_t i=0; i<t_in.size(); ++i) {
#ifdef NN_DEBUG
				std::string tm = "input node:" + std::to_string(i) + "; ";
				for (std::size_t d=0; d<t_in.at(i).size(); ++d) {
					tm = tm + std::to_string(t_in.at(i).at(d)) + "; ";
				}
				debug_call(tm);
#endif
				t_out.push_back(nodes.at(i).cycle(t_in.at(i)));	
			}
			return t_out;
		}

		// mutate (main)
		// call all nodes mutate function
		void mutate() {
			for (std::size_t i=0; i<nodes.size(); ++i) {
				nodes.at(i).mutate();
			}
		}

		// return node count(info)
		unsigned int return_node_count() {
			return nodes.size();
		}

		// export weights (side)
		std::vector <std::vector <double> > export_weights() {
			std::vector <std::vector <double> > t;
			for (std::size_t i=0; i<nodes.size(); ++i) {
				t.push_back(nodes.at(i).export_weights());
			}
			return t;
		}

		// import weights (side)
		void import_weights(std::vector <std::vector <double> >& in) {
			if (in.size() != nodes.size()) error_call("import weights size mismatch");
			for (std::size_t i=0; i<in.size(); ++i) {
				nodes.at(i).import_weights(in.at(i));
			}
		}
	};


	class Network
	{
	private:
		/// variables
		std::vector <Layer> layers;
		bool runtime_error;
		/// functions
		// 
		void error_call(std::string in) {
			std::cout << "ERROR:Network: " << in << std::endl;
			runtime_error = true;
			exit(0);
		}
		// 
		void debug_call(std::string in) {
			std::cout << "DEBUG:Network: " << in << std::endl;
		}
		//
		void info_call(std::string in) {
			std::cout << "VERB:Network: " << in << std::endl;
		}
		//
		void print_intro() {
			std::cout << std::endl;
			std::cout << "Neural Network V3.0" << std::endl;
			std::cout << "WG2016" << std::endl;
			// npl, version, creator, other network config
			std::cout << "---------------------" << std::endl;
		}

		// add bias to npl count (sub) (multi)
		// input: desired npl layer given to setup
		// output: returned with added bias to proper layers
		std::vector <unsigned int> add_bias(std::vector <unsigned int>& in_npl) {
#ifdef NN_DEBUG
			debug_call("add bias start");
#endif
			std::vector <unsigned int> t_nc = in_npl;  // FIX
			// add bias node to all layers but the output
			for (std::size_t i = 0; i<(t_nc.size()-1); ++i) {
				t_nc.at(i) = (t_nc.at(i) + 1);
			}
			return t_nc;
		}

		// create network (sub) (single)
		// input: nodes per layer (excluding bias), mutate mod and chance
		void create_network(std::vector <unsigned int>& in_npl, double& in_mod, double& in_chance)
		{
#ifdef NN_DEBUG
			debug_call("create network start");
#endif
			npl = add_bias(in_npl);  // set node per layer with bias
			// create layers
			for (std::size_t i = 0; i < npl.size(); ++i) {
				// next node count
				unsigned int t_nnc;
				if (i == (npl.size()-1)) t_nnc = 0;  // output layer
				else t_nnc = npl.at(i+1);
				// layer type
				unsigned int t_type;
				if (i == 0) t_type = 0;  // input
				else if (i == (npl.size()-1)) t_type = 2;  // output
				else t_type = 1;  // hidden
				// debug
#ifdef NN_DEBUG
				std::string tm = "Creating layer: " + std::to_string(i) +
					"; node: " + std::to_string(npl.at(i)) +
					"; connections: " + std::to_string(t_nnc) + ";";
				debug_call(tm);
#endif
				// layer
				Layer l;
				unsigned int t_num = i+1;
				l.setup(npl.at(i), t_nnc, in_mod, in_chance, t_type, t_num);
				layers.push_back(l);
			}
#ifdef NN_DEBUG
			std::string tm = "creating network: ";
			for (std::size_t m = 0; m<npl.size(); ++m) {
				tm = tm + std::to_string(npl.at(m)) + " ";
			}
			debug_call(tm);
#endif
		}

		// format input (sub) (multi)
		// input: input vector to network
		// output: formatted for passing to input layer
		std::vector <std::vector <double> > format_input(std::vector <double>& in) {
#ifdef NN_DEBUG
			debug_call("format input start");
#endif
			std::vector <std::vector <double> > t_out;
			std::vector <double> t_in = in;  // FIX
			// bias node
			t_in.push_back(1.0);
			t_out.push_back(t_in);
			return t_out;
		}

		/*
		// scale input (sub) (multi)
		// input: input vector before formatting
		// output: input vector scalled
		std::vector <double> scale_input(std::vector <double> in) {
			std::vector <double> t_out;
			for (std::size_t i=0; i<in.size(); ++i) {
				t_out.push_out(in.at(i)/input_max);
			}
			return t_out;
		}

		// scale output (sub) (mulit)
		// needed??
		// input: output vector
		// output: output vector scalled
		std::vector <double> scale_output(std::vector <double> in) {
			std::vector <double> t_out;
			for (std::size_t i=0; i<in.size(); ++i) {
				t_out.push_back(in.at(i));
			}
		}
		*/

	public:
		/// variables
		unsigned int run_type;  // 0:single, 1:train
		unsigned int ID_value;
		std::vector <unsigned int> npl;  // including bias
		//double input_max;  // for scaling
		//double input_min;
		//double output_max;  // result of input to activation
		//double output_min;

		/// functions

		// constructor
		Network() {
			runtime_error = false;
			ID_value = 007;
			run_type = 0;
			//srand(time(0));
		}

		// setup (main) (single)
		void setup( std::vector <unsigned int>& in_npl,
					double& in_m,
					double& in_c ) {
#ifdef NN_DEBUG
			debug_call("setup start");
#endif
			create_network(in_npl, in_m, in_c);
			switch (run_type) {
				case 0:
					print_intro();
			}
		}

		// cycle network (main) (multi) (time)
		// input: input vector (one double per input node)
		// output: network output (one double per output node)
		std::vector <double> cycle(std::vector <double>& in) {
			std::string tm = ""; // test and verbose
#ifdef NN_DEBUG
			debug_call("cycle start");
#endif
			std::vector <std::vector <double> > inputs;
			std::vector <std::vector <double> > outputs;
			for (std::size_t i=0; i<layers.size(); ++i) {
				// input layer
				switch(i) {
					case 0:
						inputs = format_input(in);
						break;
					default:
						inputs = outputs;
				}
				if (inputs.at(0).size() != layers.at(i).return_node_count()) {
					tm = "cycle - input size mismatch to nodes in layer. layer: " +
						std::to_string(i) + "; input: " +
						std::to_string(inputs.size()) + "; nodes: " +
						std::to_string(layers.at(i).return_node_count()) + ";"; 
					error_call(tm);
				}
#ifdef NN_VERBOSE
				tm = "cycle layer: " + std::to_string(i);
				info_call(tm);
				tm = "inputs: ";
				for (std::size_t d=0; d<inputs.size(); ++d) {
					for (std::size_t e=0; e<inputs.at(d).size(); ++e) {
						tm = tm + std::to_string(inputs.at(d).at(e)) + "; ";
					}
					if (d != inputs.size()-1) tm = tm + "\t";
				}
				info_call(tm);
#endif
				// cycle
				outputs = layers.at(i).cycle(inputs);
#ifdef NN_VERBOSE
				tm = "outputs: ";
				for (std::size_t d=0; d<outputs.size(); ++d) {
					for (std::size_t e=0; e<outputs.at(d).size(); ++e) {
						tm = tm + std::to_string(outputs.at(d).at(e)) + "; ";
					}
					if (d != outputs.size()-1) tm = tm + "\t";
				}
				info_call(tm);
#endif
#ifdef NN_DEBUG
				debug_call("cycle layer complete");
#endif
			}
			// prepare output
			std::vector <double> outputs_simple;
			// simplify output vector
			for (std::size_t i=0; i<outputs.size(); ++i) {
				// output nodes will only have a value at the first index
				outputs_simple.push_back(outputs.at(i).at(0));
			}
#ifdef NN_DEBUG
			debug_call("cycle end");
#endif
			return outputs_simple;
		}

		// mutate (main) (semi-multi)(time)
		void mutate() {
#ifdef NN_DEBUG
			debug_call("mutate start");
#endif
			for (std::size_t i=0; i<layers.size(); ++i) {
				layers.at(i).mutate();
			}
		}

		// export weights (side)
		std::vector <std::vector <std::vector <double> > > export_weights()
		{
#ifdef NN_DEBUG
			debug_call("export weights start");
#endif
			std::vector <std::vector <std::vector <double> > > t;
			for (std::size_t i=0; i<layers.size(); ++i) {
				t.push_back(layers.at(i).export_weights());
			}
			return t;
		}

		// import weights (side)
		void import_weights(std::vector <std::vector <std::vector <double> > > in)
		{
#ifdef NN_DEBUG
			debug_call("import weights start");
#endif
			if (in.size() != layers.size()) error_call("import_weights size mismatch");
			for (std::size_t i=0; i<in.size(); ++i) {
				layers.at(i).import_weights(in.at(i));
			}
		}
	};
}

#endif
