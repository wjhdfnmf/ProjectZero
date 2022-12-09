
/*

Test Code

Copyright 2016 William Gregory.	 All rights reserved.

*/

#include <iostream>

#include "../neural_network.hpp"

#define PI 3.14159

// agent
void neural_network() {
	std::cout << "neural_network" << std::endl;
	std::vector <unsigned int> nc = {2,4,2};
	double mutate_mod = 1.0;
	double mutate_chance = 0.5;
	//
	Network::Network n;
	n.setup(nc, mutate_mod, mutate_chance);
	std::cout << "neural_network - done" << std::endl;
}

int main () {

	std::cout << "\nTesting:" << std::endl;
	std::cout << "------------" << std::endl;

	neural_network();

	std::cout << "-------------------" << std::endl;
	std::cout << "training complete" << std::endl;
	std::cout << "" << std::endl;

	return 0;
}
