
// Run Network Trainer for Neural Network : Domain Gridspace

#include <iostream>

#include "network_trainer.h"

//#define NT_DEBUG
//#define NT_VERBOSE
//#define NT_TEST

int main()
{
	Trainer::Trainer t;

	// - settings start -
	//t.test_count = 100;
	//t.round_max = 100;
	//t.population_size = 100;
	//t.input_layer_size = 3;
	//t.hidden_layer_size = 4;
	//t.output_layer_size = 1;
	//t.mutate_mod = 0.1;
	//t.mutate_chance = 0.4;
	// - settings end -

	t.train();

	return 0;
}
