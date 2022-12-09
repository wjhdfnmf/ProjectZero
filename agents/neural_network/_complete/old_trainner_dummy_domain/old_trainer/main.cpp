
// Run Network Trainer for Neural Network

#include <iostream>

#include "network_trainer.h"

//#define NT_DEBUG
//#define NT_VERBOSE
//#define NT_TEST

int main()
{
	Trainer::Trainer t;

	// - settings start -

	// (0:x^2+1, 1:step, 2:exp(x), 3:sin(x), 4:cos(x), 5:const, 6:sqrt(x)+1, 7:-x)
	t.train_func_type = 0;

	t.population_size = 100;
	t.round_max = 10000;
	t.sub_round_max = 60;
	t.input_min = 0;
	t.input_max = 2;
	t.hidden_layer_size = 4;
	t.mutate_mod = 0.1;
	t.mutate_chance = 0.3;  // X% chance for weight to mutate
	// - settings end -

	t.train();

	return 0;
}
