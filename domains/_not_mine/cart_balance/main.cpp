#include "cart_balance.h"


int main()
{

	int rounds = 2000;

	// variables
	CB::Pendulum pend(rounds, 90);

	for (std::size_t i=0; i<rounds; ++i) {
		// calculate all of the states of the pendulum along the curve
		//pend.cycle();
		std::vector <double> in_action = {0.1*i};
		pend.get_action(in_action);
	}

	pend.export_all_states();

	// initialize cart weight
	//Cart cart; // Object call cart of type cart
	//cart.x = 0;
	
	return 0;
}
