#include "cart_balance.h"


int main()
{
	// SETTINGS
	unsigned int rounds = 100;	// rounds of simulation [dt]
	double theta = 260;			// inital theta [degrees]
	double mass_p = 5;			// mass of pendulum [kg]
	double length = 1;			// length of pendulum [m]
	double max_torq = 10;		// max torq [N*m]
	double max_force = 100;		// max force [N]
	double mass_c = 10;			// mass of cart [kg]
	double track_length = 10;	// length of track [m]

	// variables
	CB::Pendulum pend(rounds, theta, mass_p, length, max_torq, max_force, mass_c, track_length);

	for (std::size_t i=0; i<rounds; ++i) {
		// input: torq, force
		double torq = 0;
		double force = 0;
		//force = -1*max_force*i;
		std::vector <double> in_action = {torq, force};
		pend.get_action(in_action);
	}

	pend.export_all_states();

	// initialize cart weight
	//Cart cart; // Object call cart of type cart
	//cart.x = 0;
	
	return 0;
}
