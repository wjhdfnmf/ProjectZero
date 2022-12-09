// Cart Balance by Sierra
// inverted pendulum on a cart
// main objective to keep theta around 90*

#ifndef _CART_BALANCE_CODE_
#define _CART_BALANCE_CODE_

//#define CB_CONSULE

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>


namespace CB {  // Cart Balance

	// global constants
	const double g = 9.81; // gravity - m/s^2
	const double dt = 0.01; // time step - s

	// cart structure
	struct Cart {
		// mass of the cart - constant
		const int mass_c;
		// constrained to x axis
		double x; // position
		double x_dot; // velocity
		double x_dd; // acceleration
	};

	// pendulum state structure
	struct PendState {
		// with regards to the end of the pendulum
		double theta; // current theta - rad
		double theta_dot; // current velocity of theta - rad/s
		double theta_dd; // current acceleration of theta - rad/s^2
	};


	class Pendulum {
	private:
		/// variables
		// all actions for log
		std::vector <double> torq_history;
		// all fitness' for log
		std::vector <double> fitness_history;
		// all states.  last element is most current
		std::vector <PendState> pend;
		// applied torq - action
		double torq;
		// last determined fitness
		double fitness;
		/// functions
		// determine fitness
		double determine_reward();
	public:
		Pendulum(unsigned int);
		// static variables
		double mass_p; // mass of pendulum
		double length; // length of the pendulum
		double theta_init; // inital theta
		// network training
		bool below_horizontal;
		bool return_below_horizontal();
		/// functions
		// return last state
		std::vector <double> give_state();
		// get action then cycle
		void get_action(std::vector <double>);
		// return last reward
		std::vector <double> give_reward();
		// calculates next state given previous and an action
		void cycle();
		//
		double delta_theta(double);
		// log all state history to file
		void export_all_states();
	};

	Pendulum::Pendulum(unsigned int in_rounds)
	{
		PendState initial;
		torq_history.reserve(in_rounds);
		fitness_history.reserve(in_rounds);
		pend.reserve(in_rounds);
		// - settings begin -
		mass_p = 5; // kg
		length = 1; // m
		theta_init = 89; // degrees
		// - settings end -

		// do not modify - setup inital state
		below_horizontal = false;
		initial.theta = theta_init * M_PI / 180;
		initial.theta_dot = 0; // rad/s
		initial.theta_dd = 0;  // rad/s^2
		torq = 0; // N*m
		pend.push_back(initial);
		torq_history.push_back(torq);
		fitness_history.push_back(determine_reward());
	}

	// calculates the next state given a previously set action with a timestep of 'dt'
	// note: aciton must have been set. use 'get_action'
	void Pendulum::cycle() {
		PendState nextState;
		// - calculations - 
		// torque to theta dd
		nextState.theta_dd = -g*cos(pend.at(pend.size() - 1).theta) / (mass_p*length) + torq;
		// theta_dd to theta_dot
		nextState.theta_dot = pend.at(pend.size() - 1).theta_dot + nextState.theta_dd*dt;
		// theta_dot to theta
		nextState.theta = pend.at(pend.size() - 1).theta + nextState.theta_dot*dt;
		// - end calculations -
		// keep theta between 0 and 2*PI
		if (nextState.theta < 0.0) nextState.theta = nextState.theta + 2*M_PI;
		if (nextState.theta > 2*M_PI) nextState.theta = nextState.theta - 2*M_PI;
#ifdef CB_CONSULE
		std::cout << nextState.theta << "," << nextState.theta_dot << "," \
		<< nextState.theta_dd << "," << cos(nextState.theta) << "," << sin(nextState.theta) \
		<< std::endl;
#endif
		pend.push_back(nextState); // save new state
		torq_history.push_back(torq); // save torq for log
		fitness = determine_reward(); // determine fitness
		fitness_history.push_back(fitness);
	}

	// determine reward based on current state
	// max range for each fitness is 0 to 1 times its set weight
	double Pendulum::determine_reward() {

		// fitness weights
		// ---------------------
		double tp_weight = 1000.0;  	// theta
		double tv_weight = 0.0;  		// angular velocity
		double ch_weight = 1000000.0;  	// below horizontal axis
		//double tu_weight = 0.0;  		// torq used penalty - not normalized
		//double wd_weight = 1.0;		// wrong direction
		// ---------------------

		// theta position
		double fitness_1 = abs((delta_theta(pend.at(pend.size()-1)))/2*M_PI*tp_weight);
		// theta velocity
		double fitness_2 = abs((pend.at(pend.size()-1).theta_dot)*tv_weight);
		if (fitness_2 > 10) fitness_2 = 10;
		fitness_2 = fitness_2/10;
		// below horizontal axis
		double fitness_ch = 0.0;
		if (pend.at(pend.size()-1).theta > M_PI) {
			fitness_ch = ch_weight;
			below_horizontal = true;
		}
		// effort used
		//double fitness_tu = tu_weight * torq;
		// wrong direction
		/*
		double fitness_wd = 0.0;
		if (pend.at(pend.size()-2).theta > M_PI/2 && pend.at(pend.size()-1).theta > pend.at(pend.size()-2))
			fitness_wd = wd_weight * (pend.at(pend.size()-1).theta - pend.at(pend.size()-2).theta) / 3/2*M_PI;
		else if (pend.at(pend.size()-2).theta < M_PI/2 && pend.at(pend.size()-1).theta < pend.at(pend.size()-2))
			fitness_wd = wd_weight * (pend.at(pend.size()-2).theta - pend.at(pend.size()-1).theta) / 1/2*M_PI;
		*/

		double total_fitness;
		total_fitness = fitness_1 + fitness_2 + fitness_ch; //+ fitness_tu;
		return total_fitness;
	}


	// --------------------------------------
	// return current state
	// - theta, theta_dot
	std::vector <double> Pendulum::give_state() {
		std::vector <double> temp_state;
		temp_state.push_back(pend.at(pend.size()-1).theta);
		temp_state.push_back(pend.at(pend.size()-1).theta_dot);
		return temp_state;
	}

	// get action from controller
	// vector with torq in first element
	void Pendulum::get_action(std::vector <double> in_action) {
		torq = in_action.at(0);
		cycle();
	}

	// return reward for current state
	std::vector <double> Pendulum::give_reward() {
		std::vector <double> total_fitness;
		total_fitness.push_back(fitness);

		return total_fitness;
	}
	// --------------------------------------

	// return delta theta from PI/2 to input
	// untested
	double Pendulum::delta_theta(double in) {
		double t = 0.0;
		if (in > M_PI/2 && in < 3/2*M_PI) {
			// between: PI/2 and 3/2*PI
			t = in - M_PI/2;
		} else {
			if (in < M_PI/2) {
				// between: 0 and PI/2
				t = M_PI/2 - in;
			} else {
				// between: 3/2*PI and 2*PI
				t = (5/2*M_PI - in); // 2*PI - in + PI/2
			}
		}
		return t;

	}

	// export all state history to file "pend_state_log.csv" for review
	void Pendulum::export_all_states() {
		std::ofstream fout;
		fout.open("pend_state_log.csv", std::ofstream::out | std::ofstream::trunc);
		fout << "torq, x, y, theta, theta_dot, theta_dd, fitness" << "\n";
		for (std::size_t i=0; i<pend.size(); ++i) {
			fout << torq_history.at(i) << ", " << cos(pend.at(i).theta) << \
				", " << sin(pend.at(i).theta) << ", " << pend.at(i).theta*180/M_PI << \
				", " << pend.at(i).theta_dot << ", " << pend.at(i).theta_dd << \
				", " << fitness_history.at(i) << "\n";
		}	
		fout.close();
	}

	bool return_below_horizontal() {
		return below_horizontal;
	}

}

#endif
