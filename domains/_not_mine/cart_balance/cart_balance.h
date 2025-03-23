// Cart Balance by Sierra
// inverted pendulum on a cart
// main objective to keep theta around 90*

#ifndef _CART_BALANCE_CODE_
#define _CART_BALANCE_CODE_

//#define CB_DEBUG
//#define CB_VERBOSE

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>


namespace CB {  // Cart Balance

	// global constants
	const double g = 9.81; 		// gravity - m/s^2
	const double dt = 0.1; 		// time step - s

	struct CartState {
		const double mass;		// kg
		const double track;		// length - m
		double x; 				// position - m
		double x_dot; 			// velocity - m/s
		double x_dd; 			// acceleration - m/s^2

		CartState(double in_mass, double in_len) : mass(in_mass), track(in_len), x(0.0), x_dot(0.0), x_dd(0.0) {}
	};

	// in regard to the end of the pendulum
	struct PendState {
		const double mass; 		// kg
		const double length;	// m
		double theta; 			// position - rad
		double theta_dot; 		// velocity - rad/s
		double theta_dd; 		// acceleration - rad/s^2

		PendState(double in_mass, double in_len) : mass(in_mass), length(in_len), theta(0.0), theta_dot(0.0), theta_dd(0.0) {}
	};


	class Pendulum {
	private:
		std::vector <double> torq_history;
		std::vector <double> force_history;
		std::vector <double> fitness_history;
		// all states from first to last
		std::vector <PendState> pend;
		std::vector <CartState> cart;
		// maximum action
		double max_torq;
		double max_force;
		// applied action to pendulum
		double torq;
		double force;
		// states
		double last_fitness;
		bool below_horizontal;

		double determine_reward(); 		// determine fitness

		void cycle(); 					// calculates next state given previous and an action

	public:
		Pendulum(unsigned int, double, double, double, double, double, double, double);
		bool return_below_horizontal();

		// functions
		std::vector <double> give_state(); 		// return last state

		std::vector <double> get_action(std::vector <double>); 	// get action then cycle

		double delta_theta(double); 	// return delta theta from PI/2 to input
		void export_all_states(); 		// log all state history to file
	};

	Pendulum::Pendulum(	
		unsigned int in_rounds = 100,	// rounds of simulation [dt]
		double in_theta = 89,			// inital theta [degrees]
		double in_mass_p = 5,			// mass of pendulum [kg]
		double in_length = 1,			// length of pendulum [m]
		double in_max_torq = 10,		// max torq [N*m]
		double in_max_force = 100,		// max force [N]
		double in_mass_c = 10,			// mass of cart [kg]
		double in_track_length = 10		// length of track [m]
		)
	{
#ifdef CB_DEBUG
		std::cout << "debug: Pendulum::Pendulum()" << std::endl;
#endif
		PendState initial_pend = {in_mass_p, in_length};
		CartState initial_cart = {in_mass_c, in_track_length};
		torq_history.reserve(in_rounds);
		fitness_history.reserve(in_rounds);
		pend.reserve(in_rounds);
		max_torq = in_max_torq;
		max_force = in_max_force;

		// do not modify - setup inital state
		below_horizontal = false;
		// convert the inital theta to radians
		initial_pend.theta = in_theta * M_PI / 180;
		initial_pend.theta_dot = 0; 	// rad/s
		initial_pend.theta_dd = 0;  	// rad/s^2
		//
		initial_cart.x = 0; 			// m
		initial_cart.x_dot = 0; 		// m/s
		initial_cart.x_dd = 0; 			// m/s^2
		//
		torq = 0; // N*m
		force = 0; // N
		pend.push_back(initial_pend);
		cart.push_back(initial_cart);
		torq_history.push_back(torq);
		force_history.push_back(force);
		fitness_history.push_back(determine_reward());
	}

	// calculates the next state
	// given a previously set action with a timestep of 'dt'
	// note: aciton must have been set - use 'get_action'
	void Pendulum::cycle() {
#ifdef CB_DEBUG
		std::cout << "debug: Pendulum::cycle()" << std::endl;
#endif
		// Create new state objects with the last state
		PendState pendState = {pend.at(pend.size() - 1).mass, pend.at(pend.size() - 1).length};
		CartState cartState = {cart.at(cart.size() - 1).mass, cart.at(cart.size() - 1).track};

		// Current state (from last entry)
		double theta = pend.at(pend.size() - 1).theta;
		double theta_dot = pend.at(pend.size() - 1).theta_dot;
		double x = cart.at(cart.size() - 1).x;
		double x_dot = cart.at(cart.size() - 1).x_dot;

		double mass_p = pendState.mass;
		double length = pendState.length;
		double mass_c = cartState.mass;
	
		// Compute accelerations (coupled equations)
		double x_dd = 0.0;      // Cart acceleration
		double theta_dd = 0.0;  // Pendulum angular acceleration
	
		// Step 1: Approximate x_dd assuming theta_dd contribution is small initially
		x_dd = (force - mass_p * length * theta_dot * theta_dot * cos(theta)) / (mass_c + mass_p);
	
		// Step 2: Compute theta_dd with the cart's acceleration
		theta_dd = (-g * cos(theta) + torq / (mass_p * length) - x_dd * sin(theta)) / length;
	
		// Step 3: Refine x_dd with theta_dd (optional, for better coupling)
		x_dd = (force - mass_p * length * (theta_dd * sin(theta) + theta_dot * theta_dot * cos(theta))) / (mass_c + mass_p);
	
		// Update cart state
		cartState.x_dd = x_dd;
		cartState.x_dot = cart.at(cart.size() - 1).x_dot + cartState.x_dd * dt;
		cartState.x = cart.at(cart.size() - 1).x + cartState.x_dot * dt;
	
		// Update pendulum state
		pendState.theta_dd = theta_dd;
		pendState.theta_dot = pend.at(pend.size() - 1).theta_dot + pendState.theta_dd * dt;
		pendState.theta = pend.at(pend.size() - 1).theta + pendState.theta_dot * dt;

		// keep theta between 0 and 2*PI
		if (pendState.theta < 0.0) pendState.theta = pendState.theta + 2*M_PI;
		if (pendState.theta > 2*M_PI) pendState.theta = pendState.theta - 2*M_PI;

#ifdef CB_CONSULE
		std::cout << pendState.theta << "," << pendState.theta_dot << "," \
		<< pendState.theta_dd << "," << cos(pendState.theta) << "," << sin(pendState.theta) \
		<< std::endl;
#endif
		pend.push_back(pendState); 				// save new pend state
		cart.push_back(cartState); 				// save new cart state
		torq_history.push_back(torq); 			// save torq for log
		force_history.push_back(force); 		// save force for log
	}

	// determine reward based on current state
	// max range for each fitness is 0 to 1 times its set weight
	double Pendulum::determine_reward() {
#ifdef CB_DEBUG
		std::cout << "debug: Pendulum::determine_reward()" << std::endl;
#endif
		// fitness weights
		// ---------------------
		double tp_weight = 100.0;  			// theta
		double tv_weight = 1000.0;  		// angular velocity
		double bh_weight = 10.0;  			// below horizontal axis
		double tu_weight = 0.0;  			// torq used penalty
		double wd_weight = 0.0;				// wrong direction
		double pi_weight = 0.0;				// integral of position error
		// ---------------------

		// theta position
		double fitness_tp = abs((delta_theta(pend.at(pend.size()-1).theta))/2*M_PI*tp_weight);

		// theta velocity
		double fitness_tv = abs((pend.at(pend.size()-1).theta_dot)*tv_weight);
		if (fitness_tv > 10) fitness_tv = 10;
		fitness_tv = fitness_tv/10;

		// below horizontal axis
		double fitness_bh = 0.0;
		if (pend.at(pend.size()-1).theta > M_PI) {
			fitness_bh = bh_weight;
			below_horizontal = true;
		} else {
			below_horizontal = false;
		}

		// torq used
		double fitness_tu = tu_weight * torq;

		// wrong direction
		double fitness_wd = 0.0;
		if (pend.size() >= 2) { // Safety check to avoid out-of-bounds access
			double prev_theta = pend.at(pend.size() - 2).theta;
			double curr_theta = pend.at(pend.size() - 1).theta;
			double target = M_PI / 2; // Target angle (90°)

			// Penalize if moving further away from target
			double prev_delta = std::abs(prev_theta - target);
			double curr_delta = std::abs(curr_theta - target);

			if (curr_delta > prev_delta) { // Moving away from M_PI/2
				fitness_wd = wd_weight * (curr_delta - prev_delta) / (M_PI); // Normalize by max reasonable change
			}
		}

		// integral of position error
		double fitness_pi = 0.0;
		for (std::size_t i=0; i<pend.size(); ++i) {
			fitness_pi += abs((delta_theta(pend.at(i).theta))/2*M_PI*pi_weight);
		}
		fitness_pi = fitness_pi / pend.size();

		// total fitness
		double total_fitness;
		total_fitness = fitness_tp + fitness_tv + fitness_bh + fitness_tu + fitness_wd + fitness_pi;
		return total_fitness;
	}


	// --------------------------------------
	// return current state
	// - theta, theta_dot
	std::vector <double> Pendulum::give_state() {
#ifdef CB_DEBUG
		std::cout << "debug: Pendulum::give_state()" << std::endl;
#endif
		std::vector <double> temp_state;
		temp_state.push_back(pend.at(pend.size()-1).theta);
		temp_state.push_back(pend.at(pend.size()-1).theta_dot);
		return temp_state;
	}

	// get action from controller
	// vector with torq in first element
	std::vector <double> Pendulum::get_action(std::vector <double> in_action) {
#ifdef CB_DEBUG
		std::cout << "debug: Pendulum::get_action()" << std::endl;
#endif
		torq = in_action.at(0);
		force = in_action.at(1);

		// max torq
		if (torq > max_torq) {
			torq = max_torq;
		}
		// min torq
		if (torq < -max_torq) {
			torq = -max_torq;
		}
		// max force
		if (force > max_force) {
			force = max_force;
		}
		// min force
		if (force < -max_force) {
			force = -max_force;
		}

		// ===	---
		cycle();
		// === ----

		// update fitness
		double fitness = determine_reward();
		fitness_history.push_back(fitness);		// fitness history

		// return reward
		std::vector <double> reward;
		reward.push_back(fitness);				// return [fitness]
		return reward;
	}

	// --------------------------------------

	// return delta theta from PI/2 to input in radians
	// untested
	double Pendulum::delta_theta(double in) {
#ifdef CB_DEBUG
		//std::cout << "debug: Pendulum::delta_theta()" << std::endl;
#endif
		double t = 0.0;
		if (in > M_PI/2 && in <= 3/2*M_PI) {
			// between: PI/2 and 3/2*PI
			t = in - M_PI/2;
		} else {
			if (in <= M_PI/2 && in >= 0) {
				// between: 0 and PI/2
				t = M_PI/2 - in;
			} else {
				// between: 3/2*PI and 2*PI
				t = (2*M_PI - in + M_PI/2); // 2*PI - in + PI/2
			}
		}

		// Ensure t is the shortest distance (in case of wrapping)
		if (t > M_PI) {
			t = 2 * M_PI - t;
		}

		return t;

	}

	// export all state history to file "pend_state_log.csv" for review
	void Pendulum::export_all_states() {
#ifdef CB_DEBUG
		std::cout << "debug: Pendulum::export_all_states()" << std::endl;
#endif
		std::ofstream fout;
		fout.open("/home/boss/data/working/personal/ProjectZero/domains/_not_mine/cart_balance/outputs/pend_state_log.csv", std::ofstream::out | std::ofstream::trunc);
		if (!fout.is_open()) {
			std::cerr << "ERROR: could not open file for writing" << std::endl;
			return;
		}
		fout << "torq, px, py, theta, theta_dot, theta_dd, fitness, delta_theta, force, cx, x_dot, x_dd" << "\n";
		for (std::size_t i=0; i<pend.size(); ++i) {
			fout << torq_history.at(i) << ", " << cos(pend.at(i).theta) << \
				", " << sin(pend.at(i).theta) << ", " << pend.at(i).theta*180/M_PI << \
				", " << pend.at(i).theta_dot << ", " << pend.at(i).theta_dd << \
				", " << fitness_history.at(i) << ", " << delta_theta(pend.at(i).theta)*180/M_PI << \
				", " << force_history.at(i) << ", " << cart.at(i).x << ", " << cart.at(i).x_dot << \
				", " << cart.at(i).x_dd << "\n";
		}	
		fout.close();
	}

	bool Pendulum::return_below_horizontal() {
		return below_horizontal;
	}

}

#endif
