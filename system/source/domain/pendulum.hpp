// Cart Balance by Sierra
// inverted pendulum on a cart
// main objective to keep theta around 90*

// add random theta given range function

#ifndef _PENDULUM_
#define _PENDULUM_

// manage in system
//#define PD_DEBUG
//#define PD_CONSULE
//#define PD_MIDLOG

#ifndef gravity
#define gravity 9.81  // m/s^2
#endif

#ifndef PI
#define PI 3.14159265
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

namespace InvPend {	 // Inverted Pendulum

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
		// all states.	last element is most current
		std::vector <PendState> pend;
		// applied torq - action
		double torq;
		// last determined fitness
		double fitness;
		// message before debug
		std::string pre_debug;
		/// functions
		// determine fitness
		double determine_reward();
	public:
		Pendulum();
		// fitness settings
		double tp_weight;  // theta
		double tv_weight;  // angular velocity
		double ch_weight;  // below horizontal axis
		// static settings
		double mass_p; // mass of pendulum
		double length; // length of the pendulum
		double theta_init; // inital theta
		double dt;	// time step
		/// functions
		// set prendulum properties
		void set_pendulum_pos(double, double);
		// set fitness weights
		void set_weights(std::vector <double>);
		// setup and set rounds
		void setup(unsigned int, double);
        // calculates next state given previous and an action
		void cycle();
		// offset from PI/2
		double delta_theta(double);
		// log all state history to file
		void export_all_states();
        // -----------------------------
        // system functions
        // -----------------
        // return last state
		std::vector <double> give_state();
		// get action then cycle
		void get_action(std::vector <double>);
		// return last reward
		std::vector <double> give_fitness();
        // network training
		bool below_horizontal;
        // send data
		std::vector <double> update_out();
        // get data
		void update_in(std::vector <double>);
	};

	Pendulum::Pendulum()
	{
		pre_debug = "D.DP\t";
#ifdef PD_DEBUG
		std::cout << pre_debug << "Pendulum()" << std::endl;
#endif
		// fitness weights
		tp_weight = 1000.0;		// theta offset 90*
		tv_weight = 0.0;	// angular velocity
		ch_weight = 1000000.0;	// below horizontal
		// pendulum
		set_pendulum_pos(5.0, 1.0);
		dt = 0.01; // s
	}

	// input: test_count, theta_inital
	void Pendulum::setup(unsigned int in_rounds, double in_ti) {
#ifdef PD_DEBUG
		std::cout << pre_debug << "setup()" << std::endl;
#endif
		theta_init = in_ti;
		PendState initial;
		torq_history.reserve(in_rounds);
		fitness_history.reserve(in_rounds);
		pend.reserve(in_rounds);
		// do not modify - setup inital state
		below_horizontal = false;
		initial.theta = theta_init * PI / 180;
		initial.theta_dot = 0;	// rad/s
		initial.theta_dd = 0;  // rad/s^2
		torq = 0; // N*m
		pend.push_back(initial);
		torq_history.push_back(torq);
		fitness_history.push_back(determine_reward());
	}

	// determine reward based on current state
	// max range for each fitness is 0 to 1 times its set weight
	double Pendulum::determine_reward() {
#ifdef PD_DEBUG
		std::cout << pre_debug << "determine_reward()" << std::endl;
#endif
		// theta position
		double fitness_1 = abs((delta_theta(pend.at(pend.size()-1).theta))/2*PI*tp_weight);
		// theta velocity
		double fitness_2 = abs((pend.at(pend.size()-1).theta_dot)*tv_weight);
		if (fitness_2 > 10) fitness_2 = 10;
		fitness_2 = fitness_2/10;
		// below horizontal axis
		double fitness_ch = 0.0;
		if (pend.at(pend.size()-1).theta > PI) {
			fitness_ch = ch_weight;
			below_horizontal = true;
		}

		double total_fitness;
		total_fitness = fitness_1 + fitness_2 + fitness_ch;
		return total_fitness;
	}

	// calculates the next state given a previously set action with a timestep of 'dt'
	// note: aciton must have been set. use 'get_action'
	void Pendulum::cycle() {
#ifdef PD_DEBUG
		std::cout << pre_debug << "cycle()" << std::endl;
#endif
		PendState nextState;
		// - calculations -
		// torque to theta dd
		nextState.theta_dd = -gravity*cos(pend.at(pend.size() - 1).theta) / (mass_p*length) + torq;
		// theta_dd to theta_dot
		nextState.theta_dot = pend.at(pend.size() - 1).theta_dot + nextState.theta_dd*dt;
		// theta_dot to theta
		nextState.theta = pend.at(pend.size() - 1).theta + nextState.theta_dot*dt;
		// - end calculations -
		// keep theta between 0 and 2*PI
		if (nextState.theta < 0.0) nextState.theta = nextState.theta + 2*PI;
		if (nextState.theta > 2*PI) nextState.theta = nextState.theta - 2*PI;
#ifdef PD_CONSULE
		std::cout << nextState.theta << "," << nextState.theta_dot << "," \
		<< nextState.theta_dd << "," << cos(nextState.theta) << "," << sin(nextState.theta) \
		<< std::endl;
#endif
#ifdef PD_MIDLOG
		std::ofstream fout;
		fout.open("_output/pendulum_midlog.csv", std::ofstream::out | std::ofstream::app);
		//if () fout << "torq, x, y, theta, theta_dot, theta_dd, fitness" << "\n";
		fout << torq_history.at(torq_history.size()-1) << ", " << cos(pend.at(torq_history.size()-1).theta) << \
			", " << sin(pend.at(torq_history.size()-1).theta) << ", " << pend.at(torq_history.size()-1).theta*180/PI << \
			", " << pend.at(torq_history.size()-1).theta_dot << ", " << pend.at(torq_history.size()-1).theta_dd << \
			", " << fitness_history.at(torq_history.size()-1) << "\n";
		fout.close();
#endif
		pend.push_back(nextState); // save new state
		torq_history.push_back(torq); // save torq for log
		fitness = determine_reward(); // determine fitness
		fitness_history.push_back(fitness);
	}

	// --------------------------------------

	void Pendulum::set_pendulum_pos(double in_m, double in_l) {
#ifdef PD_DEBUG
		std::cout << pre_debug << "set_pendulum_pos()" << std::endl;
#endif
		mass_p = in_m;
		length = in_l;
	}

	// set fitness weights
	void Pendulum::set_weights(std::vector <double> in) {
#ifdef PD_DEBUG
		std::cout << pre_debug << "set_weights" << std::endl;
#endif
		tp_weight = in.at(0);  // theta offset 90*
		tv_weight = in.at(1);  // angular velocity
		ch_weight = in.at(2);  // below horizontal
	}

	// return delta theta from PI/2 to input
	// untested - testing FIX
	double Pendulum::delta_theta(double in) {
#ifdef PD_DEBUG
		std::cout << pre_debug << "delta_theta()" << std::endl;
#endif
		double t = 0.0;
		if (in > PI/2 && in < 3/2*PI) {
			// between: 90 and 270
			t = in - PI/2;
		} else {
			if (in < PI/2) {
				// between: 0 and 90
				t = PI/2 - in;
			} else {
				// between: 270 and 360
				t = (2*PI - in - PI + PI/2); // 180 - in + 90
			}
		}
		return t;
	}

	// export all state history to file "output/pend_state_log.csv"
	void Pendulum::export_all_states() {
#ifdef PD_DEBUG
		std::cout << pre_debug << "export_all_states()" << std::endl;
#endif
		std::ofstream fout;
		fout.open("_output/domain_state_log.csv", std::ofstream::out | std::ofstream::trunc);
		for (std::size_t i=0; i<pend.size(); ++i) {
			fout << torq_history.at(i) << ", " << cos(pend.at(i).theta) << \
				", " << sin(pend.at(i).theta) << ", " << pend.at(i).theta*180/PI << \
				", " << pend.at(i).theta_dot << ", " << pend.at(i).theta_dd << \
				", " << fitness_history.at(i) << "\n";
		}
		fout.close();
	}

	// --------------------------------------
	// return current state
	// - theta, theta_dot
	std::vector <double> Pendulum::give_state() {
#ifdef PD_DEBUG
		std::cout << pre_debug << "give_state()" << std::endl;
#endif
		std::vector <double> temp_state;
		temp_state.push_back(pend.at(pend.size()-1).theta);
		temp_state.push_back(pend.at(pend.size()-1).theta_dot);
		return temp_state;
	}

	// get action from controller
	// vector with torq in first element
	void Pendulum::get_action(std::vector <double> in_action) {
#ifdef PD_DEBUG
		std::cout << pre_debug << "get_action()" << std::endl;
#endif
		torq = in_action.at(0);
		cycle();
	}

	// return reward for current state
	std::vector <double> Pendulum::give_fitness() {
#ifdef PD_DEBUG
		std::cout << pre_debug << "give_fitness()" << std::endl;
#endif
		std::vector <double> total_fitness;
		total_fitness.push_back(fitness);
		return total_fitness;
	}

	// create update sent to system
	// output
	std::vector <double> Pendulum::update_out() {
#ifdef PD_DEBUG
		std::cout << pre_debug << "update_out()" << std::endl;
#endif
		std::vector <double> t;
		// index 0: domain fail call (bool) below_horizontal
		if (below_horizontal) t.push_back(1.0);
		else t.push_back(0.0);
		// index 1: ()

		//
		return t;
	}

	// manage update in from system
	// input
	void Pendulum::update_in(std::vector <double> in) {
#ifdef PD_DEBUG
		std::cout << pre_debug << "update_in()" << std::endl;
#endif
		// index 0: export_all_states() (bool)
		if (in.at(0) == 1.0) export_all_states();
		// index 1:

	}

}

#endif
