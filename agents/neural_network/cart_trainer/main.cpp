
// Cart Balance Trainer
//
// Evolves Neural Network v3.0 to keep an inverted pendulum upright by pushing
// its cart, using Cart Balance by sierra121314 (domains/_not_mine/cart_balance)
// unchanged. Same evolutionary loop as trainer_3_0:
//   evaluate -> tournament prune -> populate with mutated copies
//
// usage: ./run [seed] [key=value ...]
//   seed=N          random seed (default: time)
//   hidden=4        hidden layer sizes, comma separated (e.g. hidden=8,8)
//   pop=100         population size
//   gens=2000       generations
//   steps=200       domain steps per episode (dt = 0.1 s, so 200 = 20 s)
//   starts=95       training start angles in degrees, comma separated (90 = upright)
//   chance=0.1      chance per weight to mutate
//   mod=0.2         mutation step size
//   anneal=0        1: shrink mod from mod to mod/20 over the run; 0: fixed
//   elite=1         1: best network always survives
//   track=1         1: training adds +10 cost per step while the cart is at/off the track end (|x| >= 5 m)
//   cartin=1        1: network also sees cart position and velocity (5 inputs instead of 3)
//   walls=1         1: turn on the domain's optional track walls (cart stops at +/- 5 m)
//   out=outputs     output folder
// trainer_3_0's settings: hidden=4 chance=0.1 mod=0.2 anneal=0 elite=0 starts=95
//
// outputs:
//   fitness.csv     generation, best_cost, avg_cost   (training cost, includes track penalty if on)
//   rollout.csv     best network from 95 deg: step, time, theta_deg, theta_dot, x, x_dot, force, cost
//   eval.csv        best network from every start 45..135 deg:
//                   start, avg_cost (domain only), fell, final_dev_deg, max_abs_x, balanced
//                   balanced = never fell, ends within 10 deg of upright, cart stayed on the track
//   weights.txt     final best network weights
// --------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <math.h>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <string>
#include <sstream>

#include "../network/neural_network.h"
#include "../../../domains/_not_mine/cart_balance/cart_balance.h"

// - SETTINGS (defaults, override with key=value) -
// defaults = "full_walls" in REPORT.md, the best setup for balancing ON the track
static unsigned int POPULATION  = 100;
static unsigned int GENERATIONS = 2000;
static unsigned int STEPS       = 200;
static std::vector <unsigned int> HIDDEN = {4};
static std::vector <double> STARTS = {95.0};
static double MUTATE_MOD        = 0.2;
static double MUTATE_CHANCE     = 0.1;
static bool ANNEAL              = false;
static bool KEEP_BEST           = true;
static bool TRACK_PENALTY       = true;
static bool CART_INPUTS         = true;
static bool WALLS               = true;
static std::string OUT          = "outputs";
// domain settings (same as trainer_3_0)
static const double MASS_P = 1.0, LENGTH = 1.0, MASS_C = 2.0, TRACK = 10.0;
static const double MAX_TORQUE = 0.0, MAX_FORCE = 40.0;
// -   END    -

typedef std::vector <std::vector <std::vector <double> > > Weights;

struct Episode {
	double avg_cost;    // mean domain cost per step (lower is better)
	double avg_penalty; // mean track penalty per step (trainer-side, 0 unless off track)
	bool fell;          // pendulum ever went below horizontal
	double final_dev;   // |theta - 90| at the end, degrees
	double max_abs_x;   // furthest cart position from start, m
	bool balanced() const { return !fell && final_dev < 10.0 && max_abs_x <= TRACK/2; }
};

CB::Pendulum make_domain(double start_deg) {
	return CB::Pendulum(STEPS + 1, start_deg, MASS_P, LENGTH, MAX_TORQUE, MAX_FORCE, MASS_C, TRACK);
}

// network inputs (same as trainer_3_0): cos(theta), sin(theta), theta_dot
// network outputs scaled by max: torque, force
std::vector <double> act(Network::Network& net, CB::Pendulum& d) {
	std::vector <double> s = d.give_state();
	std::vector <double> in = {cos(s.at(0)), sin(s.at(0)), s.at(1)};
	if (CART_INPUTS) {  // cart position scaled so the track edges are -1 and 1
		in.push_back(d.cart_state().x / (TRACK/2));
		in.push_back(d.cart_state().x_dot);
	}
	std::vector <double> out = net.cycle(in);
	return {out.at(0) * MAX_TORQUE, out.at(1) * MAX_FORCE};
}

Episode run_episode(Network::Network& net, double start_deg, std::ofstream* log = nullptr) {
	CB::Pendulum d = make_domain(start_deg);
	d.set_track_walls(WALLS);
	double cost = 0.0, penalty = 0.0, max_x = 0.0;
	bool fell = false;
	for (unsigned int i=0; i<STEPS; ++i) {
		std::vector <double> a = act(net, d);
		double c = d.get_action(a).at(0);
		cost += c;
		fell = fell || d.return_below_horizontal();
		const CB::PendState& p = d.pend_state();
		const CB::CartState& k = d.cart_state();
		max_x = std::max(max_x, std::fabs(k.x));
		// off the track; with walls on the cart can reach the edge but not pass it
		if (std::fabs(k.x) > TRACK/2 - (WALLS ? 1e-9 : 0.0)) penalty += 10.0;
		if (log) *log << i+1 << "," << (i+1)*CB::dt << "," << p.theta*180/M_PI << "," << p.theta_dot
			<< "," << k.x << "," << k.x_dot << "," << d.last_force() << "," << c << "\n";
	}
	double final_dev = d.delta_theta(d.pend_state().theta) * 180 / M_PI;
	return {cost / STEPS, penalty / STEPS, fell, final_dev, max_x};
}

// training cost: domain cost (+ track penalty if on), averaged over start angles
double evaluate(Network::Network& net) {
	double c = 0.0;
	for (double s : STARTS) {
		Episode e = run_episode(net, s);
		c += e.avg_cost + (TRACK_PENALTY ? e.avg_penalty : 0.0);
	}
	return c / STARTS.size();
}

Network::Network make_network() {
	std::vector <unsigned int> npl = {CART_INPUTS ? 5u : 3u};
	npl.insert(npl.end(), HIDDEN.begin(), HIDDEN.end());
	npl.push_back(2);
	Network::Network net;
	net.run_type = 1;
	net.setup(npl, MUTATE_MOD, MUTATE_CHANCE);
	return net;
}

void mutate(Network::Network& net, double mod) {
	if (!ANNEAL) { net.mutate(); return; }
	Weights w = net.export_weights();
	for (auto& layer : w)
		for (auto& node : layer)
			for (double& v : node)
				if (LYRAND < MUTATE_CHANCE) v += LYRAND*mod - LYRAND*mod;  // same shape as Network::mutate
	net.import_weights(w);
}

std::vector <double> parse_list(const std::string& v) {
	std::vector <double> r; std::stringstream ss(v); std::string t;
	while (std::getline(ss, t, ',')) r.push_back(std::stod(t));
	return r;
}

bool parse_args(int argc, char** argv, unsigned int& seed) {
	for (int i=1; i<argc; ++i) {
		std::string a = argv[i];
		std::size_t eq = a.find('=');
		if (eq == std::string::npos) { seed = (unsigned int)std::stoul(a); continue; }
		std::string k = a.substr(0, eq), v = a.substr(eq+1);
		if (k == "seed") seed = (unsigned int)std::stoul(v);
		else if (k == "pop") POPULATION = std::stoul(v);
		else if (k == "gens") GENERATIONS = std::stoul(v);
		else if (k == "steps") STEPS = std::stoul(v);
		else if (k == "chance") MUTATE_CHANCE = std::stod(v);
		else if (k == "mod") MUTATE_MOD = std::stod(v);
		else if (k == "anneal") ANNEAL = (v == "1");
		else if (k == "elite") KEEP_BEST = (v == "1");
		else if (k == "track") TRACK_PENALTY = (v == "1");
		else if (k == "cartin") CART_INPUTS = (v == "1");
		else if (k == "walls") WALLS = (v == "1");
		else if (k == "out") OUT = v;
		else if (k == "starts") STARTS = parse_list(v);
		else if (k == "hidden") {
			HIDDEN.clear();
			for (double h : parse_list(v)) HIDDEN.push_back((unsigned int)h);
		}
		else { std::cerr << "unknown setting: " << k << std::endl; return false; }
	}
	return true;
}

int main(int argc, char** argv) {
	unsigned int seed = (unsigned int)time(0);
	if (!parse_args(argc, argv, seed)) return 1;
	srand(seed);

	std::string shape = CART_INPUTS ? "5" : "3";
	for (unsigned int h : HIDDEN) shape += "-" + std::to_string(h);
	shape += "-2";
	std::cout << "Cart Balance Trainer - Neural Network v3.0 - domain: Cart Balance by sierra121314" << std::endl;
	std::cout << "network: " << shape << "  population: " << POPULATION << "  generations: " << GENERATIONS
		<< "  steps: " << STEPS << "  seed: " << seed << "  starts:";
	for (double s : STARTS) std::cout << " " << s;
	std::cout << "  mutation: chance " << MUTATE_CHANCE << " mod " << MUTATE_MOD
		<< (ANNEAL ? " +anneal" : "") << "  elite: " << KEEP_BEST
		<< "  track penalty: " << TRACK_PENALTY << "  cart inputs: " << CART_INPUTS << std::endl;

	std::ofstream f_fit(OUT + "/fitness.csv");
	if (!f_fit.is_open()) { std::cerr << "ERROR: could not open " << OUT << "/ for writing" << std::endl; return 1; }
	f_fit << "generation,best_cost,avg_cost\n";

	std::vector <Network::Network> pop;
	for (unsigned int i=0; i<POPULATION; ++i) pop.push_back(make_network());

	Network::Network best = pop.at(0);
	double best_cost = HUGE_VAL;
	clock_t start = clock();
	unsigned int report = std::max(1u, GENERATIONS / 5);

	for (unsigned int gen=0; gen<=GENERATIONS; ++gen) {
		std::vector <double> cost(pop.size());
		double sum = 0.0;
		unsigned int gb = 0;
		for (std::size_t i=0; i<pop.size(); ++i) {
			cost.at(i) = evaluate(pop.at(i));
			sum += cost.at(i);
			if (cost.at(i) < cost.at(gb)) gb = i;
		}
		if (cost.at(gb) < best_cost) { best_cost = cost.at(gb); best = pop.at(gb); }
		f_fit << gen << "," << cost.at(gb) << "," << sum/pop.size() << "\n";
		if (gen % report == 0)
			std::cout << "gen " << gen << "  best cost: " << cost.at(gb) << "  avg cost: " << sum/pop.size() << std::endl;
		if (gen == GENERATIONS) break;

		// prune: binary tournaments keep half
		std::vector <Network::Network> survivors;
		if (KEEP_BEST) survivors.push_back(pop.at(gb));
		while (survivors.size() < POPULATION/2) {
			std::size_t a = rand() % pop.size(), b = rand() % pop.size();
			while (a == b) b = rand() % pop.size();
			survivors.push_back(cost.at(a) < cost.at(b) ? pop.at(a) : pop.at(b));
		}
		// populate: mutated copies of random survivors
		double mod = ANNEAL ? MUTATE_MOD * std::pow(0.05, (double)gen / GENERATIONS) : MUTATE_MOD;
		pop = survivors;
		while (pop.size() < POPULATION) {
			Network::Network child = survivors.at(rand() % survivors.size());
			mutate(child, mod);
			pop.push_back(child);
		}
	}

	// rollout of the best network from 95 degrees (trainer_3_0's start)
	std::ofstream f_roll(OUT + "/rollout.csv");
	f_roll << "step,time,theta_deg,theta_dot,x,x_dot,force,cost\n";
	run_episode(best, 95.0, &f_roll);

	// evaluation from many start angles, including ones never trained on
	std::ofstream f_eval(OUT + "/eval.csv");
	f_eval << "start_deg,avg_cost,fell,final_dev_deg,max_abs_x,balanced\n";
	unsigned int upright = 0, balanced = 0, total = 0;
	double eval_cost = 0.0;
	for (int s=45; s<=135; s+=5) {
		Episode e = run_episode(best, s);
		f_eval << s << "," << e.avg_cost << "," << e.fell << "," << e.final_dev << ","
			<< e.max_abs_x << "," << e.balanced() << "\n";
		++total;
		eval_cost += e.avg_cost;
		if (!e.fell && e.final_dev < 10.0) ++upright;
		if (e.balanced()) ++balanced;
	}
	Episode e95 = run_episode(best, 95.0);

	std::ofstream f_w(OUT + "/weights.txt");
	Weights w = best.export_weights();
	for (std::size_t l=0; l<w.size(); ++l)
		for (std::size_t n=0; n<w.at(l).size(); ++n) {
			f_w << "layer " << l << " node " << n << ":";
			for (double v : w.at(l).at(n)) f_w << " " << v;
			f_w << "\n";
		}

	std::cout << "best cost: " << best_cost << "  time: " << (double)(clock()-start)/CLOCKS_PER_SEC << "s" << std::endl;
	// one machine-readable summary line for experiments.sh
	std::cout << "SUMMARY cost95=" << e95.avg_cost << " maxx95=" << e95.max_abs_x
		<< " dev95=" << e95.final_dev << " balanced95=" << e95.balanced()
		<< " upright=" << upright << " balanced=" << balanced << " total=" << total
		<< " eval_cost=" << eval_cost/total << std::endl;
	return 0;
}
