
// Sine Wave Trainer
//
// Evolves Neural Network v3.0 to approximate sin(x) on [-pi, pi]
// using the same evolutionary loop as trainer_3_0:
//   evaluate -> tournament prune -> populate with mutated copies
//
// usage: ./run [seed] [key=value ...]
//   seed=N        random seed (default: time)
//   hidden=8,8    hidden layer sizes, comma separated (e.g. hidden=8 for 1-8-1)
//   pop=100       population size
//   gens=3000     generations
//   chance=0.3    chance per weight to mutate
//   mod=0.1       mutation step size
//   mut=net       net: Network::mutate (uniform difference)
//                 gauss: Gaussian step with sigma = mod
//   anneal=1      1: shrink mod from mod to mod/20 over the run; 0: fixed
//   elite=1       1: best network always survives
//   out=outputs   output folder
// outputs:
//   fitness.csv     generation, best_mse, avg_mse
//   snapshots.csv   generation, x, target, prediction   (best network at chosen generations)
//   weights.txt     final best network weights
// --------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
#include <sstream>
#include <random>

#include "../network/neural_network.h"

// - SETTINGS (defaults, override with key=value) -
static unsigned int POPULATION  = 100;
static unsigned int GENERATIONS = 3000;
static const unsigned int SAMPLES = 64;     // points on [-pi, pi]
// defaults are the best settings from IMPROVEMENTS.md;
// original defaults: hidden=8 chance=0.1 mod=0.2 anneal=0
static std::vector <unsigned int> HIDDEN = {8, 8};
static double MUTATE_MOD        = 0.1;      // mutation step size
static double MUTATE_CHANCE     = 0.3;      // chance per weight
static std::string MUTATION     = "net";    // net | gauss
static bool ANNEAL              = true;
static bool KEEP_BEST           = true;     // elitism
static std::string OUT          = "outputs";
// -   END    -

typedef std::vector <std::vector <std::vector <double> > > Weights;

struct Sample { double x; double input; double target; };

std::mt19937 rng;

std::vector <Sample> make_samples() {
	std::vector <Sample> s;
	for (unsigned int i=0; i<SAMPLES; ++i) {
		double x = -M_PI + 2.0*M_PI*i/(SAMPLES-1);
		// scale input to [-1, 1]; target sin(x) already fits tanh's [-1, 1] range
		s.push_back({x, x/M_PI, std::sin(x)});
	}
	return s;
}

double predict(Network::Network& net, double input) {
	std::vector <double> in(1, input);
	return net.cycle(in).at(0);
}

// mean squared error over all samples (lower is better)
double evaluate(Network::Network& net, const std::vector <Sample>& samples) {
	double err = 0.0;
	for (const Sample& s : samples) {
		double d = predict(net, s.input) - s.target;
		err += d*d;
	}
	return err / samples.size();
}

Network::Network make_network() {
	std::vector <unsigned int> npl = {1};
	npl.insert(npl.end(), HIDDEN.begin(), HIDDEN.end());
	npl.push_back(1);
	Network::Network net;
	net.run_type = 1;  // quiet setup
	net.setup(npl, MUTATE_MOD, MUTATE_CHANCE);
	return net;
}

// mutate a copy; 'mod' may be annealed
void mutate(Network::Network& net, double mod) {
	if (MUTATION == "net" && !ANNEAL) { net.mutate(); return; }
	std::uniform_real_distribution <double> u(0.0, 1.0);
	std::normal_distribution <double> g(0.0, 1.0);
	Weights w = net.export_weights();
	for (auto& layer : w)
		for (auto& node : layer)
			for (double& v : node)
				if (u(rng) < MUTATE_CHANCE) {
					if (MUTATION == "gauss") v += g(rng) * mod;
					else v += u(rng)*mod - u(rng)*mod;  // same shape as Network::mutate
				}
	net.import_weights(w);
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
		else if (k == "chance") MUTATE_CHANCE = std::stod(v);
		else if (k == "mod") MUTATE_MOD = std::stod(v);
		else if (k == "mut") MUTATION = v;
		else if (k == "anneal") ANNEAL = (v == "1");
		else if (k == "elite") KEEP_BEST = (v == "1");
		else if (k == "out") OUT = v;
		else if (k == "hidden") {
			HIDDEN.clear();
			std::stringstream ss(v); std::string t;
			while (std::getline(ss, t, ',')) HIDDEN.push_back(std::stoul(t));
		}
		else { std::cerr << "unknown setting: " << k << std::endl; return false; }
	}
	return true;
}

int main(int argc, char** argv) {
	unsigned int seed = (unsigned int)time(0);
	if (!parse_args(argc, argv, seed)) return 1;
	srand(seed);
	rng.seed(seed);

	std::string shape = "1";
	for (unsigned int h : HIDDEN) shape += "-" + std::to_string(h);
	shape += "-1";
	std::cout << "Sine Wave Trainer - Neural Network v3.0" << std::endl;
	std::cout << "network: " << shape << "  population: " << POPULATION
		<< "  generations: " << GENERATIONS << "  seed: " << seed
		<< "  mutation: " << MUTATION << (ANNEAL ? "+anneal" : "")
		<< " chance " << MUTATE_CHANCE << " mod " << MUTATE_MOD
		<< "  elite: " << KEEP_BEST << std::endl;

	std::vector <Sample> samples = make_samples();
	std::vector <unsigned int> snapshots = {0, 10, 50, 200, 1000, GENERATIONS};

	std::vector <Network::Network> pop;
	for (unsigned int i=0; i<POPULATION; ++i) pop.push_back(make_network());

	std::ofstream f_fit(OUT + "/fitness.csv");
	std::ofstream f_snap(OUT + "/snapshots.csv");
	if (!f_fit.is_open() || !f_snap.is_open()) {
		std::cerr << "ERROR: could not open " << OUT << "/ for writing" << std::endl;
		return 1;
	}
	f_fit << "generation,best_mse,avg_mse\n";
	f_snap << "generation,x,target,prediction\n";

	Network::Network best = pop.at(0);
	double best_err = HUGE_VAL;
	clock_t start = clock();

	for (unsigned int gen=0; gen<=GENERATIONS; ++gen) {
		// 1. evaluate
		std::vector <double> err(pop.size());
		double sum = 0.0;
		unsigned int gen_best = 0;
		for (std::size_t i=0; i<pop.size(); ++i) {
			err.at(i) = evaluate(pop.at(i), samples);
			sum += err.at(i);
			if (err.at(i) < err.at(gen_best)) gen_best = i;
		}
		if (err.at(gen_best) < best_err) {
			best_err = err.at(gen_best);
			best = pop.at(gen_best);
		}
		f_fit << gen << "," << err.at(gen_best) << "," << sum/pop.size() << "\n";

		// snapshot of this generation's best network
		if (std::find(snapshots.begin(), snapshots.end(), gen) != snapshots.end()) {
			for (const Sample& s : samples)
				f_snap << gen << "," << s.x << "," << s.target << ","
					<< predict(pop.at(gen_best), s.input) << "\n";
			std::cout << "gen " << gen << "  best mse: " << err.at(gen_best)
				<< "  avg mse: " << sum/pop.size() << std::endl;
		}
		if (gen == GENERATIONS) break;

		// 2. prune: binary tournaments keep half the population
		std::vector <Network::Network> survivors;
		if (KEEP_BEST) survivors.push_back(pop.at(gen_best));
		while (survivors.size() < POPULATION/2) {
			std::size_t a = rand() % pop.size();
			std::size_t b = rand() % pop.size();
			while (a == b) b = rand() % pop.size();
			survivors.push_back(err.at(a) < err.at(b) ? pop.at(a) : pop.at(b));
		}

		// 3. populate: mutated copies of random survivors
		double mod = MUTATE_MOD;
		if (ANNEAL) mod = MUTATE_MOD * std::pow(0.05, (double)gen / GENERATIONS);
		pop = survivors;
		while (pop.size() < POPULATION) {
			Network::Network child = survivors.at(rand() % survivors.size());
			mutate(child, mod);
			pop.push_back(child);
		}
	}

	// final best network weights
	std::ofstream f_w(OUT + "/weights.txt");
	Weights w = best.export_weights();
	for (std::size_t l=0; l<w.size(); ++l)
		for (std::size_t n=0; n<w.at(l).size(); ++n) {
			f_w << "layer " << l << " node " << n << ":";
			for (double v : w.at(l).at(n)) f_w << " " << v;
			f_w << "\n";
		}

	std::cout << "best mse: " << best_err << "  time: "
		<< (double)(clock()-start)/CLOCKS_PER_SEC << "s" << std::endl;
	return 0;
}
