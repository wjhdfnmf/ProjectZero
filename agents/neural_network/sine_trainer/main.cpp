
// Sine Wave Trainer
//
// Evolves Neural Network v3.0 to approximate sin(x) on [-pi, pi]
// using the same evolutionary loop as trainer_3_0:
//   evaluate -> tournament prune -> populate with mutated copies
//
// usage: ./run [seed]
// outputs (./outputs):
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

#include "../network/neural_network.h"

// - SETTINGS -
static const unsigned int POPULATION    = 100;
static const unsigned int GENERATIONS   = 3000;
static const unsigned int SAMPLES       = 64;     // points on [-pi, pi]
static const unsigned int HIDDEN        = 8;      // hidden layer nodes
static double MUTATE_MOD                = 0.2;    // max mutation step
static double MUTATE_CHANCE             = 0.1;    // chance per weight
static const bool KEEP_BEST             = true;   // elitism: best network always survives
// generations to record the best network's curve
static const unsigned int SNAPSHOTS[]   = {0, 10, 50, 200, 1000, GENERATIONS};
// -   END    -

struct Sample { double x; double input; double target; };

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
	std::vector <unsigned int> npl = {1, HIDDEN, 1};
	Network::Network net;
	net.run_type = 1;  // quiet setup
	net.setup(npl, MUTATE_MOD, MUTATE_CHANCE);
	return net;
}

int main(int argc, char** argv) {
	unsigned int seed = (argc > 1) ? (unsigned int)std::atoi(argv[1]) : (unsigned int)time(0);
	srand(seed);

	std::cout << "Sine Wave Trainer - Neural Network v3.0" << std::endl;
	std::cout << "network: 1-" << HIDDEN << "-1  population: " << POPULATION
		<< "  generations: " << GENERATIONS << "  seed: " << seed << std::endl;

	std::vector <Sample> samples = make_samples();

	std::vector <Network::Network> pop;
	for (unsigned int i=0; i<POPULATION; ++i) pop.push_back(make_network());

	std::ofstream f_fit("outputs/fitness.csv");
	std::ofstream f_snap("outputs/snapshots.csv");
	if (!f_fit.is_open() || !f_snap.is_open()) {
		std::cerr << "ERROR: could not open outputs/ for writing (run from sine_trainer/)" << std::endl;
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
		if (std::find(std::begin(SNAPSHOTS), std::end(SNAPSHOTS), gen) != std::end(SNAPSHOTS)) {
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
		pop = survivors;
		while (pop.size() < POPULATION) {
			Network::Network child = survivors.at(rand() % survivors.size());
			child.mutate();
			pop.push_back(child);
		}
	}

	// final best network weights
	std::ofstream f_w("outputs/weights.txt");
	std::vector <std::vector <std::vector <double> > > w = best.export_weights();
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
