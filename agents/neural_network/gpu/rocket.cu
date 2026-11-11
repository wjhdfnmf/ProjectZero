// rocket.cu
//
// Booster Landing trainer. Every generation, every candidate network flies the
// same batch of random scenarios; all candidates x scenarios episodes run at
// once on the GPU, one thread block each.
//
// Two training algorithms:
//   algo=ga  genetic algorithm (gpu_evolve.cuh): elitism + truncation selection
//            + Gaussian mutation of a population of independent networks
//   algo=es  Evolution Strategies (es.cuh): one network, gradient estimated from
//            mirrored perturbations, centered-rank fitness shaping, Adam
//
// usage: rocket.exe [key=value ...]
//   mode=easy      easy | hard (drone ship, 40% minimum throttle, engine lag, stronger wind)
//   ship=5         hard mode: max drone-ship sway in m (20 = the first, unrealistic version)
//   algo=ga        ga | es
//   hidden=64,64   hidden layer sizes
//   hist=0         1: network also sees the previous observation and command (short-term memory)
//   out_init=1     scale of the output layer's initial weights (0.01 = start with outputs near 0)
//   gens=400       generations                scen=16      scenarios per generation
//   seed=1         test=1000    held-out test scenarios      out=outputs
//   GA:  pop=1024  sigma=0.03  sigma_end=0 (anneal to)  chance=1  elites=8  parents=0.1
//   ES:  pairs=512 (population 2*pairs)  sigma=0.02  lr=0.01  lr_end=0 (decay to)  wd=0.005
//
// outputs (out/):
//   fitness.csv       generation, best_cost, median_cost, best_success, seconds, checkpoint_val
//   test.csv          per test scenario: outcome, x/vx relative to pad, vy, theta_deg, fuel_used, time
//   trajectories.csv  first 12 test scenarios, every step
//   weights.txt       chosen network
// --------------------

#include "gpu_nn.cuh"
#include "gpu_evolve.cuh"
#include "es.cuh"
#include "rocket.cuh"

#include <chrono>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>

using namespace gnn;
using rocket::Scenario;
using rocket::State;

struct TrajPoint { float x, y, theta, throttle, gimbal, pad_x; };

// ------------------------------------------------------------------
// GPU episode kernel: block b flies network b / n_sc on scenario b % n_sc
// ------------------------------------------------------------------
constexpr int IN_BUF = 32;  // >= rocket::MAX_INPUTS

__global__ void episode_kernel(Topology t, const float* __restrict__ params, bool weights_in_shared,
                               const Scenario* __restrict__ scen, int n_sc, rocket::Params P, int hist,
                               float* cost_out, State* final_out, TrajPoint* traj, int traj_scen) {
	extern __shared__ float sh[];
	const int net = blockIdx.x / n_sc, si = blockIdx.x % n_sc;
	const float* gp = params + (size_t)net * t.n_params;
	float* w = sh;
	float* in = weights_in_shared ? sh + t.n_params : sh;
	float* a = in + IN_BUF;
	float* b = a + t.max_width;
	__shared__ State st;
	__shared__ Scenario sc;
	__shared__ float mem[rocket::MAX_OBS + 2];
	__shared__ int done;

	if (weights_in_shared)
		for (int i = threadIdx.x; i < t.n_params; i += blockDim.x) w[i] = gp[i];
	if (threadIdx.x == 0) { sc = scen[si]; st = rocket::start(sc); done = 0; }
	__syncthreads();
	const float* p = weights_in_shared ? w : gp;

	while (true) {
		if (threadIdx.x == 0) rocket::make_input(st, P, hist, mem, in);
		__syncthreads();
		const float* y = forward_block(t, p, in, a, b);
		if (threadIdx.x == 0) {
			float thr, gim;
			rocket::command(y, P, thr, gim);
			rocket::step(st, thr, gim, sc, P);
			rocket::remember_command(st, P, hist, mem, gim);
			if (traj && si < traj_scen)
				traj[(size_t)si * P.max_steps + st.steps - 1] = {st.x, st.y, st.theta, st.thr, gim, st.pad_x};
			done = st.outcome != rocket::FLYING;
		}
		__syncthreads();
		if (done) break;
	}
	if (threadIdx.x == 0) {
		cost_out[blockIdx.x] = rocket::cost(st);
		if (final_out) final_out[blockIdx.x] = st;
	}
}

// same episode on the CPU (reference for checking the GPU)
State episode_cpu(const Topology& t, const float* p, const Scenario& sc, const rocket::Params& P, int hist) {
	std::vector<float> in(IN_BUF), mem(rocket::MAX_OBS + 2), out(t.outputs()), ba(t.max_width), bb(t.max_width);
	State st = rocket::start(sc);
	while (st.outcome == rocket::FLYING) {
		rocket::make_input(st, P, hist, mem.data(), in.data());
		forward_serial(t, p, in.data(), out.data(), ba.data(), bb.data());
		float thr, gim;
		rocket::command(out.data(), P, thr, gim);
		rocket::step(st, thr, gim, sc, P);
		rocket::remember_command(st, P, hist, mem.data(), gim);
	}
	return st;
}

bool g_hard = false;      // set from mode=hard
float g_ship_amp = 5.0f;  // hard mode: max drone-ship sway (m), set from ship=
                          // (ship=20 was the first, unrealistic version: deck up to 15.7 m/s)

std::vector<Scenario> make_scenarios(int n, uint32_t seed) {
	std::mt19937 r(seed);
	auto U = [&](float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(r); };
	std::vector<Scenario> v(n);
	for (auto& s : v) {
		s.x = U(-600, 600);   s.y = U(1500, 2500);
		s.vx = U(-40, 40);    s.vy = U(-90, -50);
		s.theta = U(-0.25f, 0.25f); s.omega = U(-0.05f, 0.05f);
		s.fuel = U(4000, 7000);
		s.wind = U(-15, 15);  s.gust_amp = U(0, 8); s.gust_w = U(0.2f, 1.0f); s.gust_phase = U(0, 6.2832f);
		s.ship_amp = 0.0f; s.ship_w = 0.0f; s.ship_phase = 0.0f;
		if (g_hard) {  // drawn after the easy values, so easy scenarios are unchanged
			s.wind = U(-20, 20); s.gust_amp = U(0, 12);
			s.ship_amp = U(0, 1) * g_ship_amp; s.ship_w = 6.2832f / U(8, 14); s.ship_phase = U(0, 6.2832f);
		}
	}
	return v;
}

// runs `n_nets` consecutive networks starting at d_params on scenarios; returns costs [net][scenario]
struct Runner {
	Topology t;
	rocket::Params P;
	int hist;
	bool shared;
	size_t smem;
	int threads;
	Scenario* d_sc = nullptr; int sc_cap = 0;
	float* d_cost = nullptr; int cost_cap = 0;
	State* d_final = nullptr;
	TrajPoint* d_traj = nullptr;

	Runner(const Topology& topo, const rocket::Params& p, int h) : t(topo), P(p), hist(h) {
		const size_t buf = (IN_BUF + 2 * t.max_width) * sizeof(float);
		shared = (t.n_params * sizeof(float) + buf) <= 48 * 1024;
		smem = (shared ? t.n_params * sizeof(float) : 0) + buf;
		threads = std::max(32, std::min(256, ((t.max_width + 31) / 32) * 32));
	}
	void ensure(int n_sc, int n_blocks) {
		if (n_sc > sc_cap) { cudaFree(d_sc); CUDA_CHECK(cudaMalloc(&d_sc, n_sc * sizeof(Scenario))); sc_cap = n_sc; }
		if (n_blocks > cost_cap) {
			cudaFree(d_cost); cudaFree(d_final);
			CUDA_CHECK(cudaMalloc(&d_cost, n_blocks * sizeof(float)));
			CUDA_CHECK(cudaMalloc(&d_final, n_blocks * sizeof(State)));
			cost_cap = n_blocks;
		}
	}
	std::vector<float> run(const float* d_params, int n_nets, const std::vector<Scenario>& sc,
	                       std::vector<State>* finals = nullptr, std::vector<TrajPoint>* traj = nullptr, int traj_scen = 0) {
		const int n_sc = (int)sc.size(), blocks = n_nets * n_sc;
		ensure(n_sc, blocks);
		CUDA_CHECK(cudaMemcpy(d_sc, sc.data(), n_sc * sizeof(Scenario), cudaMemcpyHostToDevice));
		if (traj) {
			cudaFree(d_traj);
			CUDA_CHECK(cudaMalloc(&d_traj, (size_t)traj_scen * P.max_steps * sizeof(TrajPoint)));
			CUDA_CHECK(cudaMemset(d_traj, 0, (size_t)traj_scen * P.max_steps * sizeof(TrajPoint)));
		}
		episode_kernel<<<blocks, threads, smem>>>(t, d_params, shared, d_sc, n_sc, P, hist, d_cost,
		                                          finals ? d_final : nullptr, traj ? d_traj : nullptr, traj_scen);
		CUDA_CHECK(cudaGetLastError());
		std::vector<float> c(blocks);
		CUDA_CHECK(cudaMemcpy(c.data(), d_cost, blocks * sizeof(float), cudaMemcpyDeviceToHost));
		if (finals) { finals->resize(blocks); CUDA_CHECK(cudaMemcpy(finals->data(), d_final, blocks * sizeof(State), cudaMemcpyDeviceToHost)); }
		if (traj) { traj->resize((size_t)traj_scen * P.max_steps); CUDA_CHECK(cudaMemcpy(traj->data(), d_traj, traj->size() * sizeof(TrajPoint), cudaMemcpyDeviceToHost)); }
		return c;
	}
	double mean_cost(const float* d_params, const std::vector<Scenario>& sc) {
		std::vector<float> c = run(d_params, 1, sc);
		double m = 0; for (float v : c) m += v;
		return m / c.size();
	}
	~Runner() { cudaFree(d_sc); cudaFree(d_cost); cudaFree(d_final); cudaFree(d_traj); }
};

static std::vector<int> parse_ints(const std::string& v) {
	std::vector<int> r; std::stringstream ss(v); std::string x;
	while (std::getline(ss, x, ',')) r.push_back(std::stoi(x));
	return r;
}

int main(int argc, char** argv) {
	std::vector<int> hidden = {64, 64};
	EvoConfig cfg; cfg.pop = 1024; cfg.sigma = 0.03f; cfg.elites = 8; cfg.parent_frac = 0.1f;
	ESConfig ecfg;
	int gens = 400, n_scen = 16, n_test = 1000, hist = 0;
	float sigma_end = 0.0f;
	bool sigma_set = false;
	std::string out = "outputs", algo = "ga";
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i]; size_t eq = a.find('=');
		if (eq == std::string::npos) { fprintf(stderr, "expected key=value: %s\n", a.c_str()); return 1; }
		std::string k = a.substr(0, eq), v = a.substr(eq + 1);
		if (k == "hidden") hidden = parse_ints(v);
		else if (k == "pop") cfg.pop = std::stoi(v);
		else if (k == "gens") gens = std::stoi(v);
		else if (k == "scen") n_scen = std::stoi(v);
		else if (k == "sigma") { cfg.sigma = ecfg.sigma = std::stof(v); sigma_set = true; }
		else if (k == "sigma_end") sigma_end = std::stof(v);
		else if (k == "chance") cfg.chance = std::stof(v);
		else if (k == "elites") cfg.elites = std::stoi(v);
		else if (k == "parents") cfg.parent_frac = std::stof(v);
		else if (k == "seed") cfg.seed = ecfg.seed = (uint32_t)std::stoul(v);
		else if (k == "test") n_test = std::stoi(v);
		else if (k == "out") out = v;
		else if (k == "mode") g_hard = (v == "hard");
		else if (k == "ship") g_ship_amp = std::stof(v);
		else if (k == "algo") algo = v;
		else if (k == "hist") hist = std::stoi(v);
		else if (k == "pairs") ecfg.pairs = std::stoi(v);
		else if (k == "lr") ecfg.lr = std::stof(v);
		else if (k == "lr_end") ecfg.lr_end = std::stof(v);
		else if (k == "wd") ecfg.weight_decay = std::stof(v);
		else if (k == "out_init") cfg.final_gain = ecfg.final_gain = std::stof(v);
		else { fprintf(stderr, "unknown setting %s\n", k.c_str()); return 1; }
	}
	if (algo != "ga" && algo != "es") { fprintf(stderr, "algo must be ga or es\n"); return 1; }
	const bool es = algo == "es";
	if (es && !sigma_set) ecfg.sigma = 0.02f;
	rocket::Params P;
	P.hard = g_hard;
	std::vector<int> sizes = {rocket::n_inputs(P, hist)};
	sizes.insert(sizes.end(), hidden.begin(), hidden.end());
	sizes.push_back(2);
	Topology t = make_topology(sizes);
	const int pop_n = es ? 2 * ecfg.pairs : cfg.pop;

	cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	printf("Booster Landing (%s mode%s) - GPU %s on %s\n", P.hard ? "HARD" : "easy",
	       P.hard ? (", ship sway up to " + std::to_string((int)g_ship_amp) + " m").c_str() : "",
	       es ? "Evolution Strategies" : "genetic algorithm", prop.name);
	if (es)
		printf("network %s (%d weights)%s  pairs %d (population %d)  scenarios/gen %d  generations %d  sigma %g  lr %g%s  wd %g  seed %u\n",
		       describe(t).c_str(), t.n_params, hist ? " +memory" : "", ecfg.pairs, pop_n, n_scen, gens, ecfg.sigma,
		       ecfg.lr, ecfg.lr_end > 0 ? (" -> " + std::to_string(ecfg.lr_end)).c_str() : "", ecfg.weight_decay, ecfg.seed);
	else
		printf("network %s (%d weights)%s  population %d  scenarios/gen %d  generations %d  sigma %g%s  seed %u\n",
		       describe(t).c_str(), t.n_params, hist ? " +memory" : "", cfg.pop, n_scen, gens, cfg.sigma,
		       sigma_end > 0 ? (" -> " + std::to_string(sigma_end)).c_str() : "", cfg.seed);

	std::unique_ptr<Population> ga;
	std::unique_ptr<EvolutionStrategy> ess;
	if (es) ess.reset(new EvolutionStrategy(t, ecfg)); else ga.reset(new Population(t, cfg));
	Runner runner(t, P, hist);
	std::ofstream f_fit(out + "/fitness.csv");
	if (!f_fit) { fprintf(stderr, "cannot write to %s/\n", out.c_str()); return 1; }
	f_fit << "generation,best_cost,median_cost,best_success,seconds,checkpoint_val\n";

	// ES keeps the best theta seen on a fixed checkpoint set (checked every 25 generations)
	auto ckpt = make_scenarios(200, 555555u);
	float* d_best_theta = nullptr;
	double best_ckpt = 1e30;
	if (es) CUDA_CHECK(cudaMalloc(&d_best_theta, t.n_params * sizeof(float)));

	auto t0 = std::chrono::steady_clock::now();
	auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
	std::vector<float> mean(pop_n, 0.0f);
	for (int g = 0; g <= gens; ++g) {
		auto sc = make_scenarios(n_scen, cfg.seed * 100003u + g);  // fresh scenarios each generation
		if (es) ess->populate();
		const float* d_pop = es ? ess->d_pop : ga->d_params;
		std::vector<State> fin;
		std::vector<float> c = runner.run(d_pop, pop_n, sc, &fin);
		std::fill(mean.begin(), mean.end(), 0.0f);
		for (int i = 0; i < pop_n; ++i) {
			for (int s = 0; s < n_scen; ++s) mean[i] += c[(size_t)i * n_scen + s];
			mean[i] /= n_scen;
		}
		std::vector<float> sorted = mean; std::sort(sorted.begin(), sorted.end());
		const int best = (int)(std::min_element(mean.begin(), mean.end()) - mean.begin());
		int landed = 0;
		for (int s = 0; s < n_scen; ++s) landed += fin[(size_t)best * n_scen + s].outcome == rocket::LANDED;
		double ck = NAN;
		if (es && (g % 25 == 0 || g == gens)) {
			ck = runner.mean_cost(ess->d_theta, ckpt);
			if (ck < best_ckpt) {
				best_ckpt = ck;
				CUDA_CHECK(cudaMemcpy(d_best_theta, ess->d_theta, t.n_params * sizeof(float), cudaMemcpyDeviceToDevice));
			}
		}
		f_fit << g << "," << sorted[0] << "," << sorted[pop_n / 2] << "," << (float)landed / n_scen << "," << secs() << ",";
		if (!std::isnan(ck)) f_fit << ck;
		f_fit << "\n";
		if (g % std::max(1, gens / 10) == 0 || g == gens) {
			printf("gen %4d  best cost %8.2f  median %8.2f  best lands %2d/%d", g, sorted[0], sorted[pop_n / 2], landed, n_scen);
			if (es) printf("  network cost on checkpoint set %8.2f", runner.mean_cost(ess->d_theta, ckpt));
			printf("  (%.1fs)\n", secs());
		}
		if (g == gens) break;
		const float frac = (float)g / gens;
		if (es) {
			ess->update(mean, frac);
		} else {
			const float sigma = sigma_end > 0 ? cfg.sigma * powf(sigma_end / cfg.sigma, frac) : cfg.sigma;
			ga->next_generation(mean, sigma);
		}
	}
	const double train_secs = secs();

	// candidates: GA = the final generation's 16 best (training scores are noisy);
	//             ES = the final weights and the best checkpoint.
	// Each is re-checked on 500 validation scenarios; the best one is tested.
	std::vector<const float*> cand;
	if (es) {
		cand = {ess->d_theta, d_best_theta};
	} else {
		std::vector<int> rank(pop_n);
		std::iota(rank.begin(), rank.end(), 0);
		std::stable_sort(rank.begin(), rank.end(), [&](int a, int b) { return mean[a] < mean[b]; });
		for (int k = 0; k < std::min(16, pop_n); ++k) cand.push_back(ga->params(rank[k]));
	}
	auto val = make_scenarios(500, 777777u);
	const float* pick = cand[0]; double pick_cost = 1e30;
	for (const float* c : cand) {
		const double m = runner.mean_cost(c, val);
		if (m < pick_cost) { pick_cost = m; pick = c; }
	}

	// final test on held-out scenarios
	auto test = make_scenarios(n_test, 424242u);
	std::vector<State> fin;
	std::vector<TrajPoint> traj;
	const int traj_n = 12;
	std::vector<float> tc = runner.run(pick, 1, test, &fin, &traj, traj_n);
	int counts[6] = {0};
	double fuel_used = 0, mean_cost = 0;
	std::ofstream f_test(out + "/test.csv");
	f_test << "scenario,outcome,x,vx,vy,theta_deg,fuel_used,time\n";
	for (int s = 0; s < n_test; ++s) {
		const State& st = fin[s];
		counts[st.outcome]++;
		mean_cost += tc[s];
		if (st.outcome == rocket::LANDED) fuel_used += st.fuel0 - st.fuel;
		f_test << s << "," << st.outcome << "," << st.x - st.pad_x << "," << st.vx - st.pad_vx << "," << st.vy << ","
		       << st.theta * 57.2958f << "," << st.fuel0 - st.fuel << "," << st.t << "\n";
	}
	std::ofstream f_traj(out + "/trajectories.csv");
	f_traj << "scenario,step,x,y,theta,throttle,gimbal,pad_x\n";
	for (int s = 0; s < traj_n; ++s)
		for (int k = 0; k < fin[s].steps; ++k) {
			const TrajPoint& q = traj[(size_t)s * P.max_steps + k];
			f_traj << s << "," << k << "," << q.x << "," << q.y << "," << q.theta << "," << q.throttle << ","
			       << q.gimbal << "," << q.pad_x << "\n";
		}
	std::vector<float> w(t.n_params);
	CUDA_CHECK(cudaMemcpy(w.data(), pick, t.n_params * sizeof(float), cudaMemcpyDeviceToHost));
	std::ofstream f_w(out + "/weights.txt");
	f_w << describe(t) << (hist ? " hist" : "") << (P.hard ? " hard" : " easy") << "\n";
	for (float v : w) f_w << v << "\n";

	// check the GPU against the CPU reference on the first 200 test scenarios
	int same = 0; const int n_check = std::min(200, n_test);
	double max_dx = 0;
	for (int s = 0; s < n_check; ++s) {
		State c = episode_cpu(t, w.data(), test[s], P, hist);
		same += c.outcome == fin[s].outcome;
		if (c.outcome == fin[s].outcome) max_dx = std::max(max_dx, (double)fabsf(c.x - fin[s].x));
	}
	if (d_best_theta) cudaFree(d_best_theta);

	printf("\ntraining: %.1fs (%.3fs per generation, %lld episodes)\n", train_secs, train_secs / (gens + 1),
	       (long long)(gens + 1) * pop_n * n_scen);
	printf("test on %d new scenarios: landed %d (%.1f%%), crashed %d, tipped %d, out of bounds %d, timeout %d\n",
	       n_test, counts[1], 100.0 * counts[1] / n_test, counts[2], counts[3], counts[4], counts[5]);
	printf("mean test cost %.2f, mean fuel used per landing %.0f kg\n", mean_cost / n_test,
	       counts[1] ? fuel_used / counts[1] : 0.0);
	printf("GPU vs CPU reference: same outcome on %d/%d, max landing-x difference %.3g m\n", same, n_check, max_dx);
	printf("SUMMARY algo=%s mode=%s hist=%d net=%s params=%d landed=%d test=%d crashed=%d tipped=%d oob=%d timeout=%d cost=%.3f fuel=%.1f train_s=%.1f cpu_match=%d\n",
	       algo.c_str(), P.hard ? "hard" : "easy", hist, describe(t).c_str(), t.n_params, counts[1], n_test, counts[2],
	       counts[3], counts[4], counts[5], mean_cost / n_test, counts[1] ? fuel_used / counts[1] : 0.0, train_secs, same);
	return 0;
}
