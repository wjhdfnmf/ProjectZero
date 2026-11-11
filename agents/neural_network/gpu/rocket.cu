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
//   kernel=auto    auto: warp if every layer is <= 64 wide, else block
//                  warp: one warp per episode, 8 episodes of one network share a block (1.3-1.9x faster for <= 64 wide)
//                  block: one thread block per episode (the original; reproduces earlier runs bit for bit)
//   bench_kernels=1  time block vs warp kernel on one generation, report differences, and exit
//   warp_smem_kb=48  shared-memory budget per warp-kernel block (weights go to global memory above it)
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
	__shared__ float mem[rocket::MEM_FLOATS];
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
		cost_out[blockIdx.x] = rocket::cost(st, P);
		if (final_out) final_out[blockIdx.x] = st;
	}
}

// ------------------------------------------------------------------
// Warp episode kernel: one WARP per episode, W episodes per block. The W
// episodes in a block belong to the same network (W divides n_sc), so they
// share one shared-memory copy of its weights. Lane 0 does the physics; all
// 32 lanes compute the network. Only warp-level syncs inside the loop, so a
// warp never waits for another episode. Bit-identical to episode_kernel.
// ------------------------------------------------------------------
__host__ __device__ inline int warp_scratch(const Topology& t) { return IN_BUF + 2 * t.max_width + rocket::MEM_FLOATS; }

__global__ void episode_kernel_warp(Topology t, const float* __restrict__ params, bool weights_in_shared,
                                    const Scenario* __restrict__ scen, int n_sc, int W, rocket::Params P, int hist,
                                    float* cost_out, State* final_out, TrajPoint* traj, int traj_scen) {
	extern __shared__ float sh[];
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int e = blockIdx.x * W + warp;          // episode index = net * n_sc + scenario
	const int net = e / n_sc, si = e % n_sc;
	const float* gp = params + (size_t)net * t.n_params;
	float* w = sh;
	float* in = (weights_in_shared ? sh + t.n_params : sh) + (size_t)warp * warp_scratch(t);
	float* a = in + IN_BUF;
	float* b = a + t.max_width;
	float* mem = b + t.max_width;
	if (weights_in_shared)
		for (int i = threadIdx.x; i < t.n_params; i += blockDim.x) w[i] = gp[i];
	__syncthreads();
	const float* p = weights_in_shared ? w : gp;

	State st;
	Scenario sc;
	if (lane == 0) { sc = scen[si]; st = rocket::start(sc); }
	int done = 0;
	while (true) {
		if (lane == 0) rocket::make_input(st, P, hist, mem, in);
		__syncwarp();
		const float* y = forward_warp(t, p, in, a, b, lane);
		if (lane == 0) {
			float thr, gim;
			rocket::command(y, P, thr, gim);
			rocket::step(st, thr, gim, sc, P);
			rocket::remember_command(st, P, hist, mem, gim);
			if (traj && si < traj_scen)
				traj[(size_t)si * P.max_steps + st.steps - 1] = {st.x, st.y, st.theta, st.thr, gim, st.pad_x};
			done = st.outcome != rocket::FLYING;
		}
		done = __shfl_sync(0xffffffffu, done, 0);
		if (done) break;
	}
	if (lane == 0) {
		cost_out[e] = rocket::cost(st, P);
		if (final_out) final_out[e] = st;
	}
}

// same episode on the CPU (reference for checking the GPU)
State episode_cpu(const Topology& t, const float* p, const Scenario& sc, const rocket::Params& P, int hist) {
	std::vector<float> in(IN_BUF), mem(rocket::MEM_FLOATS), out(t.outputs()), ba(t.max_width), bb(t.max_width);
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

bool g_hard = false;      // set from mode=hard or mode=real
bool g_real = false;      // set from mode=real
float g_ship_amp = 5.0f;  // hard mode: max drone-ship sway (m), set from ship=
                          // (ship=20 was the first, unrealistic version: deck up to 15.7 m/s)
float g_heave_amp = 1.5f; // real mode: max deck heave (m), set from heave=

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
		s.heave_amp = 0.0f; s.heave_w = 0.0f; s.heave_phase = 0.0f; s.noise_seed = 0;
		if (g_hard) {  // drawn after the easy values, so easy scenarios are unchanged
			s.wind = U(-20, 20); s.gust_amp = U(0, 12);
			s.ship_amp = U(0, 1) * g_ship_amp; s.ship_w = 6.2832f / U(8, 14); s.ship_phase = U(0, 6.2832f);
		}
		if (g_real) {  // drawn after the hard values, so hard scenarios are unchanged
			s.heave_amp = U(0, 1) * g_heave_amp; s.heave_w = 6.2832f / U(6, 10); s.heave_phase = U(0, 6.2832f);
			s.noise_seed = r();
		}
	}
	return v;
}

// runs `n_nets` consecutive networks starting at d_params on scenarios; returns costs [net][scenario]
struct Runner {
	Topology t;
	rocket::Params P;
	int hist;
	bool warp_kernel = true;   // false = the original one-block-per-episode kernel
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
		// warp kernel may use up to 99 KB of shared memory per block (sm_86)
		CUDA_CHECK(cudaFuncSetAttribute(episode_kernel_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
	}
	// warp kernel layout for n_sc scenarios: W episodes per block (W divides n_sc),
	// weights in shared memory if they fit in 99 KB alongside W scratch areas
	int warp_shared_limit = 48 * 1024;  // max bytes of weights+scratch to keep weights in shared memory
	void warp_layout(int n_sc, int& W, bool& w_shared, size_t& bytes) const {
		const size_t wbytes = (size_t)t.n_params * sizeof(float);
		const size_t scratch = (size_t)warp_scratch(t) * sizeof(float);
		for (W = 8; W > 1 && n_sc % W != 0; W /= 2) {}
		w_shared = wbytes + W * scratch <= (size_t)warp_shared_limit;
		bytes = (w_shared ? wbytes : 0) + W * scratch;
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
		if (warp_kernel) {
			int W; bool w_shared; size_t bytes;
			warp_layout(n_sc, W, w_shared, bytes);
			episode_kernel_warp<<<blocks / W, 32 * W, bytes>>>(t, d_params, w_shared, d_sc, n_sc, W, P, hist, d_cost,
			                                                  finals ? d_final : nullptr, traj ? d_traj : nullptr, traj_scen);
		} else {
			episode_kernel<<<blocks, threads, smem>>>(t, d_params, shared, d_sc, n_sc, P, hist, d_cost,
			                                          finals ? d_final : nullptr, traj ? d_traj : nullptr, traj_scen);
		}
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
	std::string out = "outputs", algo = "ga", kernel = "auto";
	bool bench_kernels = false;
	int warp_smem_kb = 48;
	// real mode settings (-1 = use the real-mode default)
	int ignitions = -1, score = -1, delay = -1;
	float noise = -1.0f;
	std::string init_file;
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
		else if (k == "mode") { g_hard = (v == "hard" || v == "real"); g_real = (v == "real"); }
		else if (k == "ship") g_ship_amp = std::stof(v);
		else if (k == "heave") g_heave_amp = std::stof(v);
		else if (k == "ignitions") ignitions = std::stoi(v);
		else if (k == "score") score = std::stoi(v);
		else if (k == "noise") noise = std::stof(v);
		else if (k == "delay") delay = std::stoi(v);
		else if (k == "init") init_file = v;
		else if (k == "kernel") kernel = v;
		else if (k == "bench_kernels") bench_kernels = (v == "1");
		else if (k == "warp_smem_kb") warp_smem_kb = std::stoi(v);
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
	P.real = g_real;
	if (g_real) {  // real-mode defaults, each overridable
		P.max_ignitions = ignitions >= 0 ? ignitions : 3;
		P.score = score >= 0 ? score : 1;
		P.noise = noise >= 0 ? noise : 1.0f;
		P.delay = delay >= 0 ? delay : 2;
	} else if (ignitions >= 0 || score >= 0 || noise >= 0 || delay >= 0) {
		fprintf(stderr, "ignitions/score/noise/delay need mode=real\n"); return 1;
	}
	if (P.delay > rocket::MAX_DELAY) { fprintf(stderr, "delay max is %d\n", rocket::MAX_DELAY); return 1; }
	std::vector<int> sizes = {rocket::n_inputs(P, hist)};
	sizes.insert(sizes.end(), hidden.begin(), hidden.end());
	sizes.push_back(2);
	Topology t = make_topology(sizes);
	const int pop_n = es ? 2 * ecfg.pairs : cfg.pop;

	cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	printf("Booster Landing (%s mode%s) - GPU %s on %s\n", P.real ? "REAL" : (P.hard ? "HARD" : "easy"),
	       P.hard ? (", ship sway up to " + std::to_string((int)g_ship_amp) + " m").c_str() : "",
	       es ? "Evolution Strategies" : "genetic algorithm", prop.name);
	if (P.real)
		printf("real mode: engine starts %d%s, score %s, sensor noise x%g, sensor delay %d steps, deck heave up to %g m\n",
		       P.max_ignitions, P.max_ignitions ? "" : " (unlimited)", P.score ? "safety-first" : "original",
		       P.noise, P.delay, g_heave_amp);
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

	// init=<weights.txt>: start from a trained network (ES: theta = it; GA: every member = it).
	// Transfer learning: the file's network may have FEWER inputs (e.g. a hard-mode
	// network, 10 inputs, into real mode, 12 inputs) as long as the rest of its shape
	// matches. The extra inputs get zero weights, so the network starts out behaving
	// exactly as it did before; training then learns to use them.
	if (!init_file.empty()) {
		std::ifstream f(init_file);
		std::string header;
		if (!f || !std::getline(f, header)) { fprintf(stderr, "cannot read %s\n", init_file.c_str()); return 1; }
		const std::string shape = header.substr(0, header.find(' '));
		std::vector<int> old_sizes;
		{ std::stringstream ss(shape); std::string x; while (std::getline(ss, x, '-')) old_sizes.push_back(std::stoi(x)); }
		bool ok = old_sizes.size() == (size_t)t.n_layers + 1 && old_sizes[0] <= t.size[0];
		for (int l = 1; ok && l <= t.n_layers; ++l) ok = old_sizes[l] == t.size[l];
		if (!ok) {
			fprintf(stderr, "%s holds a %s network; this run is %s (only the input count may grow)\n",
			        init_file.c_str(), shape.c_str(), describe(t).c_str());
			return 1;
		}
		const Topology ot = make_topology(old_sizes);
		std::vector<float> wold(ot.n_params);
		for (float& v : wold) if (!(f >> v)) { fprintf(stderr, "%s: too few weights\n", init_file.c_str()); return 1; }
		std::vector<float> w0(t.n_params, 0.0f);
		for (int l = 0; l < t.n_layers; ++l) {
			const int in_old = ot.size[l], out = t.size[l + 1];
			for (int k = 0; k < in_old; ++k)
				for (int j = 0; j < out; ++j) w0[t.w_off[l] + k * out + j] = wold[ot.w_off[l] + k * out + j];
			for (int j = 0; j < out; ++j) w0[t.b_off[l] + j] = wold[ot.b_off[l] + j];
		}
		if (ot.size[0] < t.size[0])
			printf("transfer: %s -> %s (%d new inputs start with zero weight)\n", shape.c_str(), describe(t).c_str(),
			       t.size[0] - ot.size[0]);
		if (es) {
			CUDA_CHECK(cudaMemcpy(ess->d_theta, w0.data(), t.n_params * sizeof(float), cudaMemcpyHostToDevice));
		} else {
			for (int i = 0; i < cfg.pop; ++i)
				CUDA_CHECK(cudaMemcpy(ga->d_params + (size_t)i * t.n_params, w0.data(), t.n_params * sizeof(float), cudaMemcpyHostToDevice));
		}
		printf("starting from %s\n", init_file.c_str());
	}
	Runner runner(t, P, hist);
	// auto: warp kernel when every layer is <= 64 wide (measured 1.3-1.9x faster),
	// block kernel for wider layers (warp kernel measured 0.8x for 128-wide)
	runner.warp_kernel = (kernel == "warp") || (kernel == "auto" && t.max_width <= 64);
	runner.warp_shared_limit = warp_smem_kb * 1024;

	if (bench_kernels) {
		// time the block and warp kernels on the same population and scenarios,
		// and check they give exactly the same costs
		Population bp(t, cfg);
		auto sc = make_scenarios(n_scen, 1);
		std::vector<float> res[2];
		double sec[2];
		for (int k = 0; k < 2; ++k) {
			runner.warp_kernel = (k == 1);
			runner.run(bp.d_params, cfg.pop, sc);  // warm up
			auto b0 = std::chrono::steady_clock::now();
			for (int r = 0; r < 3; ++r) res[k] = runner.run(bp.d_params, cfg.pop, sc);
			sec[k] = std::chrono::duration<double>(std::chrono::steady_clock::now() - b0).count() / 3;
		}
		int diff = 0, big = 0;
		double m0 = 0, m1 = 0;
		for (size_t i = 0; i < res[0].size(); ++i) {
			diff += res[0][i] != res[1][i];
			big += fabsf(res[0][i] - res[1][i]) > 1.0f;  // a visibly different flight
			m0 += res[0][i]; m1 += res[1][i];
		}
		int W; bool ws; size_t bytes;
		runner.warp_layout(n_scen, W, ws, bytes);
		printf("BENCH net=%s episodes=%d block_kernel=%.3fs warp_kernel=%.3fs speedup=%.2fx episodes_per_block=%d weights_in_shared=%d "
		       "differ=%d differ_by_more_than_1=%d mean_cost_block=%.4f mean_cost_warp=%.4f\n",
		       describe(t).c_str(), cfg.pop * n_scen, sec[0], sec[1], sec[0] / sec[1], W, (int)ws, diff, big,
		       m0 / res[0].size(), m1 / res[1].size());
		return 0;
	}

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
	double fuel_used = 0, mean_cost = 0, ign_used = 0;
	std::ofstream f_test(out + "/test.csv");
	f_test << "scenario,outcome,x,vx,vy,theta_deg,fuel_used,time,ignitions\n";
	for (int s = 0; s < n_test; ++s) {
		const State& st = fin[s];
		counts[st.outcome]++;
		mean_cost += tc[s];
		ign_used += st.ignitions;
		if (st.outcome == rocket::LANDED) fuel_used += st.fuel0 - st.fuel;
		f_test << s << "," << st.outcome << "," << st.x - st.pad_x << "," << st.vx - st.pad_vx << "," << st.vy - st.pad_vy << ","
		       << st.theta * 57.2958f << "," << st.fuel0 - st.fuel << "," << st.t << "," << st.ignitions << "\n";
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
	const char* mode_name = P.real ? "real" : (P.hard ? "hard" : "easy");
	f_w << describe(t) << (hist ? " hist " : " ") << mode_name << "\n";
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
	printf("mean engine starts per flight %.2f\n", ign_used / n_test);
	printf("GPU vs CPU reference: same outcome on %d/%d, max landing-x difference %.3g m\n", same, n_check, max_dx);
	printf("SUMMARY algo=%s mode=%s hist=%d net=%s params=%d landed=%d test=%d crashed=%d tipped=%d oob=%d timeout=%d cost=%.3f fuel=%.1f train_s=%.1f cpu_match=%d ignitions=%.2f\n",
	       algo.c_str(), mode_name, hist, describe(t).c_str(), t.n_params, counts[1], n_test, counts[2],
	       counts[3], counts[4], counts[5], mean_cost / n_test, counts[1] ? fuel_used / counts[1] : 0.0, train_secs, same,
	       ign_used / n_test);
	return 0;
}
