// gfold.cu
//
// G-FOLD on Booster Landing: the convex-optimization flight computer (gfold.h)
// on its own, and as a teacher for the neural network.
//
// usage: gfold.exe <command> [key=value ...]
//   demo      fly the held-out test scenarios with G-FOLD alone        -> out/test.csv, summary.txt, trajectories.csv
//   trace     one flight, printed step by step (scenario=K)
//   data      fly training scenarios, record (network input, G-FOLD command) pairs -> out/bc_data.bin
//   bc        train a network to imitate G-FOLD (behavior cloning)     -> out/weights.txt (rocket.exe init= format)
//
// settings: mode=real (easy | hard | real)  test=1000  scenario=0  threads=0 (all cores)  out=outputs/gfold
//           data: flights=4000 seed=7            bc: data=<bc_data.bin> hidden=8 epochs=60 lr=0.003 batch=256
// --------------------

#include "gfold.h"
#include "gpu_nn.cuh"

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>

using rocket::Scenario;
using rocket::State;

static std::vector<Scenario> make_scenarios(int n, uint32_t seed, bool hard, bool real, float ship_amp = 5.0f,
                                            float heave_amp = 1.5f) {  // same as rocket.cu
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
		if (hard) {
			s.wind = U(-20, 20); s.gust_amp = U(0, 12);
			s.ship_amp = U(0, 1) * ship_amp; s.ship_w = 6.2832f / U(8, 14); s.ship_phase = U(0, 6.2832f);
		}
		if (real) {
			s.heave_amp = U(0, 1) * heave_amp; s.heave_w = 6.2832f / U(6, 10); s.heave_phase = U(0, 6.2832f);
			s.noise_seed = r();
		}
	}
	return v;
}

static rocket::Params make_params(const std::string& mode) {
	rocket::Params P;
	P.hard = mode == "hard" || mode == "real";
	P.real = mode == "real";
	if (P.real) { P.max_ignitions = 3; P.score = 1; P.noise = 1.0f; P.delay = 2; }
	return P;
}

// G-FOLD command (throttle, gimbal) -> the network's output space (inverse of rocket::command)
static void to_outputs(float thr, float gim, const rocket::Params& P, float* y) {
	if (P.hard) y[0] = thr <= 0.0f ? -0.6f : -0.2f + 1.2f * (thr - P.min_throttle) / (1.0f - P.min_throttle);
	else y[0] = 2.0f * thr - 1.0f;
	y[0] = std::max(-0.99f, std::min(0.99f, y[0]));
	y[1] = std::max(-0.99f, std::min(0.99f, gim / P.gimbal_max));
}

struct FlightLog {
	State st;
	gfold::Guidance g;   // final guidance state (stats)
	double ms = 0;       // computer time
	std::vector<float> traj;  // per step: x, y, theta, thr, gimbal, pad_x, phase, plan_tgo
	std::vector<float> inputs, targets;
	std::vector<std::vector<float>> plans;  // snapshots: t0, then x,y per node (relative to pad)
};

// tuning overrides from the command line (g.<name>=value)
static std::map<std::string, double> g_tune;
static void apply_tune(gfold::Guidance& g) {
	for (auto& kv : g_tune) {
		const std::string& k = kv.first; const double v = kv.second;
		if (k == "rate_x") g.V.rate_x = v; else if (k == "rate_y") g.V.rate_y = v;
		else if (k == "glide") g.V.glide_deg = v; else if (k == "lat_weight") g.V.lat_weight = v;
		else if (k == "tilt") g.V.max_tilt_deg = v; else if (k == "final_tilt_time") g.V.final_tilt_time = v;
		else if (k == "final_tilt") g.V.final_tilt_deg = v; else if (k == "plan_max") g.V.plan_max = v;
		else if (k == "vy_final") g.V.vy_final = v; else if (k == "miss_w") g.V.miss_weight = v; else if (k == "early") g.ignite_early = v != 0;
		else if (k == "replan") g.replan_every = v; else if (k == "dist_use") g.D.use = v;
		else if (k == "kd") g.kd_dist = v; else if (k == "kp_att") g.att_kp = v; else if (k == "kd_att") g.att_kd = v;
		else if (k == "lead") g.attitude_lead = v; else if (k == "ctrl_tilt") g.ctrl_tilt_deg = v; else if (k == "term_h") g.terminal_height = v;
		else if (k == "tt0") g.term_tilt0 = v; else if (k == "tts") g.term_tilt_slope = v;
		else if (k == "divert") g.divert = v != 0; else if (k == "divert_t") g.divert_max_t = v;
		else if (k == "slew") g.phi_slew_deg = v; else if (k == "fb_p") g.fb_p = v; else if (k == "fb_v") g.fb_v = v;
		else if (k == "pad_ff") g.pad_ff = v; else if (k == "pad_tau") g.pad_tau = v;
		else if (k == "tkp") g.term_kp = v; else if (k == "tkv") g.term_kv = v; else if (k == "N") g.N = (int)v;
		else { fprintf(stderr, "unknown tuning key g.%s\n", k.c_str()); exit(1); }
	}
}

static FlightLog fly(const Scenario& sc, const rocket::Params& P, bool keep_traj, bool keep_data, bool verbose = false) {
	FlightLog L;
	auto t0 = std::chrono::steady_clock::now();
	L.g.reset(P);
	apply_tune(L.g);
	L.g.verbose = verbose;
	float mem[rocket::MEM_FLOATS], in[32];
	State st = rocket::start(sc);
	int last_plans = 0;
	while (st.outcome == rocket::FLYING) {
		rocket::make_input(st, P, 0, mem, in);
		float thr, gim;
		L.g.act(in, st.t, thr, gim);
		if (keep_data) {
			const int n = rocket::n_obs(P);
			L.inputs.insert(L.inputs.end(), in, in + n);
			float y[2]; to_outputs(thr, gim, P, y);
			L.targets.insert(L.targets.end(), y, y + 2);
		}
		if (keep_traj && L.g.plans != last_plans && L.g.plan.ok) {
			last_plans = L.g.plans;
			std::vector<float> snap = {st.t, (float)st.pad_x, (float)st.pad_y};
			for (int k = 0; k <= L.g.plan.N; ++k) { snap.push_back((float)L.g.plan.x[k]); snap.push_back((float)L.g.plan.y[k]); }
			L.plans.push_back(snap);
		}
		if (verbose && st.steps % 10 == 0)
			printf("t=%5.2f %-15s h=%7.1f x=%7.1f vx=%6.2f vy=%7.2f | est h=%7.1f x=%7.1f vx=%6.2f vy=%7.2f d=(%5.2f,%5.2f) th=%6.2f phi=%6.2f a=(%5.2f,%5.2f) thr=%.2f cmd=%.2f gim=%6.3f fuel=%5.0f\n",
			       st.t, gfold::phase_name(L.g.phase), st.y - st.pad_y, st.x - st.pad_x, st.vx - st.pad_vx, st.vy - st.pad_vy,
			       L.g.py, L.g.px, L.g.vx, L.g.vy, L.g.D.dx, L.g.D.dy, st.theta * 57.3, L.g.last_phi * 57.3, L.g.last_ax, L.g.last_ay, st.thr, thr, gim, st.fuel);
		rocket::step(st, thr, gim, sc, P);
		if (keep_traj) {
			const float tgo = L.g.plan.ok ? (float)(L.g.plan.tf - (st.t - L.g.plan_t0)) : -1.0f;
			const float row[8] = {st.x, st.y, st.theta, st.thr, gim, st.pad_x, (float)L.g.phase, tgo};
			L.traj.insert(L.traj.end(), row, row + 8);
		}
	}
	L.st = st;
	L.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	return L;
}

static const char* outcome_name(int o) {
	static const char* n[] = {"flying", "landed", "crashed", "tipped", "out_of_bounds", "timeout"};
	return n[o];
}

template <class F> static void parallel_for(int n, int threads, F f) {
	std::atomic<int> next(0);
	std::vector<std::thread> pool;
	for (int t = 0; t < threads; ++t) pool.emplace_back([&] { for (int i; (i = next++) < n;) f(i); });
	for (auto& th : pool) th.join();
}

// ------------------------------------------------------------------
// behavior cloning: plain backprop on a tanh MLP in gpu_nn's parameter layout
// (per layer W[in][out] then b[out]), mini-batch Adam, MSE on the outputs
// ------------------------------------------------------------------
struct MLP {
	std::vector<int> sizes;
	std::vector<float> p;
	std::vector<size_t> off;  // per layer: offset of W
	void init(const std::vector<int>& s, uint32_t seed) {
		sizes = s; off.clear();
		size_t n = 0;
		for (size_t l = 0; l + 1 < s.size(); ++l) { off.push_back(n); n += (size_t)s[l] * s[l + 1] + s[l + 1]; }
		p.assign(n, 0.0f);
		std::mt19937 r(seed);
		for (size_t l = 0; l + 1 < s.size(); ++l) {
			const float a = std::sqrt(6.0f / (s[l] + s[l + 1]));
			for (int i = 0; i < s[l] * s[l + 1]; ++i) p[off[l] + i] = std::uniform_real_distribution<float>(-a, a)(r);
		}
	}
	// forward, keeping every layer's activations
	void forward(const float* x, std::vector<std::vector<float>>& act) const {
		act.resize(sizes.size());
		act[0].assign(x, x + sizes[0]);
		for (size_t l = 0; l + 1 < sizes.size(); ++l) {
			const int ni = sizes[l], no = sizes[l + 1];
			const float* W = &p[off[l]];
			const float* b = W + (size_t)ni * no;
			act[l + 1].assign(b, b + no);
			for (int i = 0; i < ni; ++i) { const float a = act[l][i]; for (int o = 0; o < no; ++o) act[l + 1][o] += a * W[i * no + o]; }
			for (int o = 0; o < no; ++o) act[l + 1][o] = std::tanh(act[l + 1][o]);
		}
	}
};

static int train_bc(const std::vector<float>& X, const std::vector<float>& Y, int nin, const std::vector<int>& hidden,
                    int epochs, float lr, int batch, uint32_t seed, const std::string& out_path, const std::string& mode) {
	const size_t n = Y.size() / 2;
	std::vector<int> sizes = {nin};
	sizes.insert(sizes.end(), hidden.begin(), hidden.end());
	sizes.push_back(2);
	MLP net; net.init(sizes, seed);
	std::vector<float> m1(net.p.size(), 0), m2(net.p.size(), 0), grad(net.p.size());
	std::vector<size_t> idx(n);
	for (size_t i = 0; i < n; ++i) idx[i] = i;
	std::mt19937 r(seed);
	// hold out 5% to check generalization
	std::shuffle(idx.begin(), idx.end(), r);
	const size_t n_val = n / 20, n_tr = n - n_val;
	std::vector<std::vector<float>> act, delta(sizes.size());
	long t_adam = 0;
	const int L = (int)sizes.size() - 1;
	auto loss_on = [&](size_t from, size_t to) {
		double s = 0;
		for (size_t j = from; j < to; ++j) {
			const size_t i = idx[j];
			net.forward(&X[i * nin], act);
			for (int o = 0; o < 2; ++o) { const double e = act[L][o] - Y[i * 2 + o]; s += e * e; }
		}
		return s / (to - from) / 2;
	};
	printf("behavior cloning: %zu samples (%zu held out), network", n, n_val);
	for (int s : sizes) printf(" %d", s);
	printf(", %d epochs\n", epochs);
	for (int ep = 0; ep < epochs; ++ep) {
		std::shuffle(idx.begin(), idx.begin() + n_tr, r);
		const float lr_ep = lr * (0.5f * (1.0f + std::cos(3.14159f * ep / epochs)));  // cosine decay
		for (size_t b0 = 0; b0 < n_tr; b0 += batch) {
			const size_t b1 = std::min(n_tr, b0 + batch);
			std::fill(grad.begin(), grad.end(), 0.0f);
			for (size_t j = b0; j < b1; ++j) {
				const size_t i = idx[j];
				net.forward(&X[i * nin], act);
				delta[L].resize(2);
				for (int o = 0; o < 2; ++o) delta[L][o] = (act[L][o] - Y[i * 2 + o]) * (1 - act[L][o] * act[L][o]);
				for (int l = L - 1; l >= 0; --l) {
					const int ni = sizes[l], no = sizes[l + 1];
					float* gW = &grad[net.off[l]];
					float* gb = gW + (size_t)ni * no;
					const float* W = &net.p[net.off[l]];
					for (int o = 0; o < no; ++o) gb[o] += delta[l + 1][o];
					for (int i2 = 0; i2 < ni; ++i2) for (int o = 0; o < no; ++o) gW[i2 * no + o] += act[l][i2] * delta[l + 1][o];
					if (l > 0) {
						delta[l].assign(ni, 0.0f);
						for (int i2 = 0; i2 < ni; ++i2) {
							float s = 0; for (int o = 0; o < no; ++o) s += W[i2 * no + o] * delta[l + 1][o];
							delta[l][i2] = s * (1 - act[l][i2] * act[l][i2]);
						}
					}
				}
			}
			++t_adam;
			const float inv = 1.0f / (b1 - b0), b1c = 1 - std::pow(0.9f, (float)t_adam), b2c = 1 - std::pow(0.999f, (float)t_adam);
			for (size_t k = 0; k < net.p.size(); ++k) {
				const float g = grad[k] * inv;
				m1[k] = 0.9f * m1[k] + 0.1f * g; m2[k] = 0.999f * m2[k] + 0.001f * g * g;
				net.p[k] -= lr_ep * (m1[k] / b1c) / (std::sqrt(m2[k] / b2c) + 1e-8f);
			}
		}
		if (ep % 5 == 4 || ep == epochs - 1)
			printf("  epoch %3d  train mse %.5f  held-out mse %.5f\n", ep + 1, loss_on(0, std::min(n_tr, (size_t)50000)), loss_on(n_tr, n));
	}
	// write in rocket.cu's weights.txt format
	std::ofstream f(out_path);
	for (size_t i = 0; i < sizes.size(); ++i) f << (i ? "-" : "") << sizes[i];
	f << " " << mode << "\n";
	f.precision(9);
	for (float v : net.p) f << v << "\n";
	printf("wrote %s\n", out_path.c_str());
	return 0;
}

int main(int argc, char** argv) {
	if (argc < 2) { fprintf(stderr, "usage: gfold.exe demo|trace|data|bc [key=value ...]\n"); return 1; }
	const std::string cmd = argv[1];
	std::map<std::string, std::string> kv;
	for (int i = 2; i < argc; ++i) { std::string a = argv[i]; auto e = a.find('='); if (e != std::string::npos) kv[a.substr(0, e)] = a.substr(e + 1); }
	auto get = [&](const char* k, const char* d) { return kv.count(k) ? kv[k] : std::string(d); };
	for (auto& p : kv) if (p.first.rfind("g.", 0) == 0) g_tune[p.first.substr(2)] = std::stod(p.second);
	const std::string mode = get("mode", "real"), out = get("out", "outputs/gfold");
	const rocket::Params P = make_params(mode);
	int threads = std::stoi(get("threads", "0"));
	if (threads <= 0) threads = std::max(1u, std::thread::hardware_concurrency());
	std::string mk = "mkdir \"" + out + "\" 2>nul";
	for (auto& c : mk) if (c == '/') c = '\\';
	system(mk.c_str());

	if (cmd == "solve") {  // one planning problem: status per burn time
		gfold::Vehicle V; gfold::Disturbance D;
		const double x = std::stod(get("x", "-180")), y = std::stod(get("y", "1527")), vx = std::stod(get("vx", "-13")),
		             vy = std::stod(get("vy", "-47")), m = std::stod(get("m", "24700"));
		const int N = std::stoi(get("N", "20")), it = std::stoi(get("iters", "3000"));
		if (!P.hard) { V.min_throttle = 0; V.plan_min = 0.02; }
		if (kv.count("u0x")) { V.has_u0 = true; V.u0x = std::stod(kv["u0x"]); V.u0y = std::stod(kv["u0y"]); }
		if (kv.count("rate_x")) V.rate_x = std::stod(kv["rate_x"]);
		if (kv.count("final_tilt_time")) V.final_tilt_time = std::stod(kv["final_tilt_time"]);
		if (kv.count("glide")) V.glide_deg = std::stod(kv["glide"]);
		const double t_lo = std::stod(get("tf_lo", "4")), t_step = std::stod(get("tf_step", "4")), t_hi = std::stod(get("tf_hi", "60"));
		const bool soft = get("soft", "0") == "1";
		for (double tf = t_lo; tf <= t_hi + 1e-9; tf += t_step) {
			std::vector<double> ex, ey;
			gfold::reference_ext(V, D, vx, vy, m, tf, N, ex, ey);
			auto t0 = std::chrono::steady_clock::now();
			gfold::Plan p = gfold::plan_fixed(V, x, y, vx, vy, m, tf, ex, ey, soft, it);
			const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
			printf("tf=%5.1f status=%d iters=%4d fuel=%7.1f miss=%6.1f vxN=%6.2f  %.1f ms\n", tf, p.status, p.iters, p.fuel, p.miss, p.ok ? p.vx.back() : 0.0, ms);
			if (p.ok && get("print", "0") == "1")
				for (int k = 0; k < p.N; ++k) printf("   k=%2d x=%8.1f y=%8.1f vx=%7.2f vy=%7.2f ux=%6.2f uy=%6.2f\n", k, p.x[k], p.y[k], p.vx[k], p.vy[k], p.ux[k], p.uy[k]);
		}
		return 0;
	}

	if (cmd == "trace") {
		auto test = make_scenarios(std::stoi(get("scenario", "0")) + 1, 424242u, P.hard, P.real);
		FlightLog L = fly(test.back(), P, false, false, true);
		const State& s = L.st;
		printf("%s  x=%.2f vx=%.2f vy=%.2f theta=%.2f deg omega=%.3f fuel used %.0f kg  t=%.1f s  ignitions %d\n",
		       outcome_name(s.outcome), s.x - s.pad_x, s.vx - s.pad_vx, s.vy - s.pad_vy, s.theta * 57.2958, s.omega,
		       s.fuel0 - s.fuel, s.t, s.ignitions);
		printf("guidance: %d plans, %d failed, %d solves, %.0f ms, ignition at %.2f s\n", L.g.plans, L.g.failed, L.g.solves, L.ms, L.g.t_ignite);
		return 0;
	}

	if (cmd == "demo") {
		const int n = std::stoi(get("test", "1000"));
		auto test = make_scenarios(n, 424242u, P.hard, P.real);
		std::vector<FlightLog> logs(n);
		auto t0 = std::chrono::steady_clock::now();
		std::atomic<int> done(0);
		parallel_for(n, threads, [&](int i) {
			logs[i] = fly(test[i], P, i < 12, false);
			const int d = ++done;
			if (d % 100 == 0) printf("  %d / %d flights\n", d, n), fflush(stdout);
		});
		const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		int cnt[6] = {0};
		double fuel = 0, ltime = 0, ms = 0, ms_max = 0, solves = 0, plans = 0;
		int landed = 0;
		std::ofstream f(out + "/test.csv");
		f << "scenario,outcome,x,vx,vy,theta_deg,fuel_used,time,ignitions,t_ignite,plans,solves,failed_plans,cpu_ms\n";
		for (int i = 0; i < n; ++i) {
			const State& s = logs[i].st;
			cnt[s.outcome]++;
			ms += logs[i].ms; ms_max = std::max(ms_max, logs[i].ms); solves += logs[i].g.solves; plans += logs[i].g.plans;
			if (s.outcome == rocket::LANDED) { ++landed; fuel += s.fuel0 - s.fuel; ltime += s.t; }
			f << i << "," << s.outcome << "," << s.x - s.pad_x << "," << s.vx - s.pad_vx << "," << s.vy - s.pad_vy << ","
			  << s.theta * 57.2958 << "," << s.fuel0 - s.fuel << "," << s.t << "," << s.ignitions << "," << logs[i].g.t_ignite << ","
			  << logs[i].g.plans << "," << logs[i].g.solves << "," << logs[i].g.failed << "," << logs[i].ms << "\n";
		}
		char buf[1024];
		snprintf(buf, sizeof buf,
		         "G-FOLD alone, %s mode, %d test flights (seed 424242)\n"
		         "landed %d (%.1f%%), crashed %d, tipped %d, out of bounds %d, timeout %d\n"
		         "landed flights: average propellant used %.0f kg, average landing time %.1f s\n"
		         "computer: %.1f plans and %.0f convex solves per flight, %.0f ms per flight on average (max %.0f ms), %.1f s total on %d threads\n",
		         mode.c_str(), n, landed, 100.0 * landed / n, cnt[2], cnt[3], cnt[4], cnt[5],
		         landed ? fuel / landed : 0.0, landed ? ltime / landed : 0.0, plans / n, solves / n, ms / n, ms_max, sec, threads);
		printf("%s", buf);
		std::ofstream(out + "/summary.txt") << buf;
		printf("SUMMARY gfold mode=%s success=%.4f fuel=%.0f land_time=%.1f\n", mode.c_str(), (double)landed / n, landed ? fuel / landed : 0.0, landed ? ltime / landed : 0.0);
		std::ofstream tr(out + "/trajectories.csv");
		tr << "scenario,step,x,y,theta,throttle,gimbal,pad_x,phase,plan_tgo\n";
		for (int i = 0; i < std::min(n, 12); ++i)
			for (size_t k = 0; k < logs[i].traj.size() / 8; ++k) {
				const float* r = &logs[i].traj[k * 8];
				tr << i << "," << k << "," << r[0] << "," << r[1] << "," << r[2] << "," << r[3] << "," << r[4] << "," << r[5] << "," << (int)r[6] << "," << r[7] << "\n";
			}
		std::ofstream pl(out + "/plans.csv");
		pl << "scenario,plan,t0,pad_x,pad_y,node,x,y\n";
		for (int i = 0; i < std::min(n, 12); ++i)
			for (size_t p = 0; p < logs[i].plans.size(); ++p) {
				const auto& s = logs[i].plans[p];
				for (size_t k = 0; 3 + 2 * k + 1 < s.size(); ++k)
					pl << i << "," << p << "," << s[0] << "," << s[1] << "," << s[2] << "," << k << "," << s[3 + 2 * k] << "," << s[4 + 2 * k] << "\n";
			}
		return 0;
	}

	if (cmd == "data") {
		const int n = std::stoi(get("flights", "4000"));
		const uint32_t seed = (uint32_t)std::stoul(get("seed", "7"));
		auto sc = make_scenarios(n, seed * 1000003u + 11u, P.hard, P.real);
		std::vector<FlightLog> logs(n);
		std::atomic<int> done(0);
		parallel_for(n, threads, [&](int i) {
			logs[i] = fly(sc[i], P, false, true);
			logs[i].g = gfold::Guidance();  // drop the plan
			const int d = ++done;
			if (d % 500 == 0) printf("  %d / %d flights\n", d, n), fflush(stdout);
		});
		const int nin = rocket::n_obs(P);
		size_t rows = 0; int landed = 0;
		// only landed flights are imitated ("filtered" behavior cloning): copying crashes would teach crashes
		for (auto& L : logs) { landed += L.st.outcome == rocket::LANDED; if (L.st.outcome == rocket::LANDED) rows += L.targets.size() / 2; }
		std::ofstream f(out + "/bc_data.bin", std::ios::binary);
		const int hdr[2] = {nin, (int)rows};
		f.write((const char*)hdr, sizeof hdr);
		for (auto& L : logs) if (L.st.outcome == rocket::LANDED) f.write((const char*)L.inputs.data(), L.inputs.size() * sizeof(float));
		for (auto& L : logs) if (L.st.outcome == rocket::LANDED) f.write((const char*)L.targets.data(), L.targets.size() * sizeof(float));
		printf("wrote %s: %zu (input, command) pairs from %d flights (G-FOLD landed %d)\n", (out + "/bc_data.bin").c_str(), rows, n, landed);
		return 0;
	}

	if (cmd == "bc") {
		std::ifstream f(get("data", (out + "/bc_data.bin").c_str()), std::ios::binary);
		int hdr[2];
		if (!f.read((char*)hdr, sizeof hdr)) { fprintf(stderr, "cannot read data\n"); return 1; }
		std::vector<float> X((size_t)hdr[0] * hdr[1]), Y((size_t)2 * hdr[1]);
		f.read((char*)X.data(), X.size() * sizeof(float));
		f.read((char*)Y.data(), Y.size() * sizeof(float));
		std::vector<int> hidden;
		std::stringstream hs(get("hidden", "8"));
		for (std::string t; std::getline(hs, t, ',');) hidden.push_back(std::stoi(t));
		return train_bc(X, Y, hdr[0], hidden, std::stoi(get("epochs", "60")), std::stof(get("lr", "0.003")),
		                std::stoi(get("batch", "256")), (uint32_t)std::stoul(get("seed", "1")), out + "/" + get("name", "weights.txt"), mode);
	}
	fprintf(stderr, "unknown command %s\n", cmd.c_str());
	return 1;
}
