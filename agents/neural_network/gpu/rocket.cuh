// rocket.cuh
//
// Booster Landing: a 2D reusable-rocket landing problem.
//
// A ~25 t booster falls toward a 30 m landing pad from about 2 km up, drifting
// sideways, tilted, and pushed by gusting wind it cannot sense. It has one
// gimballed engine (throttle 0-100%, gimbal +/-8.6 deg) and limited fuel.
// It must touch down on the pad, upright, slowly, without tipping over.
//
// HARD MODE (Params::hard): the pad is a drone ship swaying sideways, the engine
// can't throttle below 40% (so a nearly empty booster can't hover and must time
// one "hoverslam" burn), throttle changes lag ~0.3 s, and the wind is stronger.
//
// REAL MODE (Params::real, on top of hard): the engine can only be started
// `max_ignitions` times, the deck also heaves up and down, sensors are noisy and
// delayed, and the score puts landing safely ahead of saving propellant.
// Every real-mode term is exactly zero/off otherwise, so easy and hard mode are
// unchanged.
//
// Everything here is __host__ __device__, so the GPU episode kernel and the CPU
// reference run exactly the same physics.
//
// Frame: x horizontal (pad at x = 0), y altitude (ground at 0). theta is the
// tilt from vertical (positive = nose toward +x), omega its rate.
// --------------------

#pragma once

#include "gpu_nn.cuh"

namespace rocket {

struct Params {
	float g = 9.81f;
	float dry_mass = 20000.0f;     // kg
	float max_thrust = 600000.0f;  // N (thrust-to-weight ~2.2-2.4 when landing)
	float isp = 300.0f;            // s
	float g0 = 9.80665f;
	float arm = 20.0f;             // m, engine to center of mass
	float length = 40.0f;          // m, for moment of inertia
	float gimbal_max = 0.15f;      // rad
	float drag = 20.0f;            // kg/m, F = drag * |v_air| * v_air
	float dt = 0.05f;              // s
	int max_steps = 1600;          // 80 s
	float pad_half = 15.0f;        // m
	float fuel_ref = 7000.0f;      // kg, observation scale
	// HARD MODE (off = the original easy problem, unchanged)
	bool hard = false;
	float min_throttle = 0.4f;     // a lit engine can't go below 40% -> can't hover near empty
	float engine_tau = 0.3f;       // s, throttle lag (spool up / down)
	// REAL MODE (requires hard; off = hard mode unchanged)
	bool real = false;
	int max_ignitions = 0;         // engine starts allowed per flight (0 = unlimited)
	int score = 0;                 // 0 = original cost, 1 = safety first
	float noise = 0.0f;            // sensor noise scale (1 = GNSS/IMU-like, see observe())
	int delay = 0;                 // sensor delay in steps (2 = 0.1 s), max MAX_DELAY
	// any mode: bonus for landing quickly (0 = off, the original scores)
	float time_bonus = 0.0f;       // successful landing: cost -= time_bonus * (1 - flight time / 80 s)
};

// one starting situation (randomized)
struct Scenario {
	float x, y, vx, vy, theta, omega, fuel;
	float wind, gust_amp, gust_w, gust_phase;  // wind = wind + gust_amp*sin(gust_w*t + phase)
	float ship_amp, ship_w, ship_phase;        // hard mode: drone ship sways, pad_x = amp*sin(w*t + phase)
	float heave_amp, heave_w, heave_phase;     // real mode: deck heaves, pad_y = amp*sin(w*t + phase)
	uint32_t noise_seed;                       // real mode: sensor noise stream
};

enum Outcome { FLYING = 0, LANDED = 1, CRASHED = 2, TIPPED = 3, OUT_OF_BOUNDS = 4, TIMEOUT = 5 };

struct State {
	float x, y, vx, vy, theta, omega, fuel, fuel0, t;
	float thr;           // actual throttle (lags the command in hard mode)
	float pad_x, pad_vx; // landing pad position/velocity (always 0 in easy mode)
	float pad_y, pad_vy; // deck height/vertical velocity (always 0 outside real mode)
	int ignitions;       // engine starts used
	uint32_t seed;       // sensor noise stream
	int steps;
	int outcome;
};

__host__ __device__ inline State start(const Scenario& s) {
	State st;
	st.x = s.x; st.y = s.y; st.vx = s.vx; st.vy = s.vy;
	st.theta = s.theta; st.omega = s.omega; st.fuel = s.fuel; st.fuel0 = s.fuel;
	st.t = 0.0f; st.steps = 0; st.outcome = FLYING;
	st.thr = 0.0f;
	st.pad_x = s.ship_amp * sinf(s.ship_phase);
	st.pad_vx = s.ship_amp * s.ship_w * cosf(s.ship_phase);
	st.pad_y = s.heave_amp * sinf(s.heave_phase);
	st.pad_vy = s.heave_amp * s.heave_w * cosf(s.heave_phase);
	st.ignitions = 0;
	st.seed = s.noise_seed;
	return st;
}

// Observations, all roughly in [-1, 1]. Wind is NOT observed.
//   easy (8): x, y, vx, vy, sin, cos, omega, fuel
//   hard (10): position/velocity RELATIVE to the moving pad, + pad velocity, + actual throttle
//   real (12): + engine starts left, + deck vertical velocity; altitude/vertical speed relative to the deck
constexpr int MAX_OBS = 12;
constexpr int MAX_DELAY = 4;
__host__ __device__ inline int n_obs(const Params& P) { return P.real ? 12 : (P.hard ? 10 : 8); }
__host__ __device__ inline void observe(const State& st, const Params& P, float* o) {
	o[0] = (st.x - st.pad_x) / 500.0f;
	o[1] = (st.y - st.pad_y) / 1000.0f;
	o[2] = (st.vx - st.pad_vx) / 50.0f;
	o[3] = (st.vy - st.pad_vy) / 50.0f;
	o[4] = sinf(st.theta);
	o[5] = cosf(st.theta);
	o[6] = st.omega;
	o[7] = st.fuel / P.fuel_ref;
	if (P.hard) {
		o[8] = st.pad_vx / 10.0f;
		o[9] = st.thr;
	}
	if (P.real) {
		o[10] = P.max_ignitions > 0 ? (float)(P.max_ignitions - st.ignitions) / P.max_ignitions : 1.0f;
		o[11] = st.pad_vy / 2.0f;
	}
}

// sensor noise std per observation, in observation units (scaled by Params::noise):
// position 2 m, altitude 1 m, velocities 0.3 m/s, attitude 0.005, rate 0.005 rad/s,
// fuel 14 kg, deck speeds 0.1 / 0.05 m/s. Engine state (o[9], o[10]) is exact.
__host__ __device__ inline float noise_std(int i) {
	switch (i) {
	case 0: return 0.004f;  case 1: return 0.001f;  case 2: return 0.006f;  case 3: return 0.006f;
	case 4: return 0.005f;  case 5: return 0.005f;  case 6: return 0.005f;  case 7: return 0.002f;
	case 8: return 0.01f;   case 11: return 0.025f; default: return 0.0f;
	}
}

// Network input = observation, optionally delayed/noisy (real mode) and with
// short-term memory ("hist"):
//   [ current obs (n) | previous obs (n) | previous throttle, previous gimbal/max ]
// `mem` (MEM_FLOATS) carries state between steps:
//   mem[0 .. n+1]          previous obs + previous command (hist)
//   mem[MAX_OBS+2 ...]     ring of the last delay+1 observations (delay)
// With hist = 0, noise = 0 and delay = 0 this is exactly observe().
constexpr int MAX_INPUTS = 2 * MAX_OBS + 2;
constexpr int RING_OFF = MAX_OBS + 2;
constexpr int MEM_FLOATS = RING_OFF + (MAX_DELAY + 1) * MAX_OBS;
__host__ __device__ inline int n_inputs(const Params& P, int hist) { return hist ? 2 * n_obs(P) + 2 : n_obs(P); }
__host__ __device__ inline void make_input(const State& st, const Params& P, int hist, float* mem, float* in) {
	observe(st, P, in);
	const int n = n_obs(P);
	if (P.noise > 0.0f) {
		for (int i = 0; i < n; ++i) {
			const float sd = noise_std(i);
			if (sd > 0.0f) {
				const uint32_t h = gnn::hash4(st.seed, (uint32_t)st.steps, (uint32_t)i, 0x5E45u);
				in[i] += P.noise * sd * gnn::normal(h, gnn::hash32(h ^ 0xA5A5A5A5u));
			}
		}
	}
	if (P.delay > 0) {
		// sensors arrive `delay` steps late; the engine's own state (o[9], o[10]) is current
		float* ring = mem + RING_OFF;
		const int D = P.delay;
		if (st.steps == 0) for (int k = 0; k <= D; ++k) for (int i = 0; i < n; ++i) ring[k * n + i] = in[i];
		const int slot = st.steps % (D + 1);
		const float thr = in[9], ign = in[10];
		for (int i = 0; i < n; ++i) ring[slot * n + i] = in[i];
		const int old = (st.steps + 1) % (D + 1);  // oldest = written D steps ago
		for (int i = 0; i < n; ++i) in[i] = ring[old * n + i];
		in[9] = thr; in[10] = ign;
	}
	if (!hist) return;
	if (st.steps == 0) { for (int i = 0; i < n; ++i) mem[i] = in[i]; mem[n] = 0.0f; mem[n + 1] = 0.0f; }
	for (int i = 0; i < n + 2; ++i) in[n + i] = mem[i];
	for (int i = 0; i < n; ++i) mem[i] = in[i];
}
__host__ __device__ inline void remember_command(const State& st, const Params& P, int hist, float* mem, float gimbal) {
	if (!hist) return;
	const int n = n_obs(P);
	mem[n] = st.thr;
	mem[n + 1] = gimbal / P.gimbal_max;
}

// network outputs (tanh, -1..1) -> engine commands
//   easy: throttle 0..1
//   hard: output < -0.2 = engine off, otherwise min_throttle..1
__host__ __device__ inline void command(const float* y, const Params& P, float& throttle, float& gimbal) {
	if (P.hard) throttle = (y[0] < -0.2f) ? 0.0f : P.min_throttle + (1.0f - P.min_throttle) * (y[0] + 0.2f) / 1.2f;
	else throttle = 0.5f * (y[0] + 1.0f);
	gimbal = y[1] * P.gimbal_max;
}

// advance one time step; sets st.outcome when the episode ends
__host__ __device__ inline void step(State& st, float throttle, float gimbal, const Scenario& s, const Params& P) {
	const float m = P.dry_mass + st.fuel;
	if (P.hard) {
		// every start from "off" is counted; with a limit (real mode), none left = engine stays off
		if (throttle > 0.0f && st.thr == 0.0f) {
			if (P.max_ignitions > 0 && st.ignitions >= P.max_ignitions) throttle = 0.0f;
			else st.ignitions += 1;
		}
		// engine lag; shutting down (command 0) is immediate
		st.thr = (throttle == 0.0f) ? 0.0f
		       : st.thr + (fmaxf(throttle, P.min_throttle) - st.thr) * fminf(1.0f, P.dt / P.engine_tau);
		if (st.thr > 0.0f) st.thr = fmaxf(st.thr, P.min_throttle);  // lit engine never below minimum
	} else {
		st.thr = throttle;
	}
	const float T = (st.fuel > 0.0f) ? st.thr * P.max_thrust : 0.0f;
	const float dir = st.theta + gimbal;
	const float Fx = T * sinf(dir), Fy = T * cosf(dir);
	const float torque = -T * sinf(gimbal) * P.arm;
	const float I = m * P.length * P.length / 12.0f;
	const float wind = s.wind + s.gust_amp * sinf(s.gust_w * st.t + s.gust_phase);
	const float rvx = st.vx - wind, rvy = st.vy;
	const float speed = sqrtf(rvx * rvx + rvy * rvy);
	const float ax = (Fx - P.drag * speed * rvx) / m;
	const float ay = (Fy - P.drag * speed * rvy) / m - P.g;
	// semi-implicit Euler
	st.vx += ax * P.dt;  st.vy += ay * P.dt;
	st.x += st.vx * P.dt; st.y += st.vy * P.dt;
	st.omega += torque / I * P.dt;
	st.theta += st.omega * P.dt;
	st.fuel = fmaxf(0.0f, st.fuel - T / (P.isp * P.g0) * P.dt);
	st.t += P.dt;
	st.steps += 1;
	if (P.hard) {
		st.pad_x = s.ship_amp * sinf(s.ship_w * st.t + s.ship_phase);
		st.pad_vx = s.ship_amp * s.ship_w * cosf(s.ship_w * st.t + s.ship_phase);
	}
	if (P.real) {
		st.pad_y = s.heave_amp * sinf(s.heave_w * st.t + s.heave_phase);
		st.pad_vy = s.heave_amp * s.heave_w * cosf(s.heave_w * st.t + s.heave_phase);
	}

	if (st.y <= st.pad_y) {
		// landing is judged relative to the (possibly moving) pad
		const bool ok = fabsf(st.x - st.pad_x) <= P.pad_half && fabsf(st.vy - st.pad_vy) <= 3.0f &&
		                fabsf(st.vx - st.pad_vx) <= 1.5f &&
		                fabsf(st.theta) <= 0.0873f && fabsf(st.omega) <= 0.2f;
		st.outcome = ok ? LANDED : CRASHED;
		st.y = st.pad_y;
	} else if (fabsf(st.theta) > 1.5708f) {
		st.outcome = TIPPED;
	} else if (fabsf(st.x) > 3000.0f || st.y > 4000.0f) {
		st.outcome = OUT_OF_BOUNDS;
	} else if (st.steps >= P.max_steps) {
		st.outcome = TIMEOUT;
	}
}

// cost of a finished episode (lower is better)
//   score 0 (original): landing bonus 30 + 10 x share of propellant left
//   score 1 (safety first): landing bonus 60 + 5 x share left; any crash +20
//   time_bonus > 0: landing also earns time_bonus x (1 - flight time / 80 s)
__host__ __device__ inline float cost(const State& st, const Params& P) {
	const float dx = st.x - st.pad_x, dvx = st.vx - st.pad_vx, dvy = st.vy - st.pad_vy;  // == x, vx, vy in easy mode
	const float dy = st.y - st.pad_y;
	const float dist = sqrtf(dx * dx + dy * dy);
	switch (st.outcome) {
	case LANDED:
	case CRASHED: {
		const float impact = sqrtf(dvx * dvx + dvy * dvy);
		float c = 0.02f * fabsf(dx) + impact + 20.0f * fabsf(st.theta) + 5.0f * fabsf(st.omega);
		const float fuel_share = st.fuel / fmaxf(st.fuel0, 1.0f);
		if (P.score == 1) {
			if (st.outcome == LANDED) c -= 60.0f + 5.0f * fuel_share;
			else c += 20.0f;
		} else if (st.outcome == LANDED) {
			c -= 30.0f + 10.0f * fuel_share;  // bonus, more for fuel left
		}
		// optional: faster landings score better (successful landings only)
		if (P.time_bonus > 0.0f && st.outcome == LANDED)
			c -= P.time_bonus * fmaxf(0.0f, 1.0f - st.t / (P.max_steps * P.dt));
		return c;
	}
	case TIMEOUT:
		return 80.0f + 0.02f * dist + 0.2f * fabsf(dvy);
	default:  // TIPPED, OUT_OF_BOUNDS
		return 100.0f + 0.02f * dist;
	}
}

}  // namespace rocket
