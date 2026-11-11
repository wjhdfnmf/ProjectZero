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
};

// one starting situation (randomized)
struct Scenario {
	float x, y, vx, vy, theta, omega, fuel;
	float wind, gust_amp, gust_w, gust_phase;  // wind = wind + gust_amp*sin(gust_w*t + phase)
	float ship_amp, ship_w, ship_phase;        // hard mode: drone ship sways, pad_x = amp*sin(w*t + phase)
};

enum Outcome { FLYING = 0, LANDED = 1, CRASHED = 2, TIPPED = 3, OUT_OF_BOUNDS = 4, TIMEOUT = 5 };

struct State {
	float x, y, vx, vy, theta, omega, fuel, fuel0, t;
	float thr;           // actual throttle (lags the command in hard mode)
	float pad_x, pad_vx; // landing pad position/velocity (always 0 in easy mode)
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
	return st;
}

// Observations, all roughly in [-1, 1]. Wind is NOT observed.
//   easy (8): x, y, vx, vy, sin, cos, omega, fuel
//   hard (10): position/velocity RELATIVE to the moving pad, + pad velocity, + actual throttle
constexpr int MAX_OBS = 10;
__host__ __device__ inline int n_obs(const Params& P) { return P.hard ? 10 : 8; }
__host__ __device__ inline void observe(const State& st, const Params& P, float* o) {
	o[0] = (st.x - st.pad_x) / 500.0f;
	o[1] = st.y / 1000.0f;
	o[2] = (st.vx - st.pad_vx) / 50.0f;
	o[3] = st.vy / 50.0f;
	o[4] = sinf(st.theta);
	o[5] = cosf(st.theta);
	o[6] = st.omega;
	o[7] = st.fuel / P.fuel_ref;
	if (P.hard) {
		o[8] = st.pad_vx / 10.0f;
		o[9] = st.thr;
	}
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

	if (st.y <= 0.0f) {
		// landing is judged relative to the (possibly moving) pad
		const bool ok = fabsf(st.x - st.pad_x) <= P.pad_half && fabsf(st.vy) <= 3.0f &&
		                fabsf(st.vx - st.pad_vx) <= 1.5f &&
		                fabsf(st.theta) <= 0.0873f && fabsf(st.omega) <= 0.2f;
		st.outcome = ok ? LANDED : CRASHED;
		st.y = 0.0f;
	} else if (fabsf(st.theta) > 1.5708f) {
		st.outcome = TIPPED;
	} else if (fabsf(st.x) > 3000.0f || st.y > 4000.0f) {
		st.outcome = OUT_OF_BOUNDS;
	} else if (st.steps >= P.max_steps) {
		st.outcome = TIMEOUT;
	}
}

// cost of a finished episode (lower is better)
__host__ __device__ inline float cost(const State& st) {
	const float dx = st.x - st.pad_x, dvx = st.vx - st.pad_vx;  // == x, vx in easy mode
	const float dist = sqrtf(dx * dx + st.y * st.y);
	switch (st.outcome) {
	case LANDED:
	case CRASHED: {
		const float impact = sqrtf(dvx * dvx + st.vy * st.vy);
		float c = 0.02f * fabsf(dx) + impact + 20.0f * fabsf(st.theta) + 5.0f * fabsf(st.omega);
		if (st.outcome == LANDED) c -= 30.0f + 10.0f * st.fuel / fmaxf(st.fuel0, 1.0f);  // bonus, more for fuel left
		return c;
	}
	case TIMEOUT:
		return 80.0f + 0.02f * dist + 0.2f * fabsf(st.vy);
	default:  // TIPPED, OUT_OF_BOUNDS
		return 100.0f + 0.02f * dist;
	}
}

}  // namespace rocket
