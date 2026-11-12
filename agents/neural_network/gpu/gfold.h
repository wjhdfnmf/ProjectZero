// gfold.h
//
// G-FOLD powered-descent guidance (Acikmese & Ploen 2007; Blackmore, Acikmese &
// Scharf 2010), 2D, for Booster Landing. Host C++ only, no dependencies.
//
//  1. solve(): a small convex solver for second-order cone programs (SOCP), by
//     operator splitting (ADMM) in the style of OSQP: min q'x  s.t.  A x in C,
//     C = equalities x intervals x second-order cones.
//  2. plan_fixed() / plan(): G-FOLD's convex problem for a given state and burn
//     time: the fuel-optimal thrust profile to the pad. plan() also searches the
//     burn time (free final time).
//  3. Guidance: the closed-loop flight computer. It sees exactly what the neural
//     network sees (noisy, delayed sensors, no wind), estimates the state, coasts,
//     picks the latest safe ignition, re-plans every second, and turns the
//     planned thrust vector into throttle and gimbal commands.
//
// See GFOLD.md for the math.
// --------------------

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "rocket.cuh"

namespace gfold {

// ==================================================================
// 1. Convex solver (dense factorization, sparse rows; ~100-300 variables)
// ==================================================================
enum RowType { EQ = 0, BOX = 1, SOC_HEAD = 2, SOC_TAIL = 3 };
static const double INF = 1e20;

struct Problem {
	int n = 0;
	std::vector<double> q;                              // minimize q'x
	std::vector<std::vector<std::pair<int, double>>> A; // constraint rows
	std::vector<int> type, dim;                         // dim: cone size at a SOC_HEAD row
	std::vector<double> lo, hi;                         // EQ: lo = hi = b; BOX: lo <= a'x <= hi

	int row(int t, double l, double h) {
		A.emplace_back(); type.push_back(t); lo.push_back(l); hi.push_back(h); dim.push_back(0);
		return (int)A.size() - 1;
	}
	void c(int r, int var, double v) { A[r].push_back({var, v}); }
	int m() const { return (int)A.size(); }
};

enum Status { SOLVED = 0, INFEASIBLE = 1, MAX_ITER = 2, NUMERICAL = 3 };
struct Solution { std::vector<double> x; int status = MAX_ITER; int iters = 0; };

static inline bool cholesky(std::vector<double>& K, int n) {
	for (int j = 0; j < n; ++j) {
		double d = K[j * n + j];
		for (int k = 0; k < j; ++k) d -= K[j * n + k] * K[j * n + k];
		if (d <= 1e-300) return false;
		d = std::sqrt(d);
		K[j * n + j] = d;
		for (int i = j + 1; i < n; ++i) {
			double s = K[i * n + j];
			for (int k = 0; k < j; ++k) s -= K[i * n + k] * K[j * n + k];
			K[i * n + j] = s / d;
		}
	}
	return true;
}
static inline void chol_solve(const std::vector<double>& L, int n, double* b) {
	for (int i = 0; i < n; ++i) { double s = b[i]; const double* Li = &L[i * n]; for (int k = 0; k < i; ++k) s -= Li[k] * b[k]; b[i] = s / Li[i]; }
	for (int i = n - 1; i >= 0; --i) { double s = b[i]; for (int k = i + 1; k < n; ++k) s -= L[k * n + i] * b[k]; b[i] = s / L[i * n + i]; }
}

// Euclidean projection onto the second-order cone {(t, v): ||v|| <= t}
static inline void proj_soc(double* p, int d) {
	double nv = 0;
	for (int i = 1; i < d; ++i) nv += p[i] * p[i];
	nv = std::sqrt(nv);
	const double t = p[0];
	if (nv <= t) return;
	if (nv <= -t) { for (int i = 0; i < d; ++i) p[i] = 0; return; }
	const double a = 0.5 * (nv + t);
	p[0] = a;
	for (int i = 1; i < d; ++i) p[i] *= a / nv;
}

// ADMM (OSQP form with a linear objective):
//   x~ = (sigma I + A'RA)^-1 (sigma x - q + A'(R z - y))
//   z  = Proj_C(alpha A x~ + (1-alpha) z + y/R)
//   y += R (alpha A x~ + (1-alpha) z_old - z)
// R = rho per row (x1000 on equalities). Stops on small primal and dual
// residuals, or on a primal-infeasibility certificate (OSQP's test on dy).
static inline Solution solve(const Problem& P0, int max_iter = 3000, double eps = 2e-4) {
	// row equilibration: scale each row (each cone block as a whole) to unit max coefficient
	Problem P = P0;
	for (int r = 0; r < P.m();) {
		const int d = P.type[r] == SOC_HEAD ? P.dim[r] : 1;
		double mx = 0;
		for (int i = r; i < r + d; ++i) for (auto& a : P.A[i]) mx = std::max(mx, std::fabs(a.second));
		const double sc = mx > 0 ? 1.0 / mx : 1.0;
		for (int i = r; i < r + d; ++i) {
			for (auto& a : P.A[i]) a.second *= sc;
			if (P.lo[i] > -INF) P.lo[i] *= sc;
			if (P.hi[i] < INF) P.hi[i] *= sc;
		}
		r += d;
	}
	const int n = P.n, m = P.m();
	const double sigma = 1e-6, alpha = 1.6;
	double rho = 0.1;
	std::vector<double> R(m), L;
	auto set_R = [&] { for (int i = 0; i < m; ++i) R[i] = P.type[i] == EQ ? 1e3 * rho : rho; };
	auto factor = [&] {
		L.assign((size_t)n * n, 0.0);
		for (int j = 0; j < n; ++j) L[j * n + j] = sigma;
		for (int r = 0; r < m; ++r)
			for (auto& a : P.A[r]) for (auto& b : P.A[r]) L[a.first * n + b.first] += R[r] * a.second * b.second;
		return cholesky(L, n);
	};
	Solution S;
	set_R();
	if (!factor()) { S.status = NUMERICAL; return S; }
	std::vector<double> x(n, 0), z(m, 0), y(m, 0), xt(n), zt(m), w(m), Ax(m), Aty(n), yprev(m), dy(m);
	auto mulA = [&](const std::vector<double>& v, std::vector<double>& out) {
		for (int r = 0; r < m; ++r) { double s = 0; for (auto& a : P.A[r]) s += a.second * v[a.first]; out[r] = s; }
	};
	auto mulAt = [&](const std::vector<double>& u, std::vector<double>& out) {
		std::fill(out.begin(), out.end(), 0.0);
		for (int r = 0; r < m; ++r) for (auto& a : P.A[r]) out[a.first] += a.second * u[r];
	};
	auto project = [&](std::vector<double>& v) {
		for (int r = 0; r < m;) {
			if (P.type[r] == EQ) { v[r] = P.lo[r]; ++r; }
			else if (P.type[r] == BOX) { v[r] = std::min(P.hi[r], std::max(P.lo[r], v[r])); ++r; }
			else { proj_soc(&v[r], P.dim[r]); r += P.dim[r]; }
		}
	};
	auto inf_norm = [](const std::vector<double>& v) { double s = 0; for (double a : v) s = std::max(s, std::fabs(a)); return s; };
	for (int it = 1; it <= max_iter; ++it) {
		const bool check = it % 20 == 0 || it == max_iter;
		if (check) yprev = y;
		for (int r = 0; r < m; ++r) w[r] = R[r] * z[r] - y[r];
		mulAt(w, xt);
		for (int j = 0; j < n; ++j) xt[j] += sigma * x[j] - P.q[j];
		chol_solve(L, n, xt.data());
		mulA(xt, zt);
		for (int j = 0; j < n; ++j) x[j] = alpha * xt[j] + (1 - alpha) * x[j];
		for (int r = 0; r < m; ++r) { zt[r] = alpha * zt[r] + (1 - alpha) * z[r]; w[r] = zt[r] + y[r] / R[r]; }
		project(w);
		for (int r = 0; r < m; ++r) { y[r] += R[r] * (zt[r] - w[r]); z[r] = w[r]; }
		if (!check) continue;
		S.iters = it;
		mulA(x, Ax); mulAt(y, Aty);
		double rp = 0, rd = 0;
		for (int r = 0; r < m; ++r) rp = std::max(rp, std::fabs(Ax[r] - z[r]));
		for (int j = 0; j < n; ++j) rd = std::max(rd, std::fabs(P.q[j] + Aty[j]));
		const double sp = 1 + std::max(inf_norm(Ax), inf_norm(z)), sd = 1 + std::max(inf_norm(Aty), inf_norm(P.q));
		if (rp < eps * sp && rd < eps * sd) { S.status = SOLVED; break; }
		// primal infeasibility: dy with A'dy ~ 0 and support function of C at dy < 0
		for (int r = 0; r < m; ++r) dy[r] = y[r] - yprev[r];
		const double ndy = inf_norm(dy);
		if (ndy > 1e-9) {
			mulAt(dy, Aty);
			if (inf_norm(Aty) < 1e-4 * ndy) {
				double sup = 0; bool bounded = true;
				for (int r = 0; r < m && bounded;) {
					if (P.type[r] == EQ) { sup += P.lo[r] * dy[r]; ++r; }
					else if (P.type[r] == BOX) {
						if (dy[r] > 1e-6 * ndy) { if (P.hi[r] >= INF) bounded = false; else sup += P.hi[r] * dy[r]; }
						else if (dy[r] < -1e-6 * ndy) { if (P.lo[r] <= -INF) bounded = false; else sup += P.lo[r] * dy[r]; }
						++r;
					} else {  // cone: support is 0 if -dy is in the cone, else +inf
						const int d = P.dim[r];
						double nt = 0; for (int i = 1; i < d; ++i) nt += dy[r + i] * dy[r + i];
						if (std::sqrt(nt) > -dy[r] + 1e-4 * ndy) bounded = false;
						r += d;
					}
				}
				if (bounded && sup < -1e-4 * ndy) { S.status = INFEASIBLE; break; }
			}
		}
		if (it % 200 == 0) {  // OSQP's rho adaptation
			const double ratio = std::sqrt((rp / sp) / (rd / sd + 1e-12));
			if (ratio > 5 || ratio < 0.2) {
				const double old = rho;
				rho = std::min(1e5, std::max(1e-5, rho * ratio));
				set_R();
				if (!factor()) { rho = old; set_R(); factor(); }
			}
		}
	}
	S.x = x;
	return S;
}

// ==================================================================
// 2. G-FOLD problem
// ==================================================================
// scaled units for the solver: 1 km, 10 s, 100 m/s, 10 m/s^2, 25 t
static const double LU = 1000.0, TU = 10.0, VU = LU / TU, AU = LU / (TU * TU), MU = 25000.0;

struct Vehicle {
	double dry_mass = 20000, max_thrust = 600000, min_throttle = 0.4, isp = 300, g0 = 9.80665, g = 9.81;
	double plan_max = 0.85;     // plans use at most 85% thrust: the rest is margin for feedback
	double plan_min = 0.42;     // and at least 42%
	double max_tilt_deg = 15;   // pointing: thrust within 15 deg of vertical
	double glide_deg = 30;      // glide slope: stay above a 30 deg cone around the pad
	double vy_final = -0.5;     // aim to touch down descending at 1 m/s (relative to the deck)
	double final_tilt_deg = 4, final_tilt_time = 3;  // thrust within 4 deg of vertical in the last 3 s
	double rate_x = 2.0, rate_y = 8.0;               // max change of thrust acceleration, m/s^2 per s
	bool has_u0 = false; double u0x = 0, u0y = 0;    // current thrust acceleration (re-plans)
	double miss_weight = 20.0;  // soft plans: objective weight per km of miss (or per 100 m/s)
	double lat_weight = 1.0;     // objective: + lat_weight x integral of |x| (get over the pad early)
};

// Known external acceleration: drag (rocket.cuh's model, wind assumed 0) plus an
// estimated constant disturbance (unknown wind, deck motion).
struct Disturbance {
	double drag = 20.0;        // kg/m
	double dx = 0, dy = 0;     // estimated extra acceleration (m/s^2)
	double pad_vx = 0, pad_vy = 0;
	void drag_accel(double vxr, double vyr, double m, double& ax, double& ay) const {
		const double vx = vxr + pad_vx, vy = vyr + pad_vy, s = std::sqrt(vx * vx + vy * vy);
		ax = -drag * s * vx / m; ay = -drag * s * vy / m;
	}
	double use = 0.5;          // share of the estimated disturbance that plans assume persists
	void accel(double vxr, double vyr, double m, double& ax, double& ay) const {
		drag_accel(vxr, vyr, m, ax, ay); ax += use * dx; ay += use * dy;
	}
};

struct Plan {
	bool ok = false;
	int status = MAX_ITER;
	double tf = 0, fuel = 0, miss = 0;  // burn time (s), propellant (kg), |x| miss at touchdown (m)
	int N = 0, iters = 0;
	std::vector<double> x, y, vx, vy, m;  // nodes 0..N: m and m/s relative to the pad, kg
	std::vector<double> ux, uy, ex, ey;   // steps 0..N-1: thrust acceleration, known external acceleration (m/s^2)
	double h() const { return tf / N; }
	// state on the plan tau s after its start (exact for zero-order-hold thrust)
	void at(double tau, double& px, double& py, double& pvx, double& pvy, double& ax, double& ay, double g) const {
		tau = std::max(0.0, std::min(tau, tf));
		const int k = std::min(N - 1, (int)(tau / h()));
		const double s = tau - k * h();
		ax = ux[k]; ay = uy[k];
		const double tx = ux[k] + ex[k], ty = uy[k] + ey[k] - g;
		px = x[k] + vx[k] * s + 0.5 * tx * s * s;  py = y[k] + vy[k] * s + 0.5 * ty * s * s;
		pvx = vx[k] + tx * s;                     pvy = vy[k] + ty * s;
	}
	double mass_at(double tau) const { const int k = std::max(0, std::min(N, (int)(tau / h()))); return m[k]; }
};

// Fuel-optimal burn of length tf from (x, y, vx, vy) relative to the pad, mass m0,
// with known external acceleration (ex, ey) per step, ending on the pad with
// velocity (0, vy_final). soft_x: allow a sideways miss, heavily penalized
// (G-FOLD's "minimum landing error" fallback).
static inline Plan plan_fixed(const Vehicle& V, double x0, double y0, double vx0, double vy0, double m0, double tf,
                              const std::vector<double>& ex, const std::vector<double>& ey, bool soft_x = false,
                              int max_iter = 3000) {
	Plan out;
	const int N = (int)ex.size();
	out.tf = tf; out.N = N;
	const double h = tf / TU / N;
	const double ve = V.isp * V.g0 / VU;                                      // exhaust velocity
	const double r1 = V.plan_min * V.max_thrust / (MU * AU), r2 = V.plan_max * V.max_thrust / (MU * AU);
	const double g = V.g / AU, M0 = m0 / MU;
	const double ct = std::cos(V.max_tilt_deg * M_PI / 180), tg = std::tan(V.glide_deg * M_PI / 180);
	// per node k: r(2) v(2) z(1); per step k: u(2) s(1); node N has no step; + miss slack E
	auto R = [&](int k) { return 8 * k; };
	auto Vv = [&](int k) { return 8 * k + 2; };
	auto Z = [&](int k) { return 8 * k + 4; };
	auto U = [&](int k) { return 8 * k + 5; };
	auto S = [&](int k) { return 8 * k + 7; };
	const int E = 8 * N + 5;
	Problem P;
	const int A0 = 8 * N + 6;  // a_k >= |x_k|, nodes 1..N-1
	P.n = 8 * N + 6 + N;
	P.q.assign(P.n, 0.0);
	for (int k = 0; k < N; ++k) P.q[S(k)] = h * M0;                         // ~ propellant used (scaled)
	for (int k = 1; k < N; ++k) P.q[A0 + k] = V.lat_weight * h;
	if (soft_x) P.q[E] = V.miss_weight;                                              // 1 km miss ~ 20 full tanks
	auto eq = [&](double b) { return P.row(EQ, b, b); };
	int r;
	r = eq(x0 / LU);  P.c(r, R(0), 1);
	r = eq(y0 / LU);  P.c(r, R(0) + 1, 1);
	r = eq(vx0 / VU); P.c(r, Vv(0), 1);
	r = eq(vy0 / VU); P.c(r, Vv(0) + 1, 1);
	r = eq(std::log(M0)); P.c(r, Z(0), 1);
	for (int k = 0; k < N; ++k) {
		const double ek[2] = {ex[k] / AU, ey[k] / AU - g};
		for (int d = 0; d < 2; ++d) {  // v' = v + h(u + e);  r' = r + h v + h^2/2 (u + e)
			r = eq(h * ek[d]); P.c(r, Vv(k + 1) + d, 1); P.c(r, Vv(k) + d, -1); P.c(r, U(k) + d, -h);
			r = eq(0.5 * h * h * ek[d]); P.c(r, R(k + 1) + d, 1); P.c(r, R(k) + d, -1); P.c(r, Vv(k) + d, -h); P.c(r, U(k) + d, -0.5 * h * h);
		}
		r = eq(0); P.c(r, Z(k + 1), 1); P.c(r, Z(k), -1); P.c(r, S(k), h / ve);   // z' = z - h s / ve
		// ||u|| <= s  (the lossless-convexification slack)
		r = P.row(SOC_HEAD, 0, 0); P.c(r, S(k), 1); P.dim[r] = 3;
		r = P.row(SOC_TAIL, 0, 0); P.c(r, U(k), 1);
		r = P.row(SOC_TAIL, 0, 0); P.c(r, U(k) + 1, 1);
		// thrust bounds rho1 e^-z <= s <= rho2 e^-z, linearized about z0(t) = ln(m0 - rho2 t / ve)
		const double tk = k * h;
		const double z0 = std::log(std::max(1e-3, M0 - r2 * tk / ve)), e0 = std::exp(-z0);
		r = P.row(BOX, -INF, r2 * e0 * (1 + z0)); P.c(r, S(k), 1); P.c(r, Z(k), r2 * e0);
		r = P.row(BOX, r1 * e0 * (1 + z0), INF);  P.c(r, S(k), 1); P.c(r, Z(k), r1 * e0);
		// mass stays between "full thrust so far" and "minimum thrust so far"
		if (k > 0) { r = P.row(BOX, z0, std::log(std::max(1e-3, M0 - r1 * tk / ve))); P.c(r, Z(k), 1); }
		// pointing: u_y >= cos(max tilt) s
		r = P.row(BOX, 0, INF); P.c(r, U(k) + 1, 1); P.c(r, S(k), -ct);
		// dry mass
		r = P.row(BOX, std::log(V.dry_mass / MU), INF); P.c(r, Z(k + 1), 1);
		// glide slope at nodes 1..N-1: y >= tan(gs) |x|  (dropped in the minimum-miss fallback)
		if (k + 1 < N && !soft_x) {
			r = P.row(BOX, 0, INF); P.c(r, R(k + 1) + 1, 1); P.c(r, R(k + 1), -tg);
			r = P.row(BOX, 0, INF); P.c(r, R(k + 1) + 1, 1); P.c(r, R(k + 1), tg);
		}
		// lateral-distance slack a >= |x| (for the lat_weight objective)
		if (k + 1 < N) {
			r = P.row(BOX, 0, INF); P.c(r, A0 + k + 1, 1); P.c(r, R(k + 1), -1);
			r = P.row(BOX, 0, INF); P.c(r, A0 + k + 1, 1); P.c(r, R(k + 1), 1);
		}
		// final approach: thrust nearly vertical (|u_x| <= tan(final tilt) u_y) in the last seconds
		if ((N - k) * h * TU <= V.final_tilt_time + 1e-9) {
			const double tf_ = std::tan(V.final_tilt_deg * M_PI / 180);
			r = P.row(BOX, 0, INF); P.c(r, U(k) + 1, tf_); P.c(r, U(k), -1);
			r = P.row(BOX, 0, INF); P.c(r, U(k) + 1, tf_); P.c(r, U(k), 1);
		}
		// thrust-vector rate limits: the booster needs time to rotate, the engine to spool
		const double dmx = V.rate_x * h * TU / AU, dmy = V.rate_y * h * TU / AU;
		if (k + 1 < N) {
			r = P.row(BOX, -dmx, dmx); P.c(r, U(k + 1), 1); P.c(r, U(k), -1);
			r = P.row(BOX, -dmy, dmy); P.c(r, U(k + 1) + 1, 1); P.c(r, U(k) + 1, -1);
		}
		if (k == 0 && V.has_u0) {  // start from the thrust the engine is producing now
			r = P.row(BOX, V.u0x / AU - dmx, V.u0x / AU + dmx); P.c(r, U(0), 1);
			r = P.row(BOX, V.u0y / AU - dmy, V.u0y / AU + dmy); P.c(r, U(0) + 1, 1);
		}
	}
	if (soft_x) {  // E >= |x_N| and E >= |vx_N| (scaled: 1 m/s counts like 10 m)
		r = P.row(BOX, 0, INF); P.c(r, E, 1); P.c(r, R(N), -1);
		r = P.row(BOX, 0, INF); P.c(r, E, 1); P.c(r, R(N), 1);
		r = P.row(BOX, 0, INF); P.c(r, E, 1); P.c(r, Vv(N), -1);
		r = P.row(BOX, 0, INF); P.c(r, E, 1); P.c(r, Vv(N), 1);
	} else {
		r = eq(0); P.c(r, R(N), 1);
		r = eq(0); P.c(r, E, 1);
		r = eq(0); P.c(r, Vv(N), 1);
	}
	r = eq(0); P.c(r, R(N) + 1, 1);
	r = eq(V.vy_final / VU); P.c(r, Vv(N) + 1, 1);

	Solution s = solve(P, max_iter);
	out.status = s.status; out.iters = s.iters;
	if (s.status != SOLVED) return out;
	out.ok = true;
	for (int k = 0; k <= N; ++k) {
		out.x.push_back(s.x[R(k)] * LU);   out.y.push_back(s.x[R(k) + 1] * LU);
		out.vx.push_back(s.x[Vv(k)] * VU); out.vy.push_back(s.x[Vv(k) + 1] * VU);
		out.m.push_back(std::exp(s.x[Z(k)]) * MU);
		if (k < N) { out.ux.push_back(s.x[U(k)] * AU); out.uy.push_back(s.x[U(k) + 1] * AU); }
	}
	out.ex = ex; out.ey = ey;
	out.fuel = m0 - out.m.back();
	out.miss = std::fabs(out.x.back());
	return out;
}

// external accelerations on a straight-line reference (velocity linear to the final value)
static inline void reference_ext(const Vehicle& V, const Disturbance& D, double vx0, double vy0, double m0, double tf, int N,
                                 std::vector<double>& ex, std::vector<double>& ey) {
	ex.resize(N); ey.resize(N);
	const double mdot = 0.6 * V.max_thrust / (V.isp * V.g0);
	for (int k = 0; k < N; ++k) {
		const double f = (k + 0.5) / N;
		D.accel(vx0 * (1 - f), vy0 * (1 - f) + V.vy_final * f, std::max(V.dry_mass, m0 - mdot * f * tf), ex[k], ey[k]);
	}
}
// external accelerations along an existing plan's velocity and mass, from time `shift` on
static inline void plan_ext(const Plan& p, const Disturbance& D, double shift, double tf, int N, double g,
                            std::vector<double>& ex, std::vector<double>& ey) {
	ex.resize(N); ey.resize(N);
	for (int k = 0; k < N; ++k) {
		const double tau = shift + (k + 0.5) * tf / N;
		double px, py, vx, vy, ax, ay;
		p.at(tau, px, py, vx, vy, ax, ay, g);
		D.accel(vx, vy, p.mass_at(tau), ex[k], ey[k]);
	}
}

// Free final time: scan burn times, refine with golden section on fuel(tf), then
// re-solve once with drag evaluated on the plan's own velocity profile.
// any = true: return the first feasible plan (a fast feasibility test).
static inline Plan plan(const Vehicle& V, const Disturbance& D, double x0, double y0, double vx0, double vy0, double m0,
                        double tf_lo, double tf_hi, int N = 20, bool soft_x = false, bool any = false, int scan = 8,
                        int* solves = nullptr) {
	std::vector<double> ex, ey;
	auto one = [&](double tf) {
		reference_ext(V, D, vx0, vy0, m0, tf, N, ex, ey);
		if (solves) ++*solves;
		return plan_fixed(V, x0, y0, vx0, vy0, m0, tf, ex, ey, soft_x, any ? 1500 : 3000);
	};
	auto score = [&](const Plan& p) { return p.ok ? p.fuel + (soft_x ? 50.0 * p.miss : 0.0) : 1e30; };
	Plan best;
	double best_tf = -1;
	const double step = (tf_hi - tf_lo) / scan;
	for (int i = 0; i <= scan; ++i) {
		const double tf = tf_lo + step * i;
		Plan p = one(tf);
		if (p.ok && any) return p;
		if (score(p) < score(best)) { best = p; best_tf = tf; }
	}
	if (best_tf < 0 || any) return best;
	double a = std::max(tf_lo, best_tf - step), b = std::min(tf_hi, best_tf + step);
	double c = b - 0.618 * (b - a), d = a + 0.618 * (b - a);
	Plan pc = one(c), pd = one(d);
	for (int it = 0; it < 4; ++it) {
		if (score(pc) < score(best)) best = pc;
		if (score(pd) < score(best)) best = pd;
		if (score(pc) < score(pd)) { b = d; d = c; pd = pc; c = b - 0.618 * (b - a); pc = one(c); }
		else { a = c; c = d; pc = pd; d = a + 0.618 * (b - a); pd = one(d); }
	}
	if (score(pc) < score(best)) best = pc;
	if (score(pd) < score(best)) best = pd;
	return best;
}

// ==================================================================
// 3. Closed-loop guidance
// ==================================================================
enum Phase { STABILIZE = 0, COAST = 1, BURN = 2, TERMINAL = 3 };
static inline const char* phase_name(int p) {
	static const char* n[] = {"stabilize", "coast", "powered descent", "terminal"};
	return n[p];
}

struct Guidance {
	rocket::Params P;
	Vehicle V;
	Disturbance D;
	int N = 20;
	double replan_every = 0.5;   // s (model-predictive: re-plan from the current state twice a second)
	double ignition_lead = 1.0;  // s: look this far ahead when testing an ignition time
	bool ignite_early = true;    // true: ignite as soon as a landing is possible (longest burn, most time to
	                             // divert: safety first). false: at the last moment (least propellant).
	
	int phase = STABILIZE;
	// estimate now (relative to the deck)
	double px = 0, py = 0, vx = 0, vy = 0, th = 0, om = 0, m = 0;
	// filter state at the (delayed) measurement time
	double fx = 0, fy = 0, fvx = 0, fvy = 0;
	std::vector<double> acc_x, acc_y;  // model acceleration applied at each step (engine + drag + gravity)
	int steps = 0;
	// plan
	Plan plan;
	double plan_t0 = 0, last_replan = -1e9, last_check = -1e9;
	double gimbal = 0;
	int ignitions_left = 0;
	// stats
	int solves = 0, plans = 0, failed = 0;
	double t_stab_end = 0, t_ignite = -1;
	bool window_open = false;
	bool verbose = false;
	double kd_dist = 0.015;      // disturbance-estimate gain per step
	double term_tilt0 = 3.0, term_tilt_slope = 0.4;  // terminal tilt limit: 3 deg at the deck, +0.4 deg per m
	double term_kp = 0.15, term_kv = 0.9;
	double terminal_height = 40;
	bool divert = false; double divert_max_t = 15;  // divert burn: tested, hurts (GFOLD.md)
	double phi_slew_deg = 10;     // deg/s
	double fb_p = 0.02, fb_v = 0.25;  // feedback toward the plan between re-plans
	double pad_ax = 0, pad_ay = 0, pad_vx_prev = 0, pad_vy_prev = 0, pad_tau = 0.6, pad_ff = 1.0;  // m: below this, the terminal descent law takes over from the plan
	bool hop = false; int hops = 0;
	double ctrl_tilt_deg = 25;  // controller tilt limit (plans stay within V.max_tilt_deg)
	double att_kp = 6.0, att_kd = 5.0, attitude_lead = 0.7;  // attitude loop (~1.7 rad/s) and look-ahead (s)
	double last_cmd = 0, last_phi = 0, last_ax = 0, last_ay = 0;

	void reset(const rocket::Params& p) {
		*this = Guidance();
		P = p;
		V.dry_mass = P.dry_mass; V.max_thrust = P.max_thrust; V.min_throttle = P.hard ? P.min_throttle : 0.0;
		V.isp = P.isp; V.g0 = P.g0; V.g = P.g;
		V.plan_min = V.min_throttle + 0.02;
		D.drag = P.drag;
	}

	// attitude: gimbal that turns the booster (and so its thrust) toward `phi` rad from vertical
	double attitude(double phi, double thr) const {
		const double I = m * P.length * P.length / 12.0, T = std::max(thr, 0.05) * P.max_thrust;
		const double alpha = att_kp * (phi - th) - att_kd * om;  // PD, well damped
		const double smax = std::sin((double)P.gimbal_max);
		const double s = std::max(-smax, std::min(smax, -alpha * I / (T * P.arm)));
		return std::asin(s);
	}

	// coast prediction (engine off: drag + disturbance + gravity), 0.1 s steps
	void coast(double t, double& x, double& y, double& ux, double& uy) const {
		for (double s = 0; s < t - 1e-9; s += 0.1) {
			const double h = std::min(0.1, t - s);
			double ax, ay; D.accel(ux, uy, m, ax, ay); ay -= P.g;
			ux += ax * h; uy += ay * h; x += ux * h; y += uy * h;
		}
	}

	// could we land if we ignite `lead` s from now? Tries burn times in 2 s steps,
	// nearest first to a constant-deceleration guess, and stops at the first feasible one.
	bool feasible_after(double lead) {
		double x = px, y = py, ux = vx, uy = vy;
		coast(lead, x, y, ux, uy);
		if (y < 5) return false;
		set_u0(0.0);  // ignition: minimum thrust along the current attitude
		const double tf_max = std::min(60.0, (m - V.dry_mass) / (V.plan_min * V.max_thrust / (V.isp * V.g0)));
		const double guess = std::max(3.0, std::min(tf_max, 2.0 * y / std::max(1.0, -uy)));
		std::vector<double> tfs;
		for (double tf = 3; tf <= tf_max; tf += 2) tfs.push_back(tf);
		std::sort(tfs.begin(), tfs.end(), [&](double a, double b) { return std::fabs(a - guess) < std::fabs(b - guess); });
		std::vector<double> ex, ey;
		for (double tf : tfs) {
			reference_ext(V, D, ux, uy, m, tf, N, ex, ey);
			++solves;
			if (plan_fixed(V, x, y, ux, uy, m, tf, ex, ey, false, 1500).ok) return true;
		}
		return false;
	}
	// altitude below which we ignite no matter what: stopping distance at 85% thrust, +30%
	double panic_altitude() const {
		const double a = std::max(0.5, V.plan_max * V.max_thrust / m - V.g);
		return 1.3 * vy * vy / (2 * a) + 50.0;
	}

	// the thrust acceleration the engine gives now (the plan's first step starts near it)
	void set_u0(double thr) {
		const double a = std::max(thr, V.min_throttle) * P.max_thrust / m;
		V.has_u0 = true; V.u0x = a * std::sin(th + gimbal); V.u0y = a * std::cos(th + gimbal);
	}

	void new_plan(double t, bool full_search) {
		++plans;
		Plan p;
		const double tgo = plan.ok ? plan.tf - (t - plan_t0) : 0;
		if (!full_search && plan.ok && tgo > 1.5) {
			std::vector<double> ex, ey;
			const int n = std::max(6, std::min(N, (int)std::ceil(tgo / 0.5)));  // steps >= 0.5 s
			// drag on a straight-line velocity reference (feeding the last plan's own profile
			// back in drifted toward ever-later burns that miss the pad)
			reference_ext(V, D, vx, vy, m, tgo, n, ex, ey);
			++solves;
			// re-plans minimize fuel + a heavy penalty on any miss, so a small disturbance
			// can't make them infeasible sideways (only vertically, then search tf)
			p = plan_fixed(V, px, py, vx, vy, m, tgo, ex, ey, true);
			if (verbose) {
				std::vector<double> rx, ry; reference_ext(V, D, vx, vy, m, tgo, n, rx, ry);
				Plan q = plan_fixed(V, px, py, vx, vy, m, tgo, rx, ry, true);
				double se = 0, sr = 0; for (int k = 0; k < n; ++k) { se += ex[k]; sr += rx[k]; }
				printf("    debug: n=%d sum ex=%.2f (ref %.2f) ex0=%.2f,%.2f ref0=%.2f,%.2f | ref-ext plan miss=%.1f fuel=%.0f\n", n, se, sr, ex[0], ey[0], rx[0], ry[0], q.miss, q.fuel);
			}
			if (!p.ok) p = gfold::plan(V, D, px, py, vx, vy, m, std::max(2.0, tgo - 4), tgo + 8, N, true, false, 6, &solves);
		} else {
			p = gfold::plan(V, D, px, py, vx, vy, m, 4, 60, N, false, false, 8, &solves);
		}
		if (!p.ok) {  // no exact landing possible: minimize the miss
			const double lo = plan.ok ? std::max(2.0, tgo - 4) : 4, hi = plan.ok ? tgo + 10 : 60;
			p = gfold::plan(V, D, px, py, vx, vy, m, lo, hi, N, true, false, 6, &solves);
		}
		if (verbose)
			printf("  plan t=%.2f from h=%.1f x=%.1f vx=%.2f vy=%.2f m=%.0f u0=(%.2f,%.2f) tgo_old=%.2f -> %s tf=%.2f status=%d fuel=%.0f miss=%.1f\n",
			       t, py, px, vx, vy, m, V.u0x, V.u0y, tgo, p.ok ? "ok" : "FAILED", p.tf, p.status, p.fuel, p.miss);
		if (p.ok) { plan = p; plan_t0 = t; }
		else ++failed;
		last_replan = t;
	}

	// One control step. `in` = rocket::make_input(...) with hist = 0: exactly what the
	// network sees. t = flight time (the flight computer's clock).
	void act(const float* in, double t, float& throttle_out, float& gimbal_out) {
		// --- sensors ---
		const double mx = in[0] * 500.0, my = in[1] * 1000.0, mvx = in[2] * 50.0, mvy = in[3] * 50.0;
		const double mth = std::atan2(in[4], in[5]), mom = in[6];
		const double fuel = std::max(0.0, (double)in[7] * P.fuel_ref);
		const double thr = P.hard ? in[9] : last_cmd;  // engine state is not delayed (easy mode: throttle is instant)
		D.pad_vx = P.hard ? in[8] * 10.0 : 0.0;
		D.pad_vy = P.real ? in[11] * 2.0 : 0.0;
		// deck acceleration: filtered derivative of the deck-velocity sensors (the deck sways and
		// heaves; everything here is relative to it, so its acceleration acts like a force)
		if (steps > 0) {
			const double k = std::min(1.0, P.dt / pad_tau);
			pad_ax += k * ((D.pad_vx - pad_vx_prev) / P.dt - pad_ax);
			pad_ay += k * ((D.pad_vy - pad_vy_prev) / P.dt - pad_ay);
		}
		pad_vx_prev = D.pad_vx; pad_vy_prev = D.pad_vy;
		const double fpx = pad_ff * pad_ax, fpy = pad_ff * pad_ay;
		ignitions_left = P.real && P.max_ignitions > 0 ? (int)std::lround(in[10] * P.max_ignitions) : 99;
		m = P.dry_mass + fuel;
		const int Dl = P.delay;
		const double dt = P.dt;
		// --- estimator: position/velocity filter with a disturbance state, at the measurement
		// time (Dl steps ago), then pushed forward to now through the known model accelerations
		const int k_meas = steps - Dl;  // the measurement describes step k_meas
		if (k_meas <= 0) { fx = mx; fy = my; fvx = mvx; fvy = mvy; }
		else {
			const double ax = acc_x[k_meas - 1] + D.dx, ay = acc_y[k_meas - 1] + D.dy;
			fvx += ax * dt; fvy += ay * dt; fx += fvx * dt; fy += fvy * dt;  // semi-implicit, like rocket.cuh
			const double kp = 0.08, kv = 0.1, kd = kd_dist;
			const double evx = mvx - fvx, evy = mvy - fvy;
			fx += kp * (mx - fx); fy += kp * (my - fy);
			fvx += kv * evx; fvy += kv * evy;
			if (k_meas > 10) { D.dx += kd * evx; D.dy += kd * evy; }
			D.dx = std::max(-2.0, std::min(2.0, D.dx)); D.dy = std::max(-2.0, std::min(2.0, D.dy));
		}
		px = fx; py = fy; vx = fvx; vy = fvy;
		for (int k = std::max(0, k_meas); k < steps; ++k) {
			vx += (acc_x[k] + D.dx) * dt; vy += (acc_y[k] + D.dy) * dt; px += vx * dt; py += vy * dt;
		}
		th = mth + mom * Dl * dt; om = mom;

		// --- phase logic ---
		double thr_cmd = 0, phi = 0;
		if (phase == STABILIZE) {
			// Divert burn (first engine start, at minimum throttle): stop the tumble and set the
			// sideways velocity so the coast ends above the pad. The booster turns slowly, so big
			// sideways corrections are much easier up here than in the short landing burn.
			const double t_coast = std::max(6.0, (py - 500.0) / std::max(30.0, -vy)) + 6.0;  // until ~mid landing burn
			const double vx_target = -px / t_coast;
			const double ex = vx_target - vx;
			const bool can = divert && P.hard && ignitions_left >= 2;
			if (steps == 0 && !can) phase = COAST;
			else if (t > 2.0 && std::fabs(ex) < 0.8 && std::fabs(th) < 0.03 && std::fabs(om) < 0.01) phase = COAST;
			else if (t > divert_max_t || py < panic_altitude() + 300) phase = COAST;
			if (phase == COAST) t_stab_end = t;
			else {
				thr_cmd = V.min_throttle;
				// lean into the needed sideways change, but come back upright as it's reached
				phi = std::max(-0.35, std::min(0.35, std::atan2(ex / 3.0, V.g)));
				if (std::fabs(ex) < 0.8) phi = 0.0;
			}
		}
		if (phase == COAST && t - last_check >= 0.5) {
			// Landing is possible only for ignition times inside a window: too early and even
			// minimum thrust stops the booster too high; too late and full thrust can't stop it.
			// Wait for the window to open, then ignite at its last moment.
			last_check = t;
			const bool ok_later = feasible_after(ignition_lead);
			if (ok_later) window_open = true;
			const bool go = ignite_early ? ok_later : (window_open && !ok_later);
			if (go || py < panic_altitude()) { phase = BURN; t_ignite = t; set_u0(0.0); new_plan(t, true); }
		}
		if (phase == BURN) {
			const double tgo = plan.ok ? plan.tf - (t - plan_t0) : 0;
			if (t - last_replan >= replan_every - 1e-9 && py > 15 && (tgo > 2.5 || !plan.ok)) { set_u0(thr); new_plan(t, !plan.ok); }
			const double tgo2 = plan.ok ? plan.tf - (t - plan_t0) : 0;
			if ((plan.ok && tgo2 < 1.5) || py < terminal_height) phase = TERMINAL;
		}
		double ax = 0, ay = 0, ax_dir = 1e30;
		if (phase == BURN && !plan.ok) {
			// no plan (yet): fly a stopping-distance profile until one is found
			double dax, day; D.accel(vx, vy, m, dax, day);
			const double vyt = -std::sqrt(1.0 + 2.0 * 0.5 * (V.plan_max * V.max_thrust / m - V.g) * std::max(0.0, py));
			ax = -0.02 * px - 0.3 * vx - dax;
			ay = P.g + 0.8 * (vyt - vy) - day;
		} else if (phase == BURN) {
			// follow the plan: planned thrust + feedback on its position and velocity
			double rx, ry, rvx, rvy, uxp, uyp;
			plan.at(t - plan_t0, rx, ry, rvx, rvy, uxp, uyp, V.g);
			ax = uxp + fb_p * (rx - px) + fb_v * (rvx - vx) + fpx;
			ay = uyp + fb_p * (ry - py) + fb_v * (rvy - vy) + fpy;
			// the booster takes ~1 s to rotate: steer toward where the plan's thrust points a bit ahead
			double r2x, r2y, v2x, v2y, u2x, u2y;
			plan.at(t - plan_t0 + attitude_lead, r2x, r2y, v2x, v2y, u2x, u2y, V.g);
			ax_dir = u2x + fb_p * (rx - px) + fb_v * (rvx - vx) + fpx;
		} else if (phase == TERMINAL) {
			// final seconds: follow a descent-speed profile v(h) = -sqrt(v_td^2 + 2 a h) down to the
			// deck, null the sideways error, cancel known forces. With a 40% minimum throttle a light
			// booster can't hover, so the profile's deceleration a must be one minimum thrust can fly.
			double dax, day; D.accel(vx, vy, m, dax, day);
			const double h = std::max(0.0, py);
			const double a_min = (P.hard ? V.min_throttle * P.max_thrust / m : 0.0) - V.g + day;  // net decel at min thrust
			const double a_d = std::max(1.0, 1.3 * a_min + 0.3);
			const double vtd = std::fabs(V.vy_final);
			const double vyt = -std::sqrt(vtd * vtd + 2.0 * a_d * h);
			ax = -term_kp * px - term_kv * vx - dax - (1.0 - D.use) * D.dx + fpx;  // full disturbance estimate here
			ay = P.g + 1.5 * (vyt - vy) - day + fpy;
			// too slow above the deck at minimum throttle (it would float back up): cut the engine,
			// drop onto the profile, relight (needs an engine start left)
			if (!hop && P.hard && ignitions_left >= 1 && thr <= V.min_throttle + 0.02 && h > 2.5 && h < 15.0 && vy > vyt + 1.0 && a_min > 0.3) {
				hop = true; ++hops;
			}
			if (hop && (vy <= vyt || h < 2.0)) hop = false;  // relight
		}
		if (phase == TERMINAL && hop) {
			thr_cmd = 0.0; phi = 0.0;
		} else if (phase == BURN || phase == TERMINAL) {
			ay = std::max(ay, 0.1);
			const double h = std::max(0.0, py);
			double tilt_max = ctrl_tilt_deg * M_PI / 180;  // the controller may tilt past the plan's limit to correct
			if (phase == TERMINAL) tilt_max = std::min(tilt_max, (term_tilt0 + term_tilt_slope * h) * M_PI / 180);
			if (ax_dir > 1e29) ax_dir = ax;
			phi = std::max(-tilt_max, std::min(tilt_max, std::atan2(ax_dir, ay)));
			// slew limit on the commanded direction, matched to the plans' thrust-rate limit:
			// commands that jump faster than the booster can turn start a lateral limit cycle
			const double slew = phi_slew_deg * M_PI / 180 * P.dt;
			phi = std::max(last_phi - slew, std::min(last_phi + slew, phi));
			// throttle: the commanded acceleration, but at least enough for its vertical part
			// along the direction the thrust actually points
			// thrust magnitude: enough for the vertical part along the (clipped) direction
			// and along the direction the thrust actually points now
			const double c = std::max(0.5, std::min(std::cos(phi), std::cos(th + gimbal)));
			const double a = std::min(std::sqrt(ax * ax + ay * ay), ay / std::cos(phi)) * 0.0 + ay / c;
			thr_cmd = a * m / P.max_thrust;
			thr_cmd += 0.5 * (thr_cmd - thr);  // lead, against the 0.3 s engine lag
			thr_cmd = std::max(std::max(0.02, V.min_throttle), std::min(1.0, thr_cmd));
		}
		last_phi = phi; last_ax = ax; last_ay = ay;
		gimbal = thr_cmd > 0 ? attitude(phi, std::max(thr, thr_cmd)) : 0.0;  // engine off: no steering
		// model acceleration this command produces (for the estimator)
		double thr_next = thr_cmd;
		if (P.hard) {
			if (thr_cmd == 0) thr_next = 0;
			else {
				thr_next = (thr == 0 ? 0.0 : thr) + (std::max(thr_cmd, V.min_throttle) - thr) * std::min(1.0, dt / P.engine_tau);
				thr_next = std::max(thr_next, V.min_throttle);
			}
		}
		const double T = fuel > 0 ? thr_next * P.max_thrust : 0.0;
		double dax, day; D.drag_accel(vx, vy, m, dax, day);
		acc_x.push_back(T * std::sin(th + gimbal) / m + dax - pad_ax);
		acc_y.push_back(T * std::cos(th + gimbal) / m + day - P.g - pad_ay);
		++steps;
		last_cmd = thr_cmd;
		throttle_out = (float)thr_cmd;
		gimbal_out = (float)gimbal;
	}
};

}  // namespace gfold
