// es.cuh
//
// Evolution Strategies (Salimans et al. 2017, "OpenAI-ES") on the GPU.
//
// Instead of a population of independent networks (gpu_evolve.cuh), ES keeps ONE
// set of weights theta and estimates which direction improves it:
//   1. sample `pairs` noise vectors eps_p; evaluate theta + sigma*eps_p and
//      theta - sigma*eps_p (mirrored sampling cancels much of the noise)
//   2. replace costs by centered ranks in [-0.5, 0.5] (fitness shaping: robust
//      to outliers and to the scale of the cost)
//   3. gradient estimate g = sum_p (u+_p - u-_p) eps_p / (2 * pairs * sigma)
//   4. Adam step on theta, with L2 weight decay
// The noise is regenerated from a counter-based hash (seed, generation, pair,
// weight), so it is never stored: memory is O(weights), not O(pop x weights)
// for the noise.
// --------------------

#pragma once

#include "gpu_nn.cuh"
#include "gpu_evolve.cuh"  // init_kernel
#include <numeric>

namespace gnn {

struct ESConfig {
	int pairs = 512;          // population = 2 * pairs
	float sigma = 0.02f;      // perturbation std
	float lr = 0.01f;         // Adam learning rate
	float lr_end = 0.0f;      // if > 0, decay lr geometrically to this by the last generation
	float weight_decay = 0.005f;
	float init_gain = 1.0f;
	float final_gain = 1.0f;  // extra scale for the output layer's initial weights
	uint32_t seed = 1;
};

__host__ __device__ inline float es_eps(uint32_t seed, uint32_t gen, int pair, int j) {
	const uint32_t h = hash4(seed ^ 0xE5E5E5E5u, gen, (uint32_t)pair, (uint32_t)j);
	return normal(h, hash32(h ^ 0x5bd1e995u));
}

// pop[2p] = theta + sigma*eps_p, pop[2p+1] = theta - sigma*eps_p
__global__ void es_population_kernel(const float* __restrict__ theta, float* __restrict__ pop, int pairs,
                                     int n_params, float sigma, uint32_t seed, uint32_t gen) {
	const size_t total = (size_t)2 * pairs * n_params;
	for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
		const int member = (int)(i / n_params), j = (int)(i % n_params);
		const float e = sigma * es_eps(seed, gen, member >> 1, j);
		pop[i] = theta[j] + ((member & 1) ? -e : e);
	}
}

// grad[j] = sum_p udiff[p] * eps_p[j] / (2 * pairs * sigma)   (direction that INCREASES utility)
__global__ void es_grad_kernel(const float* __restrict__ udiff, int pairs, int n_params, float sigma,
                               uint32_t seed, uint32_t gen, float* __restrict__ grad) {
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_params; j += gridDim.x * blockDim.x) {
		float g = 0.0f;
		for (int p = 0; p < pairs; ++p) g = fmaf(udiff[p], es_eps(seed, gen, p, j), g);
		grad[j] = g / (2.0f * pairs * sigma);
	}
}

// Adam ascent on utility, with decoupled-style L2 decay folded into the gradient
__global__ void adam_kernel(float* theta, const float* __restrict__ grad, float* m, float* v, int n,
                            float lr, float wd, int t) {
	const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
	const float c1 = 1.0f - powf(b1, (float)t), c2 = 1.0f - powf(b2, (float)t);
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
		const float g = grad[j] - wd * theta[j];
		m[j] = b1 * m[j] + (1.0f - b1) * g;
		v[j] = b2 * v[j] + (1.0f - b2) * g * g;
		theta[j] += lr * (m[j] / c1) / (sqrtf(v[j] / c2) + eps);
	}
}

class EvolutionStrategy {
public:
	Topology topo;
	ESConfig cfg;
	float *d_theta = nullptr, *d_pop = nullptr, *d_grad = nullptr, *d_m = nullptr, *d_v = nullptr, *d_udiff = nullptr;
	int gen = 0;

	EvolutionStrategy(const Topology& t, const ESConfig& c) : topo(t), cfg(c) {
		const size_t P = topo.n_params;
		CUDA_CHECK(cudaMalloc(&d_theta, P * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&d_pop, (size_t)2 * cfg.pairs * P * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&d_grad, P * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&d_m, P * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&d_v, P * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&d_udiff, cfg.pairs * sizeof(float)));
		CUDA_CHECK(cudaMemset(d_m, 0, P * sizeof(float)));
		CUDA_CHECK(cudaMemset(d_v, 0, P * sizeof(float)));
		init_kernel<<<256, 256>>>(topo, d_theta, 1, cfg.init_gain, cfg.seed, cfg.final_gain);
		CUDA_CHECK(cudaGetLastError());
	}
	~EvolutionStrategy() { cudaFree(d_theta); cudaFree(d_pop); cudaFree(d_grad); cudaFree(d_m); cudaFree(d_v); cudaFree(d_udiff); }
	EvolutionStrategy(const EvolutionStrategy&) = delete;
	EvolutionStrategy& operator=(const EvolutionStrategy&) = delete;

	int pop_size() const { return 2 * cfg.pairs; }

	// fill d_pop with the mirrored perturbations of theta for this generation
	void populate() {
		es_population_kernel<<<2048, 256>>>(d_theta, d_pop, cfg.pairs, topo.n_params, cfg.sigma, cfg.seed, (uint32_t)gen);
		CUDA_CHECK(cudaGetLastError());
	}

	// cost[i] for population member i (lower is better); progress = 0..1 for lr decay
	void update(const std::vector<float>& cost, float progress) {
		const int n = pop_size();
		std::vector<int> order(n);
		std::iota(order.begin(), order.end(), 0);
		std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cost[a] < cost[b]; });
		// centered ranks, best = +0.5; tied costs share their average rank, so a
		// flat region (every member scores the same) gives a zero update, not noise
		std::vector<float> u(n);
		for (int r0 = 0; r0 < n;) {
			int r1 = r0;
			while (r1 + 1 < n && cost[order[r1 + 1]] == cost[order[r0]]) ++r1;
			const float avg_rank = 0.5f * (r0 + r1);
			for (int r = r0; r <= r1; ++r) u[order[r]] = (n - 1 - avg_rank) / (n - 1) - 0.5f;
			r0 = r1 + 1;
		}
		std::vector<float> udiff(cfg.pairs);
		for (int p = 0; p < cfg.pairs; ++p) udiff[p] = u[2 * p] - u[2 * p + 1];
		CUDA_CHECK(cudaMemcpy(d_udiff, udiff.data(), cfg.pairs * sizeof(float), cudaMemcpyHostToDevice));
		const int blocks = (topo.n_params + 255) / 256;
		es_grad_kernel<<<blocks, 256>>>(d_udiff, cfg.pairs, topo.n_params, cfg.sigma, cfg.seed, (uint32_t)gen, d_grad);
		const float lr = cfg.lr_end > 0 ? cfg.lr * powf(cfg.lr_end / cfg.lr, progress) : cfg.lr;
		adam_kernel<<<blocks, 256>>>(d_theta, d_grad, d_m, d_v, topo.n_params, lr, cfg.weight_decay, gen + 1);
		CUDA_CHECK(cudaGetLastError());
		++gen;
	}
};

}  // namespace gnn
