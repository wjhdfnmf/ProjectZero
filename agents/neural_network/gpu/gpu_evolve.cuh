// gpu_evolve.cuh
//
// Population of GPU networks + the evolutionary loop's GPU side.
// All weights stay on the GPU. Each generation the CPU only receives one cost
// per network, picks parents, and sends back one parent index per child; the
// GPU copies and mutates every child's weights in parallel.
//
// Selection: the `elites` lowest-cost networks survive unchanged; every other
// child is a mutated copy of a random parent from the best `parent_frac`
// (truncation selection). Mutation: each weight, with probability `chance`,
// gets Gaussian noise with standard deviation `sigma`.
// --------------------

#pragma once

#include "gpu_nn.cuh"
#include <numeric>
#include <random>

namespace gnn {

struct EvoConfig {
	int pop = 1024;
	int elites = 8;
	float parent_frac = 0.1f;
	float sigma = 0.02f;
	float chance = 1.0f;
	float init_gain = 1.0f;   // initial weights: U(+-gain*sqrt(6/(in+out))), biases 0
	float final_gain = 1.0f;  // extra scale for the output layer (small = start with outputs near 0)
	uint32_t seed = 1;
};

// initial weights (Xavier uniform scaled by gain, output layer also by final_gain; biases 0)
__global__ void init_kernel(Topology t, float* params, int pop, float gain, uint32_t seed, float final_gain = 1.0f) {
	const size_t total = (size_t)pop * t.n_params;
	for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
		const int net = (int)(i / t.n_params), j = (int)(i % t.n_params);
		float v = 0.0f;
		for (int l = 0; l < t.n_layers; ++l) {
			if (j >= t.w_off[l] && j < t.b_off[l]) {
				const float lim = gain * sqrtf(6.0f / (t.size[l] + t.size[l + 1])) * (l == t.n_layers - 1 ? final_gain : 1.0f);
				v = (2.0f * uniform01(hash4(seed, 0xA11CEu, net, j)) - 1.0f) * lim;
			}
		}
		params[i] = v;
	}
}

// child c = parent[c] (+ mutation unless c < elites)
__global__ void reproduce_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                 const int* __restrict__ parent, int pop, int n_params, int elites,
                                 float sigma, float chance, uint32_t seed, uint32_t gen) {
	const size_t total = (size_t)pop * n_params;
	for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
		const int c = (int)(i / n_params), j = (int)(i % n_params);
		float v = src[(size_t)parent[c] * n_params + j];
		if (c >= elites) {
			const uint32_t h = hash4(seed, gen, c, j);
			if (uniform01(h) <= chance) v += sigma * normal(hash32(h ^ 0x1234567u), hash32(h ^ 0x89abcdeu));
		}
		dst[i] = v;
	}
}

class Population {
public:
	Topology topo;
	EvoConfig cfg;
	float* d_params = nullptr;   // [pop][n_params]
	float* d_next = nullptr;
	int* d_parent = nullptr;
	std::mt19937 rng;
	int gen = 0;

	Population(const Topology& t, const EvoConfig& c) : topo(t), cfg(c), rng(c.seed) {
		const size_t bytes = (size_t)cfg.pop * topo.n_params * sizeof(float);
		CUDA_CHECK(cudaMalloc(&d_params, bytes));
		CUDA_CHECK(cudaMalloc(&d_next, bytes));
		CUDA_CHECK(cudaMalloc(&d_parent, cfg.pop * sizeof(int)));
		init_kernel<<<1024, 256>>>(topo, d_params, cfg.pop, cfg.init_gain, cfg.seed, cfg.final_gain);
		CUDA_CHECK(cudaGetLastError());
	}
	~Population() { cudaFree(d_params); cudaFree(d_next); cudaFree(d_parent); }
	Population(const Population&) = delete;
	Population& operator=(const Population&) = delete;

	const float* params(int i) const { return d_params + (size_t)i * topo.n_params; }

	// cost[i] for network i (lower is better). Returns index order, best first.
	std::vector<int> next_generation(const std::vector<float>& cost, float sigma) {
		std::vector<int> order(cfg.pop);
		std::iota(order.begin(), order.end(), 0);
		std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cost[a] < cost[b]; });
		const int n_parents = std::max(1, (int)(cfg.parent_frac * cfg.pop));
		std::uniform_int_distribution<int> pick(0, n_parents - 1);
		std::vector<int> parent(cfg.pop);
		for (int c = 0; c < cfg.pop; ++c) parent[c] = (c < cfg.elites) ? order[c] : order[pick(rng)];
		CUDA_CHECK(cudaMemcpy(d_parent, parent.data(), cfg.pop * sizeof(int), cudaMemcpyHostToDevice));
		reproduce_kernel<<<2048, 256>>>(d_params, d_next, d_parent, cfg.pop, topo.n_params, cfg.elites,
		                                sigma, cfg.chance, cfg.seed, (uint32_t)gen);
		CUDA_CHECK(cudaGetLastError());
		std::swap(d_params, d_next);
		++gen;
		return order;
	}

	std::vector<float> download(int i) const {
		std::vector<float> w(topo.n_params);
		CUDA_CHECK(cudaMemcpy(w.data(), params(i), topo.n_params * sizeof(float), cudaMemcpyDeviceToHost));
		return w;
	}
};

}  // namespace gnn
