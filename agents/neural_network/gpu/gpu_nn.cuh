// gpu_nn.cuh
//
// GPU Neural Network (v4)
// Fully connected tanh networks, the same math as Neural Network v3.0
// (agents/neural_network/network/neural_network.h), rewritten so a whole
// POPULATION of networks lives on the GPU and runs in parallel.
//
// Parameter layout for one network, layer by layer:
//   W[in][out]  (row k = weights from input k to every output)  then  b[out]
// v3's bias node becomes the bias vector b; v3's unused weights into the next
// layer's bias node are dropped. Every layer applies tanh, like v3.
//
// Two ways to run a population:
//   1. forward_population(): dataset mode. Each layer is a batched, tiled
//      matrix multiply over (samples x outputs x networks), so big networks
//      and big batches use the whole GPU.
//   2. forward_block(): simulation mode. One thread block runs one network
//      step by step inside an episode kernel (see rocket/), with the
//      neurons of each layer computed in parallel by the block's threads.
// forward_serial() is the same math on one thread, usable on the CPU too.
// --------------------

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
	fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

namespace gnn {

constexpr int MAX_LAYERS = 8;

// ------------------------------------------------------------------
// Topology: layer sizes and where each layer's parameters start
// ------------------------------------------------------------------
struct Topology {
	int n_layers;                // weight layers = number of sizes - 1
	int size[MAX_LAYERS + 1];    // size[0] = inputs, size[n_layers] = outputs
	int w_off[MAX_LAYERS];       // offset of W for layer l
	int b_off[MAX_LAYERS];       // offset of b for layer l
	int n_params;                // floats per network
	int max_width;               // widest layer

	__host__ __device__ int inputs()  const { return size[0]; }
	__host__ __device__ int outputs() const { return size[n_layers]; }
};

inline Topology make_topology(const std::vector<int>& sizes) {
	if (sizes.size() < 2 || sizes.size() > MAX_LAYERS + 1) {
		fprintf(stderr, "topology needs 2..%d layer sizes\n", MAX_LAYERS + 1); exit(1);
	}
	Topology t{};
	t.n_layers = (int)sizes.size() - 1;
	int off = 0;
	t.max_width = 0;
	for (int i = 0; i < (int)sizes.size(); ++i) { t.size[i] = sizes[i]; t.max_width = std::max(t.max_width, sizes[i]); }
	for (int l = 0; l < t.n_layers; ++l) {
		t.w_off[l] = off; off += sizes[l] * sizes[l + 1];
		t.b_off[l] = off; off += sizes[l + 1];
	}
	t.n_params = off;
	return t;
}

inline std::string describe(const Topology& t) {
	std::string s;
	for (int i = 0; i <= t.n_layers; ++i) s += (i ? "-" : "") + std::to_string(t.size[i]);
	return s;
}

// ------------------------------------------------------------------
// Deterministic counter-based random numbers (same on CPU and GPU)
// ------------------------------------------------------------------
__host__ __device__ inline uint32_t hash32(uint32_t x) {
	x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
	return x;
}
__host__ __device__ inline uint32_t hash4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
	return hash32(a ^ hash32(b ^ hash32(c ^ hash32(d + 0x9e3779b9U))));
}
// uniform in (0, 1]
__host__ __device__ inline float uniform01(uint32_t h) { return ((h >> 8) + 1) * (1.0f / 16777216.0f); }
__host__ __device__ inline float normal(uint32_t h1, uint32_t h2) {
	float u1 = uniform01(h1), u2 = uniform01(h2);
	return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

// ------------------------------------------------------------------
// Single-thread forward pass (host or device)
//   in:   inputs[size[0]]
//   out:  outputs[size[n_layers]]
//   bufA, bufB: scratch of max_width floats each
// ------------------------------------------------------------------
__host__ __device__ inline void forward_serial(const Topology& t, const float* p, const float* in,
                                               float* out, float* bufA, float* bufB) {
	const float* x = in;
	float* y = bufA;
	for (int l = 0; l < t.n_layers; ++l) {
		const int ni = t.size[l], no = t.size[l + 1];
		const float* W = p + t.w_off[l];
		const float* b = p + t.b_off[l];
		float* dst = (l == t.n_layers - 1) ? out : y;
		for (int j = 0; j < no; ++j) {
			float s = b[j];
			for (int k = 0; k < ni; ++k) s += x[k] * W[k * no + j];
			dst[j] = tanhf(s);
		}
		x = dst;
		y = (y == bufA) ? bufB : bufA;
	}
}

// ------------------------------------------------------------------
// Block-cooperative forward pass (device, inside an episode kernel)
//   All threads of the block must call it. p may point to shared or global
//   memory. in, a, b are shared buffers (a, b: max_width floats).
//   Returns a pointer (a or b) holding the outputs. Ends with __syncthreads().
// ------------------------------------------------------------------
__device__ inline const float* forward_block(const Topology& t, const float* __restrict__ p,
                                             const float* in, float* a, float* b) {
	const float* x = in;
	float* y = a;
	for (int l = 0; l < t.n_layers; ++l) {
		const int ni = t.size[l], no = t.size[l + 1];
		const float* W = p + t.w_off[l];
		const float* bias = p + t.b_off[l];
		for (int j = threadIdx.x; j < no; j += blockDim.x) {
			float s = bias[j];
			// W[k*no + j]: consecutive threads read consecutive addresses
			for (int k = 0; k < ni; ++k) s = fmaf(x[k], W[k * no + j], s);
			y[j] = tanhf(s);
		}
		__syncthreads();
		x = y;
		y = (y == a) ? b : a;
	}
	return x;
}

// ------------------------------------------------------------------
// Dataset mode: one layer for every network at once (tiled matrix multiply)
//   X: inputs. Shared by all networks (x_net_stride = 0) or per network.
//   Y[net][n][out] = tanh(X[net][n][in] * W[net] + b[net])
//   grid: (ceil(out/T), ceil(n/T), pop), block: (T, T)
// ------------------------------------------------------------------
constexpr int TILE = 16;

__global__ void layer_kernel(const float* __restrict__ params, int n_params, int w_off, int b_off,
                             int in, int out, const float* __restrict__ X, size_t x_net_stride,
                             float* __restrict__ Y, int n) {
	__shared__ float Xs[TILE][TILE];
	__shared__ float Ws[TILE][TILE + 1];
	const int net = blockIdx.z;
	const int row = blockIdx.y * TILE + threadIdx.y;   // sample
	const int col = blockIdx.x * TILE + threadIdx.x;   // output neuron
	const float* W = params + (size_t)net * n_params + w_off;
	const float* Xn = X + (size_t)net * x_net_stride;
	float s = 0.0f;
	for (int k0 = 0; k0 < in; k0 += TILE) {
		Xs[threadIdx.y][threadIdx.x] = (row < n && k0 + threadIdx.x < in) ? Xn[(size_t)row * in + k0 + threadIdx.x] : 0.0f;
		Ws[threadIdx.y][threadIdx.x] = (k0 + threadIdx.y < in && col < out) ? W[(size_t)(k0 + threadIdx.y) * out + col] : 0.0f;
		__syncthreads();
		#pragma unroll
		for (int k = 0; k < TILE; ++k) s = fmaf(Xs[threadIdx.y][k], Ws[k][threadIdx.x], s);
		__syncthreads();
	}
	if (row < n && col < out) {
		const float bias = params[(size_t)net * n_params + b_off + col];
		Y[((size_t)net * n + row) * out + col] = tanhf(s + bias);
	}
}

// Runs every network in d_params on the same n samples d_X[n][inputs].
// d_bufA, d_bufB: pop * n * max_width floats each. Returns the buffer holding
// outputs [pop][n][outputs].
inline float* forward_population(const Topology& t, const float* d_params, int pop,
                                 const float* d_X, int n, float* d_bufA, float* d_bufB,
                                 cudaStream_t stream = 0) {
	const float* x = d_X;
	size_t x_stride = 0;  // first layer: inputs shared by every network
	float* y = d_bufA;
	for (int l = 0; l < t.n_layers; ++l) {
		const int in = t.size[l], out = t.size[l + 1];
		dim3 block(TILE, TILE);
		dim3 grid((out + TILE - 1) / TILE, (n + TILE - 1) / TILE, pop);
		layer_kernel<<<grid, block, 0, stream>>>(d_params, t.n_params, t.w_off[l], t.b_off[l],
		                                         in, out, x, x_stride, y, n);
		x = y;
		x_stride = (size_t)n * out;
		y = (y == d_bufA) ? d_bufB : d_bufA;
	}
	CUDA_CHECK(cudaGetLastError());
	return (float*)x;
}

}  // namespace gnn
