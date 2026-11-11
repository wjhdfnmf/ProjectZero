// verify_bench.cu
//
// 1. VERIFY: networks built with Neural Network v3.0 (neural_network.h, CPU,
//    double precision) give the same outputs on the GPU after converting their
//    weights to the v4 layout.
// 2. BENCHMARK: forward passes per second for
//      v3 on the CPU (one thread)  vs  v4 on the CPU (one thread)
//      vs  v4 on the GPU, dataset mode (batched matrix multiply)
//      vs  v4 on the GPU, simulation mode (one block per network episode)
//
// usage: verify_bench.exe            writes outputs/bench.csv
// --------------------

#include "gpu_nn.cuh"
#include "../network/neural_network.h"

#include <chrono>
#include <fstream>
#include <random>

using namespace gnn;
using Clock = std::chrono::steady_clock;

static double seconds_since(Clock::time_point t0) {
	return std::chrono::duration<double>(Clock::now() - t0).count();
}

// v3 weights -> v4 parameter vector
std::vector<float> convert_v3(Network::Network& net, const Topology& t) {
	auto ew = net.export_weights();  // [layer][node][weight to next-layer node]
	std::vector<float> p(t.n_params);
	for (int l = 0; l < t.n_layers; ++l) {
		const int in = t.size[l], out = t.size[l + 1];
		for (int k = 0; k < in; ++k)
			for (int j = 0; j < out; ++j) p[t.w_off[l] + k * out + j] = (float)ew[l][k][j];
		for (int j = 0; j < out; ++j) p[t.b_off[l] + j] = (float)ew[l][in][j];  // v3 bias node is the last node
	}
	return p;
}

Network::Network make_v3(const std::vector<int>& sizes) {
	std::vector<unsigned int> npl(sizes.begin(), sizes.end());
	double mod = 0.2, chance = 0.1;
	Network::Network net;
	net.run_type = 1;
	net.setup(npl, mod, chance);
	return net;
}

// simulation-mode benchmark: each block runs one network for `steps` steps,
// feeding its outputs back as the next inputs (like a controller in a loop)
// (weights in shared memory when they fit, otherwise read from global memory)
__global__ void episode_bench_kernel(Topology t, const float* params, bool shared_w, int steps, float* sink) {
	extern __shared__ float sh[];
	float* w = sh;
	float* in = shared_w ? w + t.n_params : sh;
	float* a = in + t.max_width;
	float* b = a + t.max_width;
	const float* gp = params + (size_t)blockIdx.x * t.n_params;
	if (shared_w) for (int i = threadIdx.x; i < t.n_params; i += blockDim.x) w[i] = gp[i];
	const float* p = shared_w ? w : gp;
	for (int i = threadIdx.x; i < t.size[0]; i += blockDim.x) in[i] = 0.1f * (i + 1);
	__syncthreads();
	float acc = 0.0f;
	for (int s = 0; s < steps; ++s) {
		const float* y = forward_block(t, p, in, a, b);
		if (threadIdx.x == 0) {
			for (int i = 0; i < t.size[0]; ++i) in[i] = 0.5f * y[i % t.outputs()] + 0.01f * i;
			acc += y[0];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) sink[blockIdx.x] = acc;
}

int main() {
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	printf("GPU: %s, %d SMs\n\n", prop.name, prop.multiProcessorCount);

	const std::vector<std::vector<int>> topologies = {
		{1, 8, 1}, {3, 4, 2}, {1, 8, 8, 1}, {8, 64, 64, 2}, {8, 256, 256, 2}, {64, 512, 512, 512, 8},
	};

	// ---------------- 1. verify against v3 ----------------
	printf("== Verify: v3 (CPU, double) vs v4 (GPU, float) ==\n");
	srand(12345);
	std::mt19937 rng(7);
	std::uniform_real_distribution<double> U(-1.0, 1.0);
	bool all_ok = true;
	for (const auto& sizes : topologies) {
		if (sizes.back() > 64) continue;
		Topology t = make_topology(sizes);
		Network::Network v3 = make_v3(sizes);
		std::vector<float> p = convert_v3(v3, t);
		const int n = 1000;
		std::vector<float> X((size_t)n * t.inputs());
		for (auto& x : X) x = (float)U(rng);
		// v3 reference
		std::vector<double> ref((size_t)n * t.outputs());
		for (int s = 0; s < n; ++s) {
			std::vector<double> in(X.begin() + (size_t)s * t.inputs(), X.begin() + (size_t)(s + 1) * t.inputs());
			std::vector<double> o = v3.cycle(in);
			for (int j = 0; j < t.outputs(); ++j) ref[(size_t)s * t.outputs() + j] = o[j];
		}
		// GPU dataset mode
		float *dP, *dX, *dA, *dB;
		CUDA_CHECK(cudaMalloc(&dP, p.size() * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&dX, X.size() * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&dA, (size_t)n * t.max_width * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&dB, (size_t)n * t.max_width * sizeof(float)));
		CUDA_CHECK(cudaMemcpy(dP, p.data(), p.size() * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
		float* dY = forward_population(t, dP, 1, dX, n, dA, dB);
		std::vector<float> Y((size_t)n * t.outputs());
		CUDA_CHECK(cudaMemcpy(Y.data(), dY, Y.size() * sizeof(float), cudaMemcpyDeviceToHost));
		// v4 on the CPU
		std::vector<float> o(t.outputs()), ba(t.max_width), bb(t.max_width);
		double max_gpu = 0, max_cpu = 0;
		for (int s = 0; s < n; ++s) {
			forward_serial(t, p.data(), &X[(size_t)s * t.inputs()], o.data(), ba.data(), bb.data());
			for (int j = 0; j < t.outputs(); ++j) {
				const double r = ref[(size_t)s * t.outputs() + j];
				max_gpu = std::max(max_gpu, std::fabs(Y[(size_t)s * t.outputs() + j] - r));
				max_cpu = std::max(max_cpu, std::fabs(o[j] - r));
			}
		}
		const bool ok = max_gpu < 1e-4 && max_cpu < 1e-4;
		all_ok = all_ok && ok;
		printf("  %-16s 1000 inputs: max |GPU - v3| = %.2e   max |v4 CPU - v3| = %.2e   %s\n",
		       describe(t).c_str(), max_gpu, max_cpu, ok ? "OK" : "MISMATCH");
		cudaFree(dP); cudaFree(dX); cudaFree(dA); cudaFree(dB);
	}
	printf("  %s\n\n", all_ok ? "all match (float rounding only)" : "SOME DID NOT MATCH");

	// ---------------- 2. benchmark ----------------
	printf("== Benchmark: forward passes per second ==\n");
	printf("  %-18s %8s %14s %14s %14s %14s\n", "network", "params", "v3 CPU", "v4 CPU", "GPU dataset", "GPU episodes");
	std::ofstream csv("outputs/bench.csv");
	csv << "network,params,v3_cpu,v4_cpu,gpu_dataset,gpu_episodes\n";
	for (const auto& sizes : topologies) {
		Topology t = make_topology(sizes);
		// v3 CPU
		Network::Network v3 = make_v3(sizes);
		std::vector<double> in3(t.inputs(), 0.3);
		long n3 = 0; auto t0 = Clock::now();
		while (seconds_since(t0) < 0.5) { for (int i = 0; i < 50; ++i) v3.cycle(in3); n3 += 50; }
		const double v3_rate = n3 / seconds_since(t0);
		// v4 CPU (same math, flat arrays)
		std::vector<float> p = convert_v3(v3, t), inp(t.inputs(), 0.3f), o(t.outputs()), ba(t.max_width), bb(t.max_width);
		long n4 = 0; t0 = Clock::now();
		while (seconds_since(t0) < 0.5) { for (int i = 0; i < 200; ++i) { forward_serial(t, p.data(), inp.data(), o.data(), ba.data(), bb.data()); inp[0] = o[0]; } n4 += 200; }
		const double v4_rate = n4 / seconds_since(t0);
		// GPU dataset mode: pop networks x n samples
		const int pop = t.n_params > 100000 ? 64 : 1024, n = 256;
		float *dP, *dX, *dA, *dB, *dS;
		CUDA_CHECK(cudaMalloc(&dP, (size_t)pop * t.n_params * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&dX, (size_t)n * t.inputs() * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&dA, (size_t)pop * n * t.max_width * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&dB, (size_t)pop * n * t.max_width * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&dS, pop * sizeof(float)));
		std::vector<float> Pall((size_t)pop * t.n_params), Xh((size_t)n * t.inputs());
		for (auto& v : Pall) v = (float)U(rng) * 0.1f;
		for (auto& v : Xh) v = (float)U(rng);
		CUDA_CHECK(cudaMemcpy(dP, Pall.data(), Pall.size() * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(dX, Xh.data(), Xh.size() * sizeof(float), cudaMemcpyHostToDevice));
		cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
		// warm up (lets the GPU reach full clock speed before timing)
		for (auto w0 = Clock::now(); seconds_since(w0) < 0.3;) { forward_population(t, dP, pop, dX, n, dA, dB); cudaDeviceSynchronize(); }
		const int reps = 20;
		cudaEventRecord(e0);
		for (int r = 0; r < reps; ++r) forward_population(t, dP, pop, dX, n, dA, dB);
		cudaEventRecord(e1); cudaEventSynchronize(e1);
		float ms; cudaEventElapsedTime(&ms, e0, e1);
		const double gpu_ds_rate = (double)pop * n * reps / (ms / 1000.0);
		// GPU simulation mode: one block per network, `steps` steps each
		const size_t buf = 3 * t.max_width * sizeof(float);
		const bool shared_w = t.n_params * sizeof(float) + buf <= 48 * 1024;
		const size_t smem = (shared_w ? t.n_params * sizeof(float) : 0) + buf;
		const int threads = std::max(32, std::min(256, ((t.max_width + 31) / 32) * 32));
		const int steps = t.n_params > 100000 ? 20 : 200;
		episode_bench_kernel<<<pop, threads, smem>>>(t, dP, shared_w, 5, dS);
		CUDA_CHECK(cudaGetLastError());
		cudaEventRecord(e0);
		episode_bench_kernel<<<pop, threads, smem>>>(t, dP, shared_w, steps, dS);
		cudaEventRecord(e1); cudaEventSynchronize(e1);
		CUDA_CHECK(cudaGetLastError());
		cudaEventElapsedTime(&ms, e0, e1);
		const double gpu_ep_rate = (double)pop * steps / (ms / 1000.0);
		printf("  %-18s %8d %14.3g %14.3g %14.3g %14.3g\n", describe(t).c_str(), t.n_params,
		       v3_rate, v4_rate, gpu_ds_rate, gpu_ep_rate);
		csv << describe(t) << "," << t.n_params << "," << v3_rate << "," << v4_rate << ","
		    << gpu_ds_rate << "," << gpu_ep_rate << "\n";
		cudaFree(dP); cudaFree(dX); cudaFree(dA); cudaFree(dB); cudaFree(dS);
		cudaEventDestroy(e0); cudaEventDestroy(e1);
	}
	printf("\nwrote outputs/bench.csv\n");
	return all_ok ? 0 : 1;
}
