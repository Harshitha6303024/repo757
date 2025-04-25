#include <iostream>
#include <cstdlib>
#include <ctime>
#include <stdint.h>
#include "run_udf.cuh"

#define WARP_SIZE 32

__global__ void profile_udf_kernel(
    int* input, int* output, int* active_cycles, uint64_t* total_cycles, int N) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    uint64_t start = clock64();

    int sum, count, active;
    run_udf(input[tid], &sum, &count, &active);

    uint64_t end = clock64();

    output[tid] = sum;
    active_cycles[tid] = active;
    total_cycles[tid] = end - start;
}

int main() {
    const int N = 64;
    int h_input[N], h_output[N], h_active_cycles[N];
    uint64_t h_total_cycles[N];

    srand(time(NULL));
    for (int i = 0; i < N; ++i)
        h_input[i] = rand() % 1000 + 100;  // Induce divergence

    int *d_input, *d_output, *d_active_cycles;
    uint64_t* d_total_cycles;

    cudaMalloc(&d_input, sizeof(int) * N);
    cudaMalloc(&d_output, sizeof(int) * N);
    cudaMalloc(&d_active_cycles, sizeof(int) * N);
    cudaMalloc(&d_total_cycles, sizeof(uint64_t) * N);

    cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);

    profile_udf_kernel<<<1, N>>>(d_input, d_output, d_active_cycles, d_total_cycles, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_active_cycles, d_active_cycles, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_total_cycles, d_total_cycles, sizeof(uint64_t) * N, cudaMemcpyDeviceToHost);

    // Find max active_cycles to normalize work values
    int max_active = 0;
    for (int i = 0; i < N; ++i) {
        if (h_active_cycles[i] > max_active)
            max_active = h_active_cycles[i];
    }

    std::cout << "\n=== Per-Thread Execution Report (with Work Value) ===\n";
    for (int i = 0; i < N; ++i) {
        double utilization = (double)h_active_cycles[i] / h_total_cycles[i] * 100.0;
        double normalized_work = (double)h_active_cycles[i] / max_active;

        std::cout << "Thread " << i
                  << " | Input: " << h_input[i]
                  << " | Output: " << h_output[i]
                  << " | Work Value (Active Cycles): " << h_active_cycles[i]
                  << " | Normalized Work: " << normalized_work
                  << " | Utilization: " << utilization << " %\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_active_cycles);
    cudaFree(d_total_cycles);
    return 0;
}
