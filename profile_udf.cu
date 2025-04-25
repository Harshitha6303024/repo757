#include <iostream>
#include <iomanip>      // For std::setw and std::setprecision
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
    const int N = 32; // one warp
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

    // Find max active_cycles to normalize work scores
    int max_active = 0;
    for (int i = 0; i < N; ++i) {
        if (h_active_cycles[i] > max_active)
            max_active = h_active_cycles[i];
    }

    std::cout << "\n=== Per-Thread Execution Report (Integer Work Score) ===\n";
    for (int i = 0; i < N; ++i) {
        int work_value = h_active_cycles[i]; // Raw active cycles
        int work_score = (int)(((double)work_value / max_active) * 100.0);  // Score from 0–100
        double utilization = (double)h_active_cycles[i] / h_total_cycles[i] * 100.0;

        std::cout << "Thread " << std::setw(2) << i
                  << " | Input: " << std::setw(4) << h_input[i]
                  << " | Output: " << std::setw(9) << h_output[i]
                  << " | Work Value: " << std::setw(4) << work_value
                  << " | Work Score: " << std::setw(3) << work_score << "/100"
                  << " | Utilization: " << std::fixed << std::setprecision(2) << utilization << " %\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_active_cycles);
    cudaFree(d_total_cycles);
    return 0;
}
