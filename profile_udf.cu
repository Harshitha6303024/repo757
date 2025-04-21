#include <iostream>
#include <fstream>
#include <stdint.h>
#include "run_udf.cuh"
#include <cuda.h>

// Kernel to profile per-thread execution time
__global__ void profile_udf_kernel(int* input, uint64_t* timing, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    uint64_t start = clock64();
    run_udf(input[tid]);
    uint64_t end = clock64();

    timing[tid] = end - start;
}

int main() {
    const int N = 64;
    int h_input[N];
    uint64_t h_timing[N];

    // Fill inputs to force divergence (mix of 0, 1, 2, 3 % 4)
    for (int i = 0; i < N; ++i)
        h_input[i] = i;

    int* d_input;
    uint64_t* d_timing;
    cudaMalloc(&d_input, sizeof(int) * N);
    cudaMalloc(&d_timing, sizeof(uint64_t) * N);

    cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);

    profile_udf_kernel<<<1, N>>>(d_input, d_timing, N);
    cudaMemcpy(h_timing, d_timing, sizeof(uint64_t) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << "Thread " << i << " (input=" << h_input[i] << ") took " 
                  << h_timing[i] << " cycles\n";
    }

    cudaFree(d_input);
    cudaFree(d_timing);
    return 0;
}
