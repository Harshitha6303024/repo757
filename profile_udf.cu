#include <iostream>
#include <cstdlib>
#include <ctime>
#include <stdint.h>
#include "run_udf_divergent.cuh"

#define WARP_SIZE 32

__global__ void profile_udf_kernel(int* input, int* output, uint64_t* timing, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    uint64_t start = clock64();
    int sum, count;
    run_udf(input[tid], &sum, &count);
    output[tid] = sum;
    uint64_t end = clock64();

    timing[tid] = end - start;
}

int main() {
    const int N = 64;
    int h_input[N];
    int h_output[N];
    uint64_t h_timing[N];

    srand(time(NULL));
    for (int i = 0; i < N; ++i)
        h_input[i] = rand() % 1000 + 100;  // Random input = divergence

    int *d_input, *d_output;
    uint64_t* d_timing;
    cudaMalloc(&d_input, sizeof(int) * N);
    cudaMalloc(&d_output, sizeof(int) * N);
    cudaMalloc(&d_timing, sizeof(uint64_t) * N);

    cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);

    profile_udf_kernel<<<1, N>>>(d_input, d_output, d_timing, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_timing, d_timing, sizeof(uint64_t) * N, cudaMemcpyDeviceToHost);

    std::cout << "\n=== Divergent Execution Times ===\n";
    for (int i = 0; i < N; ++i)
        std::cout << "Thread " << i << " | Input: " << h_input[i]
                  << " | Time: " << h_timing[i] << " cycles\n";

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_timing);
    return 0;
}
