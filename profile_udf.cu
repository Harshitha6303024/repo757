#include <iostream>
#include <fstream>
#include <stdint.h>
#include <cuda.h>
#include "run_udf.cuh"

// Kernel to profile thread execution time of the UDF
__global__ void profile_udf_kernel(int* input, uint64_t* timing, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    uint64_t start = clock64();
    run_udf(input[tid]);
    uint64_t end = clock64();

    timing[tid] = end - start;
}

int main() {
    const int N = 128; // Total number of threads (e.g. 4 warps)
    int h_input[N];
    
    // Create variable workload per thread to simulate divergence
    for (int i = 0; i < N; i++) {
        h_input[i] = (i % 32) * 10 + 1;  // Threads in the same warp get different work
    }

    int* d_input;
    uint64_t* d_timing;
    uint64_t h_timing[N];

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_timing, N * sizeof(uint64_t));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    profile_udf_kernel<<<N / 32, 32>>>(d_input, d_timing, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_timing, d_timing, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Print and save timings
    std::ofstream fout("timing_output.txt");
    for (int i = 0; i < N; ++i) {
        std::cout << "Thread " << i << ": time = " << h_timing[i] << " cycles\n";
        fout << h_timing[i] << "\n";
    }

    fout.close();
    cudaFree(d_input);
    cudaFree(d_timing);

    return 0;
}
