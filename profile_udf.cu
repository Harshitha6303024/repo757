#include <iostream>
#include <stdint.h>
#include "run_udf.cuh"
#include <cuda.h>

__global__ void profile_udf_kernel(int* input, int* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    output[tid] = run_udf(input[tid]);
}

int main() {
    const int N = 64;
    int h_input[N];
    int h_output[N];

    for (int i = 0; i < N; ++i)
        h_input[i] = i;

    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, sizeof(int) * N);
    cudaMalloc(&d_output, sizeof(int) * N);

    cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);

    profile_udf_kernel<<<1, N>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        std::cout << "Thread " << i << " output: " << h_output[i] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
