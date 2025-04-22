#ifndef RUN_UDF_CUH
#define RUN_UDF_CUH

#include <cuda_runtime.h>
#include <stdio.h>

// Divergent UDF based on random input loop count
__device__ int run_udf(int input) {
    float x = 1.0f;

    for (int i = 0; i < input; ++i) {
        x += i * 0.0001f; // Much smaller growth to avoid overflow
    }

    // Convert to int safely (x won't grow explosively now)
    return static_cast<int>(x * 1000.0f);
}

#endif
