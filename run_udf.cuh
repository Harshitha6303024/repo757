#ifndef RUN_UDF_CUH
#define RUN_UDF_CUH

#include <cuda_runtime.h>
#include <stdio.h>

// Input-dependent loop to create divergence
__device__ int run_udf(int input) {
    float x = 1.0f;
    for (int i = 0; i < input; ++i) {
        x += i * x;
    }
    return static_cast<int>(x);
}

#endif
