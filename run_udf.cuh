#ifndef RUN_UDF_CUH
#define RUN_UDF_CUH

#include <cuda_runtime.h>

// UDF that simulates input-dependent workload
__device__ void run_udf(int input) {
    int sum = 0;
    for (int i = 0; i < input * 1000; ++i) {
        sum += i % 7; // Light computation inside loop
    }
}

#endif
