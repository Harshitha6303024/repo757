#ifndef RUN_UDF_CUH
#define RUN_UDF_CUH

#include <cuda_runtime.h>
#include <stdio.h>

// UDF that introduces thread divergence based on input
__device__ void run_udf(int input) {
    int sum = 0;

    if (input % 4 == 0) {
        for (int i = 0; i < 1000; ++i)
            sum += i;
    } else if (input % 4 == 1) {
        for (int i = 0; i < 5000; ++i)
            sum += i % 7;
    } else if (input % 4 == 2) {
        for (int i = 0; i < 10000; ++i)
            sum += i % 3;
    } else {
        for (int i = 0; i < 20000; ++i)
            sum += i % 5;
    }

    // Prevent compiler optimization
    if (sum == -1) printf("Impossible\n");
}

#endif
