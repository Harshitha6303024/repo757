#ifndef RUN_UDF_CUH
#define RUN_UDF_CUH

#include <cuda_runtime.h>
#include <stdio.h>

// UDF that creates heavy divergence among threads
__device__ int run_udf(int input) {
    int sum = 0;

    if (input % 8 == 0) {
        for (int i = 0; i < 100000; ++i) sum += i % 13;
    } else if (input % 8 == 1) {
        for (int i = 0; i < 150000; ++i) sum += i % 11;
    } else if (input % 8 == 2) {
        for (int i = 0; i < 200000; ++i) sum += i % 7;
    } else if (input % 8 == 3) {
        for (int i = 0; i < 250000; ++i) sum += i % 5;
    } else if (input % 8 == 4) {
        for (int i = 0; i < 300000; ++i) sum += i % 3;
    } else if (input % 8 == 5) {
        for (int i = 0; i < 350000; ++i) sum += i % 17;
    } else if (input % 8 == 6) {
        for (int i = 0; i < 400000; ++i) sum += i % 19;
    } else {
        for (int i = 0; i < 450000; ++i) sum += i % 23;
    }

    return sum;
}

#endif
