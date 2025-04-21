#ifndef RUN_UDF_H
#define RUN_UDF_H

// This UDF simulates data-dependent work
__device__ void run_udf(int input) {
    int sum = 0;
    for (int i = 0; i < input; ++i) {
        sum += i;
    }
}

#endif
