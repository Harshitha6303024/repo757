#ifndef RUN_UDF_H
#define RUN_UDF_H

// Uniform workload: No control-flow or memory divergence
__device__ void run_udf(int input, int* out_sum, int* out_count) {
    int sum_result = 0;
    int loop_count = 0;

    int i = 0;
    while (i < input) {
        sum_result += i * i;
        loop_count++;
        i++;
    }

    *out_sum = sum_result;
    *out_count = loop_count;
}

#endif
