#ifndef RUN_UDF_H
#define RUN_UDF_H

// Divergence-heavy UDF: uses input-dependent branches and varying loop complexity
__device__ void run_udf(int input, int* out_sum, int* out_count) {
    int sum_result = 0;
    int loop_count = 0;

    int i = 0;
    while (i < input) {
        if (input % 3 == 0) {
            sum_result += i * i;           // Quadratic workload
        } else if (input % 3 == 1) {
            sum_result += i + (i % 5);     // Mixed arithmetic
        } else {
            sum_result += (i % 3 == 0) ? i * 2 : i;  // Irregular access
        }

        if ((i % 7) == 0) {
            // Simulate extra computation occasionally
            for (int j = 0; j < (input % 5); ++j) {
                sum_result += j;
            }
        }

        loop_count++;
        i++;
    }

    *out_sum = sum_result;
    *out_count = loop_count;
}

#endif
