#ifndef RUN_UDF_H
#define RUN_UDF_H

// Divergence-heavy UDF with internal active cycle approximation
__device__ void run_udf(int input, int* out_sum, int* out_count, int* out_active_cycles) {
    int sum_result = 0;
    int loop_count = 0;
    int active_cycles = 0;

    int i = 0;
    while (i < input) {
        // Simulate compute
        if (input % 3 == 0)
            sum_result += i * i;
        else if (input % 3 == 1)
            sum_result += i + (i % 5);
        else
            sum_result += (i % 3 == 0) ? i * 2 : i;

        if (i % 7 == 0) {
            for (int j = 0; j < (input % 5); ++j) {
                sum_result += j;
                active_cycles++;  // Additional work
            }
        }

        active_cycles++;  // Count normal compute
        loop_count++;
        i++;
    }

    *out_sum = sum_result;
    *out_count = loop_count;
    *out_active_cycles = active_cycles;
}

#endif
