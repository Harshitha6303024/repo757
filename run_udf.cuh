#ifndef RUN_UDF_CUH
#define RUN_UDF_CUH

__device__ void run_udf(int input) {
    int sum = 0;
    for (int i = 0; i < input; ++i) {
        sum += i;
    }
}

#endif

