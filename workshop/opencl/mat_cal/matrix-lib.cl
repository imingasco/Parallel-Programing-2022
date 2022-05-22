#define MAXN 1024
#define UINT unsigned int
 
__kernel void mul(__global UINT *A, __global UINT *B, __global UINT *C) {
    // Perform A * B = C
    int N = get_global_size(0);
    int global_r = get_global_id(0);
    int global_c = get_global_id(1);
    UINT sum = 0;
    for (int k = 0; k < N; k++) {
        sum += A[global_r * MAXN + k] * B[k * MAXN + global_c];
    }
    C[global_r * MAXN + global_c] = sum;
}
 
__kernel void add(__global UINT *A, __global UINT *B, __global UINT *C) {
    // Perform A + B = C
    int global_r = get_global_id(0);
    int global_c = get_global_id(1);
    int idx = global_r * MAXN + global_c;
    C[idx] = A[idx] + B[idx];
}
