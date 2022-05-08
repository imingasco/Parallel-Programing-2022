/* constant */
#define MAXN 1024
#define BLOCKSIZE 32

__kernel void mul(int N, __global int A[MAXN][MAXN], __global int B[MAXN][MAXN], __global int C[MAXN][MAXN]) 
{
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);
    int blockNum = N / BLOCKSIZE;

    __local int ALocal[BLOCKSIZE][BLOCKSIZE];
    __local int BLocal[BLOCKSIZE][BLOCKSIZE];
    int sum = 0;  
    for (int block = 0; block < blockNum; block++) {
        ALocal[localRow][localCol] = 
          A[globalRow][block * BLOCKSIZE + localCol];
        BLocal[localRow][localCol] = 
          B[globalRow][block * BLOCKSIZE + localCol];
        barrier(CLK_LOCAL_MEM_FENCE);
        /* inner */
        for (int k = 0; k < BLOCKSIZE; k++) 
          sum += ALocal[localRow][k] * BLocal[localRow][k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[globalRow][globalCol] = sum;
}

__kernel void add(__global int A[MAXN][MAXN], __global int B[MAXN][MAXN], __global int C[MAXN][MAXN])
{
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);
    C[globalRow][globalCol] = A[globalRow][globalCol] + B[globalRow][globalCol];
}

