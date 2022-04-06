#include "matrix.h"
#include <omp.h>

#define NTHREAD 6

void transpose(int N, unsigned long matrix[][2048]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            unsigned long tmp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = tmp;
        }
    }
    return;
}

void multiply(int N, unsigned long A[][2048], unsigned long B[][2048], unsigned long C[][2048]) {
    omp_set_num_threads(NTHREAD);
    transpose(N, B);
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            unsigned long sum = 0;    // overflow, let it go.
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[j][k];
            C[i][j] = sum;
        }
    }
}
