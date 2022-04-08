#include <stdio.h>
#include <limits.h>
#include <omp.h>
 
#define MAXH 500
#define MAXW 500
#define NTHREAD 16
 
void get_matrix(unsigned char matrix[][MAXW], int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            scanf("%d", &matrix[i][j]);
        }
    }
}
 
void print_matrix(unsigned char matrix[][MAXW], int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}
 
unsigned long long calc_diff(unsigned char A[][MAXH], int x, int y, unsigned char B[][MAXW], int Bh, int Bw) {
    unsigned long long diff = 0;
 
    for (int i = 0; i < Bh; i++) {
        for (int j = 0; j < Bw; j++) {
            diff += (A[x + i][y + j] - B[i][j]) * (A[x + i][y + j] - B[i][j]);
        }
    }
    return diff;
}
 
int match(unsigned char A[][MAXW], int Ah, int Aw, unsigned char B[][MAXW], int Bh, int Bw) {
    int _ans, nthread;
    unsigned long long min = ULONG_MAX;
#pragma omp parallel
    nthread = omp_get_num_threads();
    unsigned long long diff[nthread];
    int ans[nthread];
 
    for (int i = 0; i < nthread; i++) {
        diff[i] = ULONG_MAX;
    }
 
#pragma omp parallel
    {
        int idx = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i <= Ah - Bh; i++) {
            for (int j = 0; j <= Aw - Bw; j++) {
                unsigned long long _diff = calc_diff(A, i, j, B, Bh, Bw);
                if (_diff < diff[idx]) {
                    diff[idx] = _diff;
                    ans[idx] = (i << 9) + j;
                }
            }
        }
    }
    for (int i = 0; i < nthread; i++) {
        if (diff[i] < min) {
            _ans = ans[i];
            min = diff[i];
        }
    }
    return _ans;
}
 
int main(int argc, char **argv) {
    int Ah, Aw, Bh, Bw, ans, x, y;
    unsigned char A[MAXH][MAXW], B[MAXH][MAXW];
 
    while (scanf("%d %d %d %d", &Ah, &Aw, &Bh, &Bw) != EOF) {
        get_matrix(A, Ah, Aw);
        get_matrix(B, Bh, Bw);
        ans = match(A, Ah, Aw, B, Bh, Bw);
        x = ans >> 9, y = ans & 511;
        printf("%d %d\n", x + 1, y + 1);
    }
}
