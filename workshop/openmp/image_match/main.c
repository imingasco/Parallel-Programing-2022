#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

#define MAXH 500
#define MAXW 500
#define NTHREAD 6

inline void get_matrix(unsigned char matrix[][MAXW], int h, int w) {
    int idx = 0, i = 0, total = h * w;
    char ch, digit[4];
    while (idx < total) {
        ch = getchar_unlocked();
        switch (ch) {
            case ' ':
            case '\n':
                matrix[idx / w][idx % w] = atoi(digit);
                digit[0] = digit[1] = digit[2] = '\0';
                i = 0, idx++;
                break;
            default:
                digit[i++] = ch; 
               break;
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

int calc_diff(unsigned char A[][MAXW], int x, int y, unsigned char B[][MAXW], int Bh, int Bw) {
    int diff = 0;

    for (int i = 0; i < Bh; i++) {
        for (int j = 0; j < Bw; j++) {
            diff += (A[x + i][y + j] - B[i][j]) * (A[x + i][y + j] - B[i][j]);
        }
    }
    return diff;
}

int match(unsigned char A[][MAXW], int Ah, int Aw, unsigned char B[][MAXW], int Bh, int Bw) {
    int diff[NTHREAD] = {0};
    int min = INT_MAX;
    int ans[NTHREAD] = {0};
    int _ans;
    
    for (int i = 0; i < NTHREAD; i++) {
        diff[i] = INT_MAX;
    }

#pragma omp parallel 
    {
        int idx = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i <= Ah - Bh; i++) {
            for (int j = 0; j <= Aw - Bw; j++) {
                int _diff = calc_diff(A, i, j, B, Bh, Bw);
                if (_diff < diff[idx]) {
                    diff[idx] = _diff;
                    ans[idx] = (i << 9) + j;
                }
            }
        }
    }

    for (int i = 0; i < NTHREAD; i++) {
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

    omp_set_num_threads(NTHREAD);
    while (scanf("%d %d %d %d", &Ah, &Aw, &Bh, &Bw) == 4) {
        get_matrix(A, Ah, Aw);
        get_matrix(B, Bh, Bw);
        ans = match(A, Ah, Aw, B, Bh, Bw);
        x = ans >> 9, y = ans & 511;
        printf("%d %d\n", x + 1, y + 1);
    }
}
