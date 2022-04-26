#include <omp.h>
#include <stdio.h>
#include <string.h>
#define NTHREAD 128

inline int row_ok(int r, int candidate, int sudoku[][9]) {
    return sudoku[r][0] != candidate && sudoku[r][1] != candidate && sudoku[r][2] != candidate &&
           sudoku[r][3] != candidate && sudoku[r][4] != candidate && sudoku[r][5] != candidate &&
           sudoku[r][6] != candidate && sudoku[r][7] != candidate && sudoku[r][8] != candidate;
}

inline int col_ok(int c, int candidate, int sudoku[][9]) {
    return sudoku[0][c] != candidate && sudoku[1][c] != candidate && sudoku[2][c] != candidate &&
           sudoku[3][c] != candidate && sudoku[4][c] != candidate && sudoku[5][c] != candidate &&
           sudoku[6][c] != candidate && sudoku[7][c] != candidate && sudoku[8][c] != candidate;
}

inline int block_ok(int r, int c, int candidate, int sudoku[][9]) {
    int rstart = (r / 3) * 3, cstart = (c / 3) * 3;
    return sudoku[rstart][cstart] != candidate && sudoku[rstart][cstart + 1] != candidate &&
           sudoku[rstart][cstart + 2] != candidate && sudoku[rstart + 1][cstart] != candidate &&
           sudoku[rstart + 1][cstart + 1] != candidate && sudoku[rstart + 1][cstart + 2] != candidate &&
           sudoku[rstart + 2][cstart] != candidate && sudoku[rstart + 2][cstart + 1] != candidate &&
           sudoku[rstart + 2][cstart + 2] != candidate;
}

int ok(int r, int c, int candidate, int sudoku[][9]) {
    return row_ok(r, candidate, sudoku) && col_ok(c, candidate, sudoku) && block_ok(r, c, candidate, sudoku);
}

int solve(int idx, int sudoku[][9]) {
    if (idx == 81) {
        return 1;
    }
    int r = idx / 9, c = idx % 9, ans = 0;
    if (sudoku[r][c]) {
        return solve(idx + 1, sudoku);
    }
    for (int i = 1; i <= 9; i++) {
        if (ok(r, c, i, sudoku)) {
            sudoku[r][c] = i;
            ans += solve(idx + 1, sudoku);
            sudoku[r][c] = 0;
        }
    }
    return ans;
}

int main(int argc, char **argv) {
    int sudoku[9][9] = {};
    int zeros[4], idx = 0, ans = 0;
    omp_set_num_threads(NTHREAD);
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            scanf("%d", &sudoku[i][j]);
            if (idx < 4 && sudoku[i][j] == 0) {
                zeros[idx++] = 9 * i + j;
            }
        }
    }
#pragma omp parallel for reduction(+ : ans) collapse(4) schedule(dynamic, 4)
    for (int i = 1; i <= 9; i++) {
        for (int j = 1; j <= 9; j++) {
            for (int k = 1; k <= 9; k++) {
                for (int l = 1; l <= 9; l++) {
                    int r0 = zeros[0] / 9, c0 = zeros[0] % 9;
                    int r1 = zeros[1] / 9, c1 = zeros[1] % 9;
                    int r2 = zeros[2] / 9, c2 = zeros[2] % 9;
                    int r3 = zeros[3] / 9, c3 = zeros[3] % 9;
                    int parallel_sudoku[9][9];
                    memcpy(parallel_sudoku, sudoku, 324);
                    if (ok(r0, c0, i, parallel_sudoku)) {
                        parallel_sudoku[r0][c0] = i;
                        if (ok(r1, c1, j, parallel_sudoku)) {
                            parallel_sudoku[r1][c1] = j;
                            if (ok(r2, c2, k, parallel_sudoku)) {
                                parallel_sudoku[r2][c2] = k;
                                if (ok(r3, c3, l, parallel_sudoku)) {
                                    parallel_sudoku[r3][c3] = l;
                                    ans += solve(zeros[3] + 1, parallel_sudoku);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    printf("%d\n", ans);

    return 0;
}
