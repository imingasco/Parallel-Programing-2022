#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>
#include "utils.h"
 
#define MAXN 10000005
#define MAX_THREAD 4
uint32_t prefix_sum[MAXN];
 
int min(int a, int b) {
    return a < b? a : b;
}

void prefix_sum_parallel(uint32_t arr[], int n) {
    // Since we have 4 threads only, let's compute it directly
    /*
    for (int i = 1; i < n; i++) {
        arr[i] += arr[i - 1];
    }
    */
    arr[1] += arr[0];
    arr[2] += arr[1];
    arr[3] += arr[2];
}
 
void patch(uint32_t prefix_sum[], int n, uint32_t final[], int t, int last_idx[]) {
    for (int i = 1; i < t; i++) {
        int start = last_idx[i - 1] + 1, last = last_idx[i];
#pragma omp parallel for
        for (int j = start; j <= last; j++) {
            prefix_sum[j] += final[i - 1];
        }
    }
}
 
int main() {
    omp_set_num_threads(MAX_THREAD);
 
    int last_idx[MAX_THREAD] = {}, n;
    uint32_t final[MAX_THREAD] = {}, key;
    while (scanf("%d %" PRIu32, &n, &key) == 2) {
        int q = n >> 2, r = n & 3;
        last_idx[0] = q, last_idx[1] = q << 1, last_idx[2] = (q << 1) + q, last_idx[3] = q << 2;
        if (r) {
            for (int i = 0; i < 4; i++) {
                last_idx[i] += min(i + 1, r);
            }
        }
#pragma omp parallel
        {
            int idx = omp_get_thread_num();
            uint32_t sum = 0;
#pragma omp for
            for (int i = 1; i <= n; i++) {
                sum += encrypt(i, key);
                prefix_sum[i] = sum;
            }
            final[idx] = sum;
        } /* parallel */
        prefix_sum_parallel(final, MAX_THREAD);
        patch(prefix_sum, n, final, MAX_THREAD, last_idx);
        output(prefix_sum, n);
    }
    return 0;
}
