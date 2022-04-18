#include <omp.h>
#include <stdio.h>
#include <string.h>
#define MAXN 10000
#define MAXW 1000000
#define NTHREAD 24
 
struct Item {
    int weight;
    int value;
} items[MAXN];
 
inline int max(int a, int b) {
    return a > b? a : b;
}
 
int dp[2][MAXW + 1] = {};
 
int main(int argc, char **argv) {
    omp_set_num_threads(NTHREAD);
    int N, M, cur = 0, next = 1;
    scanf("%d %d", &N, &M);
    for (int i = 0; i < N; i++) {
        scanf("%d %d", &items[i].weight, &items[i].value);
    }
    for (int i = items[0].weight; i <= M; i++)
        dp[cur][i] = items[0].value;
    for (int i = 1; i < N; i++) {
        int weight = items[i].weight;
        int value = items[i].value;
#pragma omp parallel for
        for (int j = 0; j < weight; j++) {
            dp[next][j] = dp[cur][j];
        }
#pragma omp parallel for
        for (int j = weight; j <= M; j++) {
            dp[next][j] = max(dp[cur][j], dp[cur][j - weight] + value);
        }
        cur ^= 1, next ^= 1;
    }
    printf("%d\n", dp[0][M]);
    return 0;
}
