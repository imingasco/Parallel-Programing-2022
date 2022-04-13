#include <omp.h>
#include <stdio.h>
#include <string.h>

#define MAXN 2000
#define DEAD '0'
#define LIVE '1'

const int neighbor[8][2] = {
    {-1, -1}, {-1, 0}, {-1, 1},
    {0, -1}, {0, 1},
    {1, -1}, {1, 0}, {1, 1}
};

char game[MAXN + 2][MAXN + 3];

void print_matrix(int N) {
    for (int i = 0; i <= N + 1; i++) {
        printf("%s\n", game[i]);
    }
    return;
}

void print_game(int N) {
    for (int i = 1; i <= N; i++) {
        game[i][N + 1] = '\0';
        puts(&game[i][1]);
        // game[i][N + 1] = '0';
        // printf("%.*s\n", N, &game[i][1]);
    }
    return;
}

inline int is_live(char c) {
    return c == LIVE;
}

void next_round(int N) {
    char live_count[MAXN + 2][MAXN + 3] = {};
#pragma omp parallel 
    {
    #pragma omp for
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                if (is_live(game[i][j])) {
                    for (int k = 0; k < 8; k++) {
                        live_count[i + neighbor[k][0]][j + neighbor[k][1]]++;
                    }
                }
            }
        }
    #pragma omp for
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                int live = is_live(game[i][j]);
                if (live && (live_count[i][j] < 2 || live_count[i][j] > 3)) {
                    game[i][j] = DEAD;
                } else if (!live && live_count[i][j] == 3) {
                    game[i][j] = LIVE;
                }
            }
        }
    } /* parallel */
}

int main(int argc, char **argv) {
    int N, M;

    scanf("%d %d\n", &N, &M);
    for (int i = 0; i < N; i++) {
        /*
        for (int j = 0; j < N; j++) {
            game[i + 1][j + 1] = getchar_unlocked();
            if (game[i + 1][j + 1] == '\n')
                game[i + 1][j + 1] = getchar_unlocked();
        }
        */
        gets(&game[i + 1][1]);
        // game[0][i] = game[N + 1][i] = game[i + 1][0] = game[i + 1][N + 1] = '0'; // pad the map
    }
    // game[0][N] = game[0][N + 1] = game[N + 1][N] = game[N + 1][N + 1] = '0'; // pad the map
    for (int i = 0; i < M; i++) {
        next_round(N);
    }
    print_game(N);
    return 0;
}
