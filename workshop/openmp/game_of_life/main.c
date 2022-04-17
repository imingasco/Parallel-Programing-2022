#include <stdio.h>
#include <string.h>
 
#define MAXN 2000
#define DEAD '0'
#define LIVE '1'
 
short encoded_game[MAXN + 2][MAXN + 3] = {};
char game[MAXN + 2][MAXN + 3] = {};
char state[512] = {};

int count_one_bits(int x) {
    int ret = 0;
    for (int i = 0; i < 9; i++) {
        if ((x >> i) & 1 == 1) {
            ret++;
        }
    }
    return ret;
}

inline int fifth_bit_is_one(int x) {
    return x & 16;
}

void compute_state() {
    for (int i = 0; i < 512; i++) {
        int ones = count_one_bits(i), live = fifth_bit_is_one(i);
        if (live && (ones == 3 || ones == 4)) {
            // rule 2
            state[i] = LIVE;
        } else if (live) {
            // rule 1 & 3
            state[i] = DEAD;
        } else if (ones == 3) {
            // rule 4
            state[i] = LIVE;
        } else {
            // DEAD remains DEAD
            state[i] = DEAD;
        }
    }
}

short encode_first(int r) {
    /*  
     *  Encoding:
     *
     *  1 0 1
     *  0 1 1 -> 100010111
     *  0 0 1
    */
    return ((game[r - 1][0] - '0') << 8) | ((game[r][0] - '0') << 7) | ((game[r + 1][0] - '0') << 6) | \
           ((game[r - 1][1] - '0') << 5) | ((game[r][1] - '0') << 4) | ((game[r + 1][1] - '0') << 3) | \
           ((game[r - 1][2] - '0') << 2) | ((game[r][2] - '0') << 1) | (game[r + 1][2] - '0');
}

short encode(int r, int c) {
    return ((encoded_game[r][c - 1] << 3) & 511) | ((game[r - 1][c + 1] - '0') << 2) | \
           ((game[r][c + 1] - '0') << 1) | (game[r + 1][c + 1] - '0');
}

void encode_game(int N) {
#pragma omp parallel for
    for (int i = 1; i <= N; i++) {
        encoded_game[i][1] = encode_first(i);
        for (int j = 2; j <= N; j++) {
            encoded_game[i][j] = encode(i, j);
        }
    }
}

void print_matrix(int N) {
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            printf("%hd ", encoded_game[i][j]);
        }
        printf("\n");
    }
    return;
}
 
void print_game(int N) {
    for (int i = 1; i <= N; i++) {
        game[i][N + 1] = '\0';
        puts(&game[i][1]);
    }
    return;
}
 
void next_round(int N) {
#pragma omp parallel for
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            game[i][j] = state[encoded_game[i][j]];
        }
    }
    encode_game(N);
}
 
int main(int argc, char **argv) {
    int N, M;
    scanf("%d %d\n", &N, &M);

    for (int i = 1; i <= N; i++) {
        gets(&game[i][1]);
    }
    for (int i = 0; i <= N + 1; i++) {
        game[0][i] = game[N + 1][i] = game[i][0] = game[i][N + 1] = '0';
    }

    compute_state();
    encode_game(N);
    for (int i = 0; i < M; i++) {
        next_round(N);
    }
    print_game(N);
    return 0;
}
