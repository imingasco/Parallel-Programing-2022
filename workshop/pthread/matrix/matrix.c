#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
 
#define THREADS 6
#define MAXN 2048

typedef struct Task {
    int start, end, N;
    unsigned long *A;
    unsigned long *B;
    unsigned long *C;
}Task;

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

void *multiply_thread(void *arg) {
    Task *t = (Task *)arg;
    int start = t->start, end = t->end, N = t->N;
    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            unsigned long sum = 0;
            for (int k = 0; k < N; k++) {
                sum += t->A[i * MAXN + k] * t->B[j * MAXN + k];
            }
            t->C[i * MAXN + j] = sum;
        }
    }
    pthread_exit(NULL);
}

int allocate(int threads, int N) {
    if (threads > N) {
        return 1;
    }
    int amount = N / threads;
    if (amount * threads < N) {
        amount++;
    }
    return amount;
}

void multiply(int N, unsigned long A[][2048], unsigned long B[][2048], unsigned long C[][2048]) {
    pthread_t t[THREADS];
    pthread_attr_t attr;
    Task task[THREADS];
    int start = 0, remaining = N, total = 0, num = 0;

    transpose(N, B);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(PTHREAD_CREATE_JOINABLE);
    for (int i = 0; i < THREADS; i++) {
        if (!remaining)
            continue;
        num = allocate(THREADS - i, remaining);
        task[i].start = start, task[i].end = start + num, task[i].N = N;
        task[i].A = (unsigned long *)A, task[i].B = (unsigned long *)B, task[i].C = (unsigned long *)C;
        pthread_create(&t[i], &attr, multiply_thread, (void *)&task[i]);
        start += num;
        remaining -= num;
        total++;
    }
    for (int i = 0; i < total; i++) { 
        pthread_join(t[i], NULL);
    }
    return;
}
