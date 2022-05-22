#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
 
#define MAXN 2000
#define MAXGROUP 1024
#define DEAD '0'
#define LIVE '1'
 
int game[2][MAXN * MAXN] = {};
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
cl_mem mem[2];
 
void print_game(int N, int M) {
    int idx = M % 2;
    for (int i = 0; i < N; i++) {
        char buf[MAXN + 1];
        for (int j = 0; j < N; j++) {
            buf[j] = '0' + game[M % 2][i * MAXN + j];
        }
        buf[N] = '\0';
        puts(buf);
    }
    return;
}
 
size_t min(size_t a, size_t b) {
    return (a < b)? a : b;
}
 
size_t calculate_group_size(size_t num) {
    size_t start = min(MAXGROUP, num - 1);
    for (int i = start; i > 0; i--) {
        if (num % i) {
            return i;
        }
    }
    return 1;
}
 
void next_round(int N, int round) {
    cl_int status;
 
    /* NDrange setup */
    int total = N * N;
    size_t global_size[] = {(size_t)MAXN * MAXN};
    size_t local_size[] = {1000};
 
    /* Kernel args */
    int now = (round + 1) % 2, next = round % 2;
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *) &N);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &mem[now]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &mem[next]);
    assert(status == CL_SUCCESS);
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* Sync */
    status = clEnqueueReadBuffer(queue, mem[next], CL_TRUE, 0, sizeof(int) * MAXN * MAXN, game[next], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
}
 
void OpenCLInit() {
    cl_int status;
    cl_uint num;
 
    /* Platform */
    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, &num);
    assert(status == CL_SUCCESS);
 
    /* Device */
    cl_device_id device;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num);
    assert(status == CL_SUCCESS);
 
    /* Context */
    context = clCreateContext(NULL, 1, (const cl_device_id *) &device, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
 
    /* Command queue */
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &status);
    assert(status == CL_SUCCESS);
 
    /* Kernel source */
    char source[2048] = "";
    const char *source_ptr = source;
    FILE *fp = fopen("game-of-life.cl", "r");
    size_t source_len = fread(source, 1, 2048, fp);
    program = clCreateProgramWithSource(context, 1, &source_ptr, &source_len, &status);
    assert(status == CL_SUCCESS);
 
    /* Build program */
    status = clBuildProgram(program, 1, (const cl_device_id *) &device, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        char log[2048] = "";
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 2048, log, NULL);
        fprintf(stderr, "%s", log);
        exit(0);
    }
 
    /* Kernel */
    kernel = clCreateKernel(program, "next_round", &status);
    assert(status == CL_SUCCESS);
 
    /* Buffer */
    for (int i = 0; i < 2; i++) {
        mem[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int) * MAXN * MAXN, game[i], &status);
        assert(status == CL_SUCCESS);
    }
}
 
int main(int argc, char **argv) {
    /* Read input */
    int N, M;
    char buf[MAXN + 5];
    scanf("%d %d\n", &N, &M);
 
    for (int i = 0; i < N; i++) {
        gets(buf);
        for (int j = 0; j < N; j++) {
            game[0][i * MAXN + j] = buf[j] - '0';
        }
    }
 
    OpenCLInit();
    for (int i = 1; i <= M; i++) {
        next_round(N, i);
    }
    print_game(N, M);
    return 0;
}
