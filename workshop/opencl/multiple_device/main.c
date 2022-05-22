#include <CL/cl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
 
#define UINT uint32_t
#define MAXN 1024
#define MAXSIZE 32
#define NGPU 2
 
int N;
UINT key[6];
UINT mat[6][MAXN][MAXN];
UINT tmp[4][MAXN][MAXN];
UINT ans[2][MAXN][MAXN];
cl_context context;
cl_command_queue queue[NGPU];
cl_program program;
cl_kernel kernel_mul, kernel_add;
cl_mem mem[6][3];
cl_event event[4];
 
void rand_gen(UINT c, int N, UINT A[][MAXN]) {
    UINT x = 2, n = N*N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j)%n;
            A[i][j] = x;
        }
    }
}
 
void print_matrix(int N, UINT A[][MAXN]) {
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "[");
        for (int j = 0; j < N; j++)
            fprintf(stderr, " %u", A[i][j]);
        fprintf(stderr, " ]\n");
    }
}
 
UINT signature(int N, UINT A[][MAXN]) {
    UINT h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i][j]) * 2654435761LU;
    }
    return h;
}
 
void OpenCLInit() {
    cl_int status;
    cl_uint num;
 
    /* Platform */
    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, &num);
    assert(status == CL_SUCCESS);
 
    /* Device */
    cl_device_id device[NGPU];
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, NGPU, device, &num);
    assert(status == CL_SUCCESS);
 
    /* Context */
    context = clCreateContext(NULL, NGPU, device, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
 
    /* Command queue */
    for (int i = 0; i < NGPU; i++) {
        queue[i] = clCreateCommandQueueWithProperties(context, device[i], NULL, &status);
        assert(status == CL_SUCCESS);
    }
 
    /* Kernel source */
    char source[2048] = "";
    const char *source_ptr = source;
    FILE *fp = fopen("matrix-lib.cl", "r");
    size_t source_len = fread(source, 1, 2048, fp);
    program = clCreateProgramWithSource(context, 1, &source_ptr, &source_len, &status);
    assert(status == CL_SUCCESS);
 
    /* Build program */
    status = clBuildProgram(program, NGPU, device, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        for (int i = 0; i < NGPU; i++) {
            char log[2048] = "";
            clGetProgramBuildInfo(program, device[i], CL_PROGRAM_BUILD_LOG, 2048, log, NULL);
            fprintf(stderr, "Device %d: %s", i, log);
        }
        exit(0);
    }
 
    /* Kernel */
    kernel_mul = clCreateKernel(program, "mul", &status);
    assert(status == CL_SUCCESS);
    kernel_add = clCreateKernel(program, "add", &status);
    assert(status == CL_SUCCESS);
}
 
void mul(UINT A[][MAXN], UINT B[][MAXN], UINT C[][MAXN], cl_command_queue target, cl_mem bufs[3],
         cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    cl_int status;
    /* Buffer */
    bufs[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, A, &status);
    assert(status == CL_SUCCESS);
    bufs[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, B, &status);
    assert(status == CL_SUCCESS);
    bufs[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, C, &status);
    assert(status == CL_SUCCESS);
 
    /* NDRange setup */
    size_t global_size[] = {MAXN, MAXN};
    size_t local_size[] = {16, 16};
 
    /* Kernel args */
    for (int i = 0; i < 3; i++) {
        status = clSetKernelArg(kernel_mul, i, sizeof(cl_mem), (void *) &bufs[i]);
        assert(status == CL_SUCCESS);
    }
    status = clSetKernelArg(kernel_mul, 3, sizeof(cl_int), (void *) &N);
    assert(status == CL_SUCCESS);
    status = clEnqueueNDRangeKernel(target, kernel_mul, 2, NULL, global_size, local_size, num_events_in_wait_list, event_wait_list, event);
    assert(status == CL_SUCCESS);
}
 
void add(UINT A[][MAXN], UINT B[][MAXN], UINT C[][MAXN], cl_command_queue target, cl_mem bufs[3],
         cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    cl_int status;
    /* Buffer */
    bufs[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, A, &status);
    assert(status == CL_SUCCESS);
    bufs[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, B, &status);
    assert(status == CL_SUCCESS);
    bufs[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, C, &status);
    assert(status == CL_SUCCESS);
 
    /* NDRange setup */
    size_t global_size[] = {MAXN, MAXN};
    size_t local_size[] = {16, 16};
 
    /* Kernel args */
    for (int i = 0; i < 3; i++) {
        status = clSetKernelArg(kernel_add, i, sizeof(cl_mem), (void *) &bufs[i]);
        assert(status == CL_SUCCESS);
    }
    status = clSetKernelArg(kernel_add, 3, sizeof(cl_int), (void *) &N);
    assert(status == CL_SUCCESS);
    status = clEnqueueNDRangeKernel(target, kernel_add, 2, NULL, global_size, local_size, num_events_in_wait_list, event_wait_list, event);
    assert(status == CL_SUCCESS);
}
 
void OpenCLExecute() {
    cl_int status;
 
    /* AB */
    mul(mat[0], mat[1], tmp[0], queue[0], mem[0], 0, NULL, NULL);
 
    /* CD */
    mul(mat[2], mat[3], tmp[1], queue[1], mem[1], 0, NULL, NULL);
 
    status = clEnqueueReadBuffer(queue[0], mem[0][2], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, tmp[0], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
    status = clEnqueueReadBuffer(queue[1], mem[1][2], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, tmp[1], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* ABE */
    mul(tmp[0], mat[4], tmp[2], queue[0], mem[2], 0, NULL, NULL);
 
    /* CDF */
    mul(tmp[1], mat[5], tmp[3], queue[1], mem[3], 0, NULL, NULL);
 
    status = clEnqueueReadBuffer(queue[0], mem[2][2], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, tmp[2], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
    status = clEnqueueReadBuffer(queue[1], mem[3][2], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, tmp[3], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* X */
    add(tmp[0], tmp[1], ans[0], queue[0], mem[4], 0, NULL, NULL);
 
    /* Y */
    add(tmp[2], tmp[3], ans[1], queue[1], mem[5], 0, NULL, NULL);
 
    /* Read X and Y */
    status = clEnqueueReadBuffer(queue[0], mem[4][2], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, ans[0], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
    status = clEnqueueReadBuffer(queue[1], mem[5][2], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, ans[1], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    printf("%u\n", signature(N, ans[0]));
    printf("%u\n", signature(N, ans[1]));
}
 
void OpenCLCleanUp() {
    /* Cleanup */
    clReleaseContext(context);
    for (int i = 0; i < NGPU; i++) {
        clReleaseCommandQueue(queue[i]);
    }
    clReleaseProgram(program);
    clReleaseKernel(kernel_mul);
    clReleaseKernel(kernel_add);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            clReleaseMemObject(mem[i][j]);
        }
    }
}
 
int main(int argc, char **argv) {
    OpenCLInit();
    /* Get input */
    while(scanf("%d\n", &N) != EOF) {
        for (int i = 0; i < 6; i++) {
            scanf("%u", &key[i]);
            rand_gen(key[i], N, mat[i]);
        }
        OpenCLExecute();
    }
    OpenCLCleanUp();
    return 0;
}
