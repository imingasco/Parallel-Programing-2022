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
 
int N;
UINT key[6];
UINT mat[6][MAXN][MAXN];
UINT tmp[4][MAXN][MAXN];
UINT ans[2][MAXN][MAXN];
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel_mul, kernel_add;
cl_mem rdmem[6], rwmem[4], wrmem[2];
 
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
 
inline size_t min(size_t a, size_t b) {
    return (a < b)? a : b;
}
 
size_t get_block_size(int N) {
    size_t start = min(MAXSIZE, N - 1);
    for (int i = start; i > 0; i--) {
        if (N % i == 0) {
            return i;
        }
    }
    return 1;
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
    FILE *fp = fopen("matrix-lib.cl", "r");
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
    kernel_mul = clCreateKernel(program, "mul", &status);
    assert(status == CL_SUCCESS);
    kernel_add = clCreateKernel(program, "add", &status);
    assert(status == CL_SUCCESS);
 
    /* Buffer */
    for (int i = 0; i < 6; i++) {
        rdmem[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, mat[i], &status);
        assert(status == CL_SUCCESS);
    }
 
    for (int i = 0; i < 4; i++) {
        rwmem[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, tmp[i], &status);
        assert(status == CL_SUCCESS);
    }
 
    for (int i = 0; i < 2; i++) {
        wrmem[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, ans[i], &status);
        assert(status == CL_SUCCESS);
    }
}
 
void OpenCLExecute() {
    cl_int status;
 
    /* NDRange setup */
    size_t block_size = get_block_size(N);
    size_t global_size[] = {(size_t) N, (size_t) N};
    size_t local_size[] = {block_size, block_size};
 
    /* AB */
    status = clSetKernelArg(kernel_mul, 0, sizeof(cl_mem), (void *) &rdmem[0]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void *) &rdmem[1]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void *) &rwmem[0]);
    assert(status == CL_SUCCESS);
    status = clEnqueueNDRangeKernel(queue, kernel_mul, 2, NULL, global_size, local_size, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* CD */
    status = clSetKernelArg(kernel_mul, 0, sizeof(cl_mem), (void *) &rdmem[2]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void *) &rdmem[3]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void *) &rwmem[1]);
    assert(status == CL_SUCCESS);
    status = clEnqueueNDRangeKernel(queue, kernel_mul, 2, NULL, global_size, local_size, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* Read AB and CD */
    status = clEnqueueReadBuffer(queue, rwmem[0], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, tmp[0], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
    status = clEnqueueReadBuffer(queue, rwmem[1], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, tmp[1], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* X */
    status = clSetKernelArg(kernel_add, 0, sizeof(cl_mem), (void *) &rwmem[0]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_add, 1, sizeof(cl_mem), (void *) &rwmem[1]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_add, 2, sizeof(cl_mem), (void *) &wrmem[0]);
    assert(status == CL_SUCCESS);
    status = clEnqueueNDRangeKernel(queue, kernel_add, 2, NULL, global_size, local_size, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* Read X */
    status = clEnqueueReadBuffer(queue, wrmem[0], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, ans[0], 0, NULL, NULL);
    printf("%u\n", signature(N, ans[0]));
 
    /* ABE */
    status = clSetKernelArg(kernel_mul, 0, sizeof(cl_mem), (void *) &rwmem[0]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void *) &rdmem[4]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void *) &rwmem[2]);
    assert(status == CL_SUCCESS);
    status = clEnqueueNDRangeKernel(queue, kernel_mul, 2, NULL, global_size, local_size, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* CDF */
    status = clSetKernelArg(kernel_mul, 0, sizeof(cl_mem), (void *) &rwmem[1]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void *) &rdmem[5]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void *) &rwmem[3]);
    assert(status == CL_SUCCESS);
    status = clEnqueueNDRangeKernel(queue, kernel_mul, 2, NULL, global_size, local_size, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* Read ABE and CDF */
    status = clEnqueueReadBuffer(queue, rwmem[2], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, tmp[2], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
    status = clEnqueueReadBuffer(queue, rwmem[3], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, tmp[3], 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* Y */
    status = clSetKernelArg(kernel_add, 0, sizeof(cl_mem), (void *) &rwmem[2]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_add, 1, sizeof(cl_mem), (void *) &rwmem[3]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel_add, 2, sizeof(cl_mem), (void *) &wrmem[1]);
    assert(status == CL_SUCCESS);
    status = clEnqueueNDRangeKernel(queue, kernel_add, 2, NULL, global_size, local_size, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
 
    /* Read Y */
    status = clEnqueueReadBuffer(queue, wrmem[1], CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, ans[1], 0, NULL, NULL);
    printf("%u\n", signature(N, ans[1]));
}
 
void OpenCLCleanUp() {
    /* Cleanup */
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel_mul);
    clReleaseKernel(kernel_add);
    for (int i = 0; i < 6; i++) {
        clReleaseMemObject(rdmem[i]);
    }
    for (int i = 0; i < 4; i++) {
        clReleaseMemObject(rwmem[i]);
    }
    clReleaseMemObject(wrmem[0]);
    clReleaseMemObject(wrmem[1]);
}
 
int main(int argc, char **argv) {
    /* Get input */
    scanf("%d\n", &N);
    for (int i = 0; i < 6; i++) {
        scanf("%u", &key[i]);
        rand_gen(key[i], N, mat[i]);
    }
    OpenCLInit();
    OpenCLExecute();
    OpenCLCleanUp();
    return 0;
}
