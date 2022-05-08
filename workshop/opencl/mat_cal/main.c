#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define UINT uint32_t
#define MAXN 1024

int N, key[6];
UINT A[MAXN][MAXN], B[MAXN][MAXN], C[MAXN][MAXN], D[MAXN][MAXN], E[MAXN][MAXN], F[MAXN][MAXN];
UINT AB[MAXN][MAXN], CD[MAXN][MAXN];
UINT ABE[MAXN][MAXN], CDF[MAXN][MAXN];
UINT X[MAXN][MAXN], Y[MAXN][MAXN];
char source[2048];

cl_context context;
cl_program program;
cl_kernel kernel_mul, kernel_add;
cl_command_queue queue;
cl_mem memA, memB, memC, memD, memE, memF;
cl_mem memAB, memCD;
cl_mem memABE, memCDF;
cl_mem memX, memY;

void rand_gen(UINT c, int N, UINT A[][MAXN]) {
    UINT x = 2, n = N * N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j) % n;
            A[i][j] = x;
        }
    }
}

void rand_gen_transpose(UINT c, int N, UINT A[][MAXN]) {
    UINT x = 2, n = N * N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j) % n;
            A[j][i] = x;
        }
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
    FILE *fp = fopen("matrix-lib.cl", "r");
    size_t source_len = fread(source, sizeof(char), 2048, fp);
    const char *source_ptr = source;
    cl_int status;
    cl_uint platform_n, device_n;
    cl_platform_id platform;
    cl_device_id device;
    const cl_queue_properties queue_property = CL_QUEUE_PROFILING_ENABLE;

    /* Get platform and devices */
    clGetPlatformIDs(1, &platform, &platform_n);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &device_n);

    /* Create context and command queue */
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    queue = clCreateCommandQueueWithProperties(context, device, &queue_property, &status);

    /* Build program and create kernel */
    program = clCreateProgramWithSource(context, 1, &source_ptr, &source_len, &status);
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel_mul = clCreateKernel(program, "mul", &status);
    kernel_add = clCreateKernel(program, "add", &status);

    /* Create buffer */
    memA = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, A, &status);
    memB = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, B, &status);
    memC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, C, &status);
    memD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, D, &status);
    memE = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, E, &status);
    memF = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, F, &status);
    memX = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, X, &status);
    memY = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, Y, &status);
    memAB = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, AB, &status);
    memCD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, CD, &status);
    memABE = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, ABE, &status);
    memCDF = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, CDF, &status);
}

void OpenCLExecute() {
    cl_int status;
    size_t globalSize[] = {N, N};
    size_t localSize[] = {32, 32};

    /* AB */
    status = clSetKernelArg(kernel_mul, 0, sizeof(cl_int), (void *) &N);
    status = clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void *) &memA);
    status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void *) &memB);
    status = clSetKernelArg(kernel_mul, 3, sizeof(cl_mem), (void *) &memAB);
    status = clEnqueueNDRangeKernel(queue, kernel_mul, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    /* CD */
    status = clSetKernelArg(kernel_mul, 0, sizeof(cl_int), (void *) &N);
    status = clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void *) &memC);
    status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void *) &memD);
    status = clSetKernelArg(kernel_mul, 3, sizeof(cl_mem), (void *) &memCD);
    status = clEnqueueNDRangeKernel(queue, kernel_mul, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    /* X */
    status = clSetKernelArg(kernel_add, 0, sizeof(cl_mem), (void *) &memAB);
    status = clSetKernelArg(kernel_add, 1, sizeof(cl_mem), (void *) &memCD);
    status = clSetKernelArg(kernel_add, 2, sizeof(cl_mem), (void *) &memX);
    status = clEnqueueNDRangeKernel(queue, kernel_add, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    /* ABE */
    status = clSetKernelArg(kernel_mul, 0, sizeof(cl_int), (void *) &N);
    status = clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void *) &memAB);
    status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void *) &memE);
    status = clSetKernelArg(kernel_mul, 3, sizeof(cl_mem), (void *) &memABE);
    status = clEnqueueNDRangeKernel(queue, kernel_mul, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    /* CDF */
    status = clSetKernelArg(kernel_mul, 0, sizeof(cl_int), (void *) &N);
    status = clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void *) &memCD);
    status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void *) &memF);
    status = clSetKernelArg(kernel_mul, 3, sizeof(cl_mem), (void *) &memCDF);
    status = clEnqueueNDRangeKernel(queue, kernel_mul, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    /* Y */
    status = clSetKernelArg(kernel_add, 0, sizeof(cl_mem), (void *) &memABE);
    status = clSetKernelArg(kernel_add, 1, sizeof(cl_mem), (void *) &memCDF);
    status = clSetKernelArg(kernel_add, 2, sizeof(cl_mem), (void *) &memY);
    status = clEnqueueNDRangeKernel(queue, kernel_add, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    clFinish(queue);
    printf("%u\n", signature(N, X));
    printf("%u\n", signature(N, Y));
}

void OpenCLRelease() {
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel_mul);
    clReleaseKernel(kernel_add);
    clReleaseMemObject(memA);
    clReleaseMemObject(memB);
    clReleaseMemObject(memC);
    clReleaseMemObject(memD);
    clReleaseMemObject(memE);
    clReleaseMemObject(memF);
    clReleaseMemObject(memX);
    clReleaseMemObject(memY);
    clReleaseMemObject(memAB);
    clReleaseMemObject(memCD);
    clReleaseMemObject(memABE);
    clReleaseMemObject(memCDF);
}

int main(int argc, char **argv) {
    scanf("%d", &N);
    scanf("%u %u %u %u %u %u", &key[0], &key[1], &key[2], &key[3], &key[4], &key[5]);
    rand_gen(key[0], N, A);
    rand_gen(key[2], N, C);
    rand_gen_transpose(key[1], N, B);
    rand_gen_transpose(key[3], N, D);
    rand_gen_transpose(key[4], N, E);
    rand_gen_transpose(key[5], N, F);
    OpenCLInit();
    OpenCLExecute();
    OpenCLRelease();
    return 0;
}
