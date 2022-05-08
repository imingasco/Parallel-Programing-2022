#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define UINT uint32_t
#define MAXN 1024

int N, keyA, keyB;
UINT A[MAXN][MAXN], B[MAXN][MAXN], C[MAXN][MAXN];
char source[2048];

cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_mem memA, memB, memC;

void rand_gen(UINT c, int N, UINT A[][MAXN]) {
    UINT x = 2, n = N * N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j) % n;
            A[i][j] = x;
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
    FILE *fp = fopen("matrixmul.cl", "r");
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
    if (status != CL_SUCCESS) {
        printf("yo\n");
        exit(0);
    }
    kernel = clCreateKernel(program, "mul", &status);

    /* Create buffer */
    memA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, A, &status);
    memB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, B, &status);
    memC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(UINT) * MAXN * MAXN, C, &status);
}

void OpenCLExecute() {
    cl_int status;
    size_t globalSize[] = {N, N};
    size_t localSize[] = {64, 64};

    /* Set kernel args */
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *) &N);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &memA);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &memB);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &memC);
    assert(status == CL_SUCCESS);

    /* Enqueue kernel */
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    /* Read and output the result */
    status = clEnqueueReadBuffer(queue, memC, CL_TRUE, 0, sizeof(UINT) * MAXN * MAXN, C, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
    printf("%u\n", signature(N, C));
}

void OpenCLRelease() {
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseMemObject(memA);
    clReleaseMemObject(memB);
    clReleaseMemObject(memC);
}

int main(int argc, char **argv) {
    OpenCLInit();
    scanf("%d %u %u", &N, &keyA, &keyB);
    rand_gen(keyA, N, A);
    rand_gen(keyB, N, B);
    OpenCLExecute();
    OpenCLRelease();
    return 0;
}
