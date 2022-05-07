#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <CL/cl.h>
#include "utils.h"
#define MAXGPU 1
#define MAXN 16777216
#define GPULOCAL 256
uint32_t vec[MAXN/GPULOCAL];
int N;
uint32_t keyA, keyB;
char source[1024] = "";
// -- start working with OpenCL
cl_context context;
cl_program prog;
cl_kernel kernel;
cl_command_queue queue;
cl_mem mem;

int initAllGPU() {
    // -- generate kernel code
    FILE *fp = fopen("vecdot.cl", "r");
    size_t source_len = fread(source, 1, 1024, fp);
    cl_int status;
    cl_uint platform_n, device_n;
    cl_platform_id platform;
    cl_device_id device;
    const char *src_ptr = source;
    // -- basic OpenCL setup
    clGetPlatformIDs(1, &platform, &platform_n);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &status);
    prog = clCreateProgramWithSource(context, 1, &src_ptr, &source_len, &status);
    status = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    assert(status == CL_SUCCESS);
    kernel = clCreateKernel(prog, "vecdot", &status);
    // -- create all buffers
    cl_mem_flags flag = CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR;
    mem = clCreateBuffer(context, flag, sizeof(uint32_t) * (MAXN / GPULOCAL), vec, &status);
    return 1;
}

int executeGPU() {
    uint32_t padding = 0;
    while (N % GPULOCAL) {
        padding += encrypt(N, keyA) * encrypt(N, keyB);
        N++;
    }
    cl_int clStat;
    size_t globalOffset[] = {0};
    size_t globalSize[] = {N};
    size_t localSize[] = {GPULOCAL};
    
    // -- set argument to kernel
    clStat = clSetKernelArg(kernel, 0, sizeof(cl_uint), (void *) &keyA);
    clStat = clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *) &keyB);
    clStat = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &mem);
    // -- execute
    clStat = clEnqueueNDRangeKernel(queue, kernel, 1, globalOffset,
            globalSize, localSize, 0, NULL, NULL);
    
    // -- read back
    clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, sizeof(uint32_t) * (N / GPULOCAL), vec, 0, NULL, NULL);
    // clFinish(queue);
    uint32_t sum = 0;
    for (int i = 0; i < N / GPULOCAL; i++)
        sum += vec[i];
    printf("%u\n", sum - padding);
    return 1;
}

void onStart() {
    initAllGPU("vecdot.cl");
    while (scanf("%d %u %u", &N, &keyA, &keyB) == 3) {
        executeGPU();
    }
}

int main(int argc, char *argv[]) {
    onStart();
    return 0;
}
