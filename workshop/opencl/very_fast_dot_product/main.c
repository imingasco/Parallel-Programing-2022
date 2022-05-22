#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <CL/cl.h>
#define MAXGPU 1
#define MAXN 67108864
#define GPULOCAL 512
#define LENPERITEM 512
#define LENPERGROUP (LENPERITEM * GPULOCAL)
 
uint32_t vec[MAXN/LENPERGROUP];
int N;
uint32_t keyA, keyB;
char source[2048] = "";
// -- start working with OpenCL
cl_context context;
cl_program prog;
cl_kernel kernel;
cl_command_queue queue;
cl_mem mem;
 
int initAllGPU() {
    // -- generate kernel code
    FILE *fp = fopen("vecdot.cl", "r");
    size_t source_len = fread(source, 1, 2048, fp);
    cl_int status;
    cl_uint platform_n, device_n;
    cl_platform_id platform;
    cl_device_id device;
    const char *src_ptr = source;
    // -- basic OpenCL setup
    status = clGetPlatformIDs(1, &platform, &platform_n);
    assert(status == CL_SUCCESS);
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(status == CL_SUCCESS);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
    cl_queue_properties prop = CL_QUEUE_PROFILING_ENABLE;
    queue = clCreateCommandQueueWithProperties(context, device, &prop, &status);
    assert(status == CL_SUCCESS);
    prog = clCreateProgramWithSource(context, 1, &src_ptr, &source_len, &status);
    assert(status == CL_SUCCESS);
    status = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    assert(status == CL_SUCCESS);
    kernel = clCreateKernel(prog, "vecdot", &status);
    assert(status == CL_SUCCESS);
    // -- create all buffers
    cl_mem_flags flag = CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR;
    mem = clCreateBuffer(context, flag, sizeof(uint32_t) * (MAXN / LENPERGROUP), vec, &status);
    return 1;
}
 
int get_new_N() {
    int modular = GPULOCAL * LENPERITEM;
    return N + modular - (N % modular);
}
 
int executeGPU() {
    cl_int clStat;
    int new_N = get_new_N();
    size_t globalSize[] = {(size_t)new_N / LENPERITEM};
    size_t localSize[] = {GPULOCAL};
 
    // -- set argument to kernel
    clStat = clSetKernelArg(kernel, 0, sizeof(cl_uint), (void *) &keyA);
    clStat = clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *) &keyB);
    clStat = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &mem);
    clStat = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *) &N);
    // -- execute
    clStat = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL);
 
    // -- read back
    clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, sizeof(uint32_t) * (new_N / LENPERGROUP), vec, 0, NULL, NULL);
    uint32_t sum = 0;
    for (int i = 0; i < new_N / LENPERGROUP; i++)
        sum += vec[i];
    printf("%u\n", sum);
    return 1;
}
 
void onStart() {
    initAllGPU();
    while (scanf("%d %u %u", &N, &keyA, &keyB) == 3) {
        executeGPU();
    }
}
 
int main(int argc, char *argv[]) {
    onStart();
    return 0;
}
