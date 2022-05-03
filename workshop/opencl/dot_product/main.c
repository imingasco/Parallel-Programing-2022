#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <CL/cl.h>
#include <omp.h>

#define MAXDEVICE 16 
#define MAXN 16777216
#define GROUPSIZE 128

const char filename[16] = "vecdot.cl";
const char kernel_name[16] = "vecdot";
uint32_t vec[MAXN];
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program prog;
cl_kernel kernel;
cl_mem buffer;

int OpenCLInit() {
    cl_device_id devices[MAXDEVICE];
    cl_int status;
    cl_uint platform_num, device_num;

    status = clGetPlatformIDs(1, &platform, &platform_num);

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, MAXDEVICE, devices, &device_num);
    if (status == CL_SUCCESS) {
        FILE *fp = fopen(filename, "r");
        char source[1024] = "";
        const char *src_ptr = source;
        size_t len = fread(source, sizeof(char), 1024, fp);
        device = devices[0];
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
        queue = clCreateCommandQueueWithProperties(context, device, NULL, &status);
        prog = clCreateProgramWithSource(context, 1, &src_ptr, &len, &status);
        status = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
        kernel = clCreateKernel(prog, kernel_name, &status);
        buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint32_t) * MAXN, vec, &status);
        return 0;
    }
    return 1;
}

void OpenCLClean() {
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main(int argc, char **argv) {
    int N;
    uint32_t key1, key2;

    omp_set_num_threads(4);
    OpenCLInit();
    while (scanf("%d %" PRIu32 " %" PRIu32, &N, &key1, &key2) == 3) {
        size_t global_offset[] = {0};
        size_t global_size[] = {N};
        size_t work_item_per_work_group[] = {1};
        clSetKernelArg(kernel, 0, sizeof(cl_uint), (void *) &key1);
        clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *) &key2);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) buffer);
        clEnqueueNDRangeKernel(queue, kernel, 1, global_offset, global_size, work_item_per_work_group, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(uint32_t) * N, vec, 0, NULL, NULL);

        uint32_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
        for (int i = 0; i < N; i++) {
            sum += vec[i];
        }
        printf("%u\n", sum);
    }
    OpenCLClean();
    return 0;
}
