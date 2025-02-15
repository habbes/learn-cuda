#include <stdio.h>

// It can be helpful to create a macro that wraps cuda calls and checks for errors.
// Here's an example

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    }
    return result;
}

void init(int *a, int N)
{
    int i;
    for (i = 0; i < N; ++i)
    {
        a[i] = i;
    }
}

__global__ void doubleElements(int *a, int N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N + stride; i += stride)
    {
        a[i] *= 2;
    }
}

bool checkElementsAreDoubled(int *a, int N)
{
    int i;
    for (i = 0; i < N; ++i)
    {
        if (a[i] != i * 2)
            return false;
    }
    return true;
}

int main()
{
    /*
     * Add error handling to this source code to learn what errors
     * exist, and then correct them. Googling error messages may be
     * of service if actions for resolving them are not clear to you.
     */

    int N = 10000;
    int *a;

    size_t size = N * sizeof(int);

    // Many cuda functions return an cudaError_t value for error checking
    checkCuda(cudaMallocManaged(&a, size));

    init(a, N);

    size_t threads_per_block = 2048;
    size_t number_of_blocks = 32;

    doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
    // Kernel functions return void, so we use the cudaGetLastError function
    // to check for errors, e.g. due to invalid configuration
    checkCuda(cudaGetLastError());

    checkCuda(cudaDeviceSynchronize());

    bool areDoubled = checkElementsAreDoubled(a, N);
    printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

    checkCuda(cudaFree(a));
}
