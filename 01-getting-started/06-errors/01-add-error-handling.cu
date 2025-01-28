#include <stdio.h>

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
    cudaError_t err;
    err = cudaMallocManaged(&a, size);

    if (err != cudaSuccess)
    {
        printf("Error allocating memory: %s\n", cudaGetErrorString(err));
    }

    init(a, N);

    size_t threads_per_block = 2048;
    size_t number_of_blocks = 32;

    doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
    // Kernel functions return void, so we use the cudaGetLastError function
    // to check for errors, e.g. due to invalid configuration
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error launching kernel: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("Error synchronizing device: %s\n", cudaGetErrorString(err));
    }

    bool areDoubled = checkElementsAreDoubled(a, N);
    printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

    err = cudaFree(a);
    if (err != cudaSuccess)
    {
        printf("Error freeing memory: %s\n", cudaGetErrorString(err));
    }
}
