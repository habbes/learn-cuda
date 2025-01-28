#include <stdio.h>

void init(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
    }
}

__global__ void doubleElements(int *a, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // This is the total number of threads in the grid.
    int gridStride = gridDim.x * blockDim.x;

    // Since there might be more data elements than thread,
    // each thread will process more than one data element.
    for (; i < N; i += gridStride)
    {
        a[i] *= 2;
    }
}

bool areAllDoubled(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (a[i] != i * 2)
        {
            return false;
        }
    }

    return true;
}

int main()
{
    int N = 10000;
    int *a;
    size_t size = N * sizeof(int);

    cudaMallocManaged(&a, size);

    size_t threads_per_block = 256;
    size_t number_of_blocks = 32;

    init(a, N);

    doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);

    cudaDeviceSynchronize();

    bool success = areAllDoubled(a, N);

    if (success)
    {
        printf("SUCCESS!\n");
    }
    else
    {
        printf("FAILURE! Not all elements were double.\n");
    }

    cudaFree(a);
}