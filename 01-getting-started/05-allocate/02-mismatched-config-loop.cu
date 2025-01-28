#include <stdio.h>

__global__ void initializeElementsTo(int initialValue, int *a, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Since we be running in more threads than data elements,
    // we use this condition to ensure we don't access the array
    // out of bounds. This means that some threads won't do any meaningful work.
    if (i < N)
    {
        a[i] = initialValue;
    }
}

int main()
{
    int N = 1000;
    int *a;
    size_t size = N * sizeof(int);

    cudaMallocManaged(&a, size);

    // Assume the desired block size is known to be exactly 256 threads.
    // Note that due to hardware traits, blocks that contain number of threads
    // that is a multiple of 32 are often desirable for performance reasons.
    size_t threads_per_block = 256;

    // Since we can't get a block and threads configuration that exactly
    // matches N when the number of threads is fixed,
    // We use this formula to ensure we have at least N threads in the grid
    // and at most 1 extra block of "unused" threads.
    size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

    int initialValue = 10;
    initializeElementsTo<<<number_of_blocks, threads_per_block>>>(initialValue, a, N);
    cudaDeviceSynchronize();

    // verify
    for (int i = 0; i < N; i++)
    {
        if (a[i] != 10)
        {
            printf("FAILURE: Expected element %d to be %d, but got %d\n", i, initialValue, a[i]);
            cudaFree(a);
            exit(1);
        }
    }

    printf("SUCCESS!\n");

    cudaFree(a);
}