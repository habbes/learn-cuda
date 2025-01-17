#include <stdio.h>

__global__ void someKernel()
{
    // Since the same kernel is executed by all threads in a grid, we need
    // a way to identify which thread we are in so that we can distribute the
    // workload efficiently across threads

    // threadIdx.x is a built-in variable provided by CUDA that contains the index of the thread in the block.
    // blockIdx.x is a built-in variable provided by CUDA that contains the index of the block in the grid.
    // blockDim.x is a built-in variable provided by CUDA that contains the number of threads in a block.
    // gridDim.x is a built-in variable provided by CUDA that contains the number of blocks in the grid.

    if (threadIdx.x == 1023 && blockIdx.x == 255)
    {
        printf("Hello from thread 1023 in block 255\n");
    }
}

int main()
{
    // Launch the kernel with 256 blocks and 1024 threads per block
    someKernel<<<256, 1024>>>();
    cudaDeviceSynchronize();

    return 0;
}

