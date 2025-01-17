#include <stdio.h>

__global__ void firstParallel()
{
    printf("This should be running in parallel.\n");
}

int main()
{
    // The <<<...>>> is referred to as the execution configuration.
    // It specifies the number of thread blocks and the number of threads per block
    // that will be used to execute the kernel.
    // Tasks on the GPU are executed on many threads in parallel.
    // A logical grouping of threads is called a thread block.
    // A collection of blocks is called a grid.
    // Each block in a grid has the same number of threads.
    // In this example, the following kernel will run on 4 blocks with 5 threads each,
    // all in parallel. It will be executed 4 * 5 = 20 times. All the theads will execute the same code.
    firstParallel<<<4, 5>>>();
    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    return 0;
}