#include <stdio.h>

void cpuLoop(int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("This is iteration number %d\n", i);
    }
}

__global__ void gpuLoop()
{
    // This kernel executes a single iteration of the original loop,
    // each thread will execute a different iteration.
    // Since there's a limit to the number of threads in a block (1024),
    // to achieve massive parallelism, we need to coordinate threads on multiple
    // blocks.
    // In this example, we use the following built-in variables to determine
    // the unique index that each thread should operate on:
    // - threadIdx.x: the index of the thread in the block
    // - blockIdx.x: the index of the block in the grid
    // - blockDim.x: the number of threads in a block

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("This is iteration number %d\n", i);
}

int main()
{
    printf("CPU Loop:\n");
    cpuLoop(10);

    printf("\nGPU Loop:\n");
    gpuLoop<<<2, 5>>>();
    cudaDeviceSynchronize();
}