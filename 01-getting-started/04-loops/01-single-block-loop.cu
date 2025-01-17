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
    // each thread will execute a different iteration. We
    // use "threadIdx.x" to indicate the "iteration" we are on.
    printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
    int N = 10;
    printf("CPU Loop:\n");
    cpuLoop(N);

    printf("\nGPU Loop:\n");
    gpuLoop<<<1, N>>>();
    // I noticed that the out from the GPU loop is in the
    // same ascending order as the CPU even after running multiple times.
    // I expected the order to be random due the parallism. I wonder
    // whether this is a coincidence or there is a reason for this.
    // I will test with larger N and/or different machines to see if the result
    // is consistent.
    cudaDeviceSynchronize();
}