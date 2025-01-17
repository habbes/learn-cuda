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
    cudaDeviceSynchronize();
}