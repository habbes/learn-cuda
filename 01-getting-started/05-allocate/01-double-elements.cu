#include <stdio.h>

// Initializes the array with values.
// This will run on the host (CPU).
void initArray(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
    }
}

// Double the elements in parallel. This will run on the GPU.
__global__ void doubleElements(int *a, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // There might be more threads than elements in the array,
    // this condition ensures we don't go out of bounds.
    if (i < N)
    {
        a[i] *= 2;
    }
}

// Verify whether the elements were correctly doubled.
// This will run on the CPU.
bool verifyElementsAreDoubled(int *a, int N)
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
    int N = 100;
    int *a; // pointer to the array
    size_t size = N * sizeof(int);

    // This allocates memory that can be access both from the CPU and GPU.
    // We pass a pointer to the pointer so that we can modify the pointer
    // to refer to the allocated memory.
    // This a convenient way to share memory between the CPU and GPU that
    // requires no developer overhead and still allows us to get decent
    // performance improvements over CPU-only code. But there are
    // more sophisticated memory management techniques that can lead
    // to better performance. See: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations
    cudaMallocManaged(&a, size);

    initArray(a, N);

    size_t threads_per_block = 10;
    size_t number_of_blocks = 10;

    doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
    cudaDeviceSynchronize();

    bool isCorrect = verifyElementsAreDoubled(a, N);

    printf("Are elements correctly doubled? %s\n", isCorrect ? "Yes" : "No");

    // Frees the memory allocated with cudaMallocManaged.
    cudaFree(a);
}