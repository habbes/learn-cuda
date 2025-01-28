#include <stdio.h>

cudaError_t checkCudaError(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
    }
    return result;
}

void initWith(float num, float *a, int N)
{
    for (int i = 0; i < N; ++i)
    {
        a[i] = num;
    }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < N; i += stride)
    {
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float *array, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (array[i] != target)
        {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main()
{
    const int N = 2 << 20;
    size_t size = N * sizeof(float);

    float *a;
    float *b;
    float *c;

    checkCudaError(cudaMallocManaged(&a, size));
    checkCudaError(cudaMallocManaged(&b, size));
    checkCudaError(cudaMallocManaged(&c, size));

    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    size_t threads_per_block = 256;
    size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

    addVectorsInto<<<number_of_blocks, threads_per_block>>>(c, a, b, N);
    checkCudaError(cudaGetLastError());

    checkCudaError(cudaDeviceSynchronize());

    checkElementsAre(7, c, N);

    checkCudaError(cudaFree(a));
    checkCudaError(cudaFree(b));
    checkCudaError(cudaFree(c));
}
