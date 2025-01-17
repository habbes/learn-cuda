#include <stdio.h>

// Regular function that will run on the CPU
void helloCPU()
{
    printf("Hello from the CPU.\n");
}

// Functions that should run on the GPU are declared with the
// __global__ qualifier. This indicates that this function
// can be called "globabally" from either the host (CPU)
// or the device (GPU). They are called kernels.
// void return types is required for __global__ functions.
__global__ void helloGPU()
{
    printf("Hello from the GPU.\n");
}

int main()
{
    helloCPU();

    // Here were are invoking a kernel from CPU code.
    // The <<<...>>> is used to specify thread configuration
    // that will determine how the kernel will be parallelized on the GPU.
    helloGPU<<<1, 1>>>();
    // The GPU kernel is execute asynchronously with host code.
    // This means that the CPU will not wait for the GPU kernel to
    // finish before moving to the next line of code. If we want
    // to wait for the GPU kernel to finish before proceeding, we can use cudaDeviceSynchronize().
    // If we omit this, the program may exit before the GPU kernel has a chance to finish 
    // and we may not see the output from the GPU kernel.
    cudaDeviceSynchronize();

    return 0;
}