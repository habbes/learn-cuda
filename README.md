# Learning CUDA

Based on the course [Getting Started with Accelerated Computing in CUDA C/C++](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-04+V1) from NVDIA Deep Learning Institute.

Use the `nvidia-smi` command to get info about the NVIDIA GPUs on your machine:

```sh
nvidia-smi
```

We compile CUDA-accelerated programs using the [NVIDIA CUDA Compiler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) `nvcc`:

```sh
nvcc -o out some-CUDA.cu -run
```

where:

- `nvcc` is the command line command for using the nvcc compiler.
- `some-CUDA.cu` is passed as the file to compile (`.cu` is the file extension for CUDA-accelerated programs).
- The `o` flag is used to specify the output file for the compiled program.
- As a matter of convenience, providing the `run` flag will execute the successfully compiled binary.
