# Learning CUDA

Based on the course [Getting Started with Accelerated Computing in CUDA C/C++](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-04+V1) from NVDIA Deep Learning Institute.

Use the `nvidia-smi` command to get info about the NVIDIA GPUs on your machine:

```sh
nvidia-smi
```

We compile CUDA-accelerated programs using the [NVIDIA CUDA Compiler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) `nvcc`:

```sh
nvcc -o bin/out some-CUDA.cu -run
```

where:

- `nvcc` is the command line command for using the nvcc compiler.
- `some-CUDA.cu` is passed as the file to compile (`.cu` is the file extension for CUDA-accelerated programs).
- The `o` flag is used to specify the output file for the compiled program.
- As a matter of convenience, providing the `run` flag will execute the successfully compiled binary.

## Installing the compiler toolchain

- [Installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

If using VS Code, the [Nsight Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition) extension can provide language support features (auto-complete, debugging, etc.).

## Troubleshooting

### Getting the error `nvcc fatal: Cannot find compiler 'cl.exe' in PATH` on Windows

This means `nvcc` could not find the Microsoft C++ compiler. Ensure you have Visual Studio installed with **Desktop Development with C++** workload.

Consider using the **Developer Powershell for VS 2022** or **Developer Command Prompt for VS 2022** which usually load with relevant commands accessible.

You can also try to manually update PATH to include the directory containing the `cl.exe` command (e.g. `C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64`).

### Getting an access violation error on Windows related to `cudafe++`

Try to run in Adminstrator mode
