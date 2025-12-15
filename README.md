
## Project Overview

The goal of this mini project is to implement **general matrix multiplication (SGEMM)** entirely from scratch using **CUDA C++**, and to progressively optimize it to approach the performance of **NVIDIA’s cuBLAS SGEMM**.

Rather than jumping straight to complex optimizations, the project is built step by step, starting from a simple and readable baseline and gradually introducing performance improvements. Each stage focuses on understanding *why* an optimization matters, not just how to apply it.

Throughout the project, I explore and explain:

### CUDA Runtime API
- Kernel launches  
- Memory allocation and transfers  
- Synchronization and error handling  

### NVIDIA GPU Architecture
- CUDA cores and Streaming Multiprocessors (SMs)  
- Memory hierarchy (global, shared, registers)  
- How hardware characteristics influence performance  

### Core GPU Programming Concepts
- Global memory coalescing  
- 2D block tiling for matrix multiplication  
- 1D and 2D thread tiling strategies  
- Vectorized memory accesses for improved throughput  

The end goal is not to replace cuBLAS, but to understand how high-performance GPU GEMM kernels are built, and how close a custom implementation can get when guided by hardware-aware optimizations.
