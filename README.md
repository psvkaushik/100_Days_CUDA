# 100_Days_CUDA
Learning Cuda One Day at a Time

Template Inspiration : [1y33](https://github.com/1y33/100Days) and [a-hamdi](https://github.com/a-hamdi/cuda)

# CUDA 100 days Learning Journey

This document serves as a log of the progress and knowledge I gained while working on CUDA programming and studying the **PMPP (Parallel Programming and Optimization)** book.

Mentor: [Umar](https://github.com/hkproj/)


---

## Day 1
### File: `vecadd.cu`
**Summary:**  
Implemented vector addition by writing a simple CUDA program. Explored how to launch a kernel to perform a parallelized addition of two arrays, where each thread computes the sum of a pair of values.  

**Learned:**  
- Basics of writing a CUDA kernel.
- Understanding of grid, block, and thread hierarchy in CUDA.  
- How to allocate and manage device (GPU) memory using `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.  

### Reading:  
- Read **Chapter 1&2** of the PMPP book.  
  - Learned about the fundamentals of parallel programming, CUDA architecture, and the GPU execution model, Vector Addition and most importantly compiling CUDA code.

---

## Day 2
### File: `vecmul.cu`
**Summary:**  
Worked on matrix multiplication using CUDA. Designed the grid and block layout to handle 2D matrices in parallel, with each element processed by an individual thread.  

**Learned:**  
- How to map 2D matrix data onto multiple threads.
- Understanding thread indexing in 2D grids and blocks using `threadIdx`, `blockIdx`, `blockDim`, and `gridDim`.  
- Synchronizing threads and avoiding race conditions when writing results to an output matrix.  

### Reading:  
- Read **Chapter 3&4** of the PMPP book.  
  - Learned about scalability of GPUs, massive parallelism, and how to configure problem data to match GPU thread hierarchies.
  - Learned about performance bottlenecks such as resource partitioning and occupancy, and also about the intricacies of GPUs such as warps, latency hiding.
  

---

## Day 3
### File: `conv.cu`
**Summary:**  
Implemented 2D convolution using CUDA. Optimized performance using constant memory.  

**Learned:**  
- Utilizing constant memory for better optimization  

### Reading:  
- Read **half of Chapter 7** of the PMPP book.  


---
# Future Work

## Day 4
### File: `tile_matmul.cu`
**Summary:**  
Work on matrix multiplication but the tiled version.  

<!--**Learned:**  
- The concept of reduction in parallel programming.
- Techniques for minimizing warp divergence and balancing workload across threads.
- How to use shared memory effectively in reduction operations.  

### Reading:  
- Finished **Chapter 3** of the PMPP book.  
  - Learned about Scalable Parallel Execution including Resource Assignment and Thread Scheduling and Latency Tolerance -->
---
## Day 5
### File: `tile_conv.cu`
**Summary:**  
Work on tiled conv.
<!--
**Learned:**  
- How to calculate mean and variance in parallel using reduction algorithms.
- Strategies to stabilize floating-point operations to prevent overflow or underflow issues.
- CUDA kernel optimization for workloads involving tensor computation.  

### Reading:  
- Read **Chapter 4** of the PMPP book.  
  -  Learned about memory optimizations and strategies for GPU performance tuning.
---

## Day 6
### File: `LayerNorm.cu`
**Summary:**  
Implemented Layer Normalization in CUDA, often used in deep learning models. Explored normalization techniques across batches and layers using reduction operations. Addressed the challenge of maintaining numerical stability during computation.  

**Learned:**  
- How to calculate mean and variance in parallel using reduction algorithms.
- Strategies to stabilize floating-point operations to prevent overflow or underflow issues.
- CUDA kernel optimization for workloads involving tensor computation.  

### Reading:  
- Read **Chapter 4** of the PMPP book.  
  -  Learned about memory optimizations and strategies for GPU performance tuning. -->

