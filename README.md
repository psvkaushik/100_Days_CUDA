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

**Results :**

<img width="181" alt="image" src="https://github.com/user-attachments/assets/dcf07616-d840-4fbd-b69a-691ffd27bfb0" />




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

**Results :**

<img width="157" alt="image" src="https://github.com/user-attachments/assets/0c60b965-0612-4373-b766-b38f91806c13" />

**Learned:**  
- Utilizing constant memory for better optimization  

### Reading:  
- Read **half of Chapter 7** of the PMPP book.  


---

## Day 4
### File: `tile_matmul.cu`
**Summary:**  
Implemented the tiled version of matrix multiplication.
**Results:**

<img width="160" alt="image" src="https://github.com/user-attachments/assets/882fc46e-0870-45e0-8397-4a7415efc686" />


**Learned:**  
- The concept of shared memory in GPU architecture amd utilizing the concept of tiling in parallel programming.
- How to use shared memory and constant memory effectively in matmul operations.
  

### Reading:  
- Finished **Chapter 5** of the PMPP book.  
  - Learned about different types of memory in GPU such as shared, constant,etc and also boundary checks in tiling operations.

---
## Day 5
### File: `tile_conv.cu`
**Summary:**  
Implemented the tiled version of 2D convolution.
**Results:**

```
CPU Time : 73059.2 ms
GPU Time : 1544.33 ms
GPU Time Tiled: 1755.01 ms
Matrices match!
Conv Matrices match!
```
### Reading
- Finished second part of **Chapter 7** of the PMPP book where the tiled convolution is discussed.
---
## Day 6
### File: `tile_stencil3D.cu`
**Summary:**  
Implemented the basic tiled version of stencil.
**Results:**

```
Max threads per block: 1024
Shared mem per block: 49152 bytes
CPU time: 0.223968 seconds
GPU time: 0.00741968 seconds
CPU and GPU results match!
```
**Learned:**  
- The concept of stencil, and how it is useful in calculating the gradients.
- THE IMPORTANCE OF KNOWING YOUR HARDWARE. Didn't check the maximum threads per block my gpu can accomodate, so was facing issues with halo overwriting as the launch kernel had less threads than what I assumed it would.
### Reading
- Finished first part of **Chapter 8** of the PMPP book where the tiled stencil 3D is discussed.
---
## Day 7
### File: `reduction.cu`
**Summary:**  
Implemented the basic version of reduction.

**Learned:**  
- The concept of tree based reduction, and how it is used in various applications.
### Reading
- Finished first part of **Chapter 10** of the PMPP book where reduction is discussed.
---
## Day 8
### File: `reduction_eff.cu`
**Summary:**  
Implemented a more efficient version of reduction using shared memory.
**Results**
```
Max threads per block: 1024
Shared mem per block: 49152 bytes
CPU time: 0.0271867 seconds
CPU result: 1.67771e+07
GPU time: 0.00112141 seconds
GPU result: 1.67773e+07
```

### Reading
- Finished second half(10.3 and later) of **Chapter 10** of the PMPP book where reduction is discussed.
---
## Day 9
### File: `tc_red.cu`
**Summary:**  
Implemented an even more efficent version of reduction by utilizing thread coarsening, and by decreasing the number of threads launched acheived remarkable speed up compared to yesterday's results.

**Results**
```
Max threads per block: 1024
Shared mem per block: 49152 bytes
CPU time: 0.027211 seconds
CPU result: 1.67771e+07
GPU time: 0.00051667 seconds
GPU result: 1.67773e+07

```
**Learned:**  
- The concept of thread coarsening and its application to reduction.
- Efficiently launching threads based on the coarsening factor to reduce the number of idle threads
### Reading
- Finished  **Chapter 10** of the PMPP book where reduction using thread coarsening is discussed.
---
## Day 10
### File: `softmax.cu`
**Summary** 

Implemented a kernel to perform Safe Softmax.

**Results**

``` Work in progress to get a good benchmarking code ```

---
## Day 11
### File: `online_softmax.cu`
**Summary** 

Implemented a kernel to perform Online Softmax.

**Results**

``` Work in progress to get a good benchmarking code ```

**Videos**
- Watched Umar's video on Flash Attention from scratch to get an understanding of how Online Softmax is faster than the plain one.
---
<!--
**Learned:**  
- How to calculate mean and variance in parallel using reduction algorithms.
- Strategies to stabilize floating-point operations to prevent overflow or underflow issues.
- CUDA kernel optimization for workloads involving tensor computation.  

### Reading:  
- Read **Chapter 4** of the PMPP book.  
  -  Learned about memory optimizations and strategies for GPU performance tuning.
---
# Future Work
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

