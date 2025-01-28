# 100_Days_CUDA
Learning Cuda One Day at a Time

Template Inspiration : https://github.com/1y33/100Days

### Project Progress by Day
| Day   | Files & Summaries                                                                                                                                                                                                                          |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| day1  | **vec_add.cu**: GPU vector addition; basics of memory allocation/host-device transfer.                                                                 |
<!-- | day2  | **function.cu**: Use `__device__` function in kernel; per-thread calculations.                                                                                                                                                       |
| day3  | **addMatrix.cu**: 2D matrix addition; map row/column indices to threads.<br>**anotherMatrix.cu**: Transform matrices with custom function; 2D index operations.                                                                       |
| day4  | **layerNorm.cu**: Layer normalization using shared memory; mean/variance computation.                                                                                                                                                |
| day5  | **vectorSumTricks.cu**: Parallel vector sum via reduction; shared memory optimizations.                                                                                                                                               |
| day6  | **SMBlocks.cu**: Retrieve SM ID per thread via inline PTX.<br>**SoftMax.cu**: Shared-memory softmax; split exponent/normalization steps.<br>**TransposeMatrix.cu**: Matrix transpose via index swapping.<br>**ImportingToPython/rollcall.cu**: Python-CUDA integration.<br>**AdditionKernel/additionKernel.cu**: Modify PyTorch tensors in CUDA. |
| day7  | **naive.cu**: Naive matrix multiplication.<br>**matmul.cu**: Tiled matmul with shared memory.<br>**conv1d.cu**: 1D convolution with shared memory.<br>**pythontest.py**: Validate custom convolution against PyTorch.                               |
| day8  | **pmpbook/chapter3matvecmul.cu**: Matrix-vector multiplication.<br>**pmpbook/chapter3ex.cu**: Benchmarks different matrix add kernels.<br>**pmpbook/deviceinfo.cu**: Prints device properties.<br>**pmpbook/color2gray.cu**: Convert RGB to grayscale.<br>**pmpbook/vecaddition.cu**: Another vector addition example.<br>**pmpbook/imageblur.cu**: Simple image blur.<br>**selfAttention/selfAttention.cu**: Self-attention kernel with online softmax. |
| day9  | **flashAttentionFromTut.cu**: Minimal Flash Attention kernel with shared memory tiling.<br>**bind.cpp**: Torch C++ extension bindings for Flash Attention.<br>**test.py**: Tests the minimal Flash Attention kernel against a manual softmax-based attention for comparison. |
| day10 | **ppmbook/matrixmul.cu**: Matrix multiplication using CUDA.<br>**setup.py**: Torch extension build script for CUDA code (FlashAttention).<br>**FlashAttention.cu**: Example Flash Attention CUDA kernel.<br>**FlashAttention.cpp**: Torch bindings for the Flash Attention kernel.<br>**test.py**: Manual vs. CUDA-based attention test.<br>**linking/test.py**: Builds simple CUDA kernel for testing linking.<br>**linking/simpleKernel.cpp**: Torch extension binding for a simple CUDA kernel.<br>**linking/simpleKernel.cu**: Simple CUDA kernel that increments a tensor. |
| day11 | **FlashTestPytorch/**: Custom Flash Attention in PyTorch, tests and benchmarks.<br>**testbackward.py**: Gradient comparison between custom CUDA kernels and PyTorch. |
| day12 | **softMax.cu**: Additional softmax kernel with shared memory optimization.<br>**NN/kernels.cu**: Tiled kernel implementation and layer initialization.<br>**tileMatrix.cu**: Demonstrates tile-based matrix operations. |

| nvidiadocs | **addition.cu**: 1D/2D vector/matrix addition examples.       -->