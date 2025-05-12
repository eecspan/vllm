#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cuda_compat.h"
#include "dispatch_utils.h"
#define CEILDIV(x,y) (((x) + (y) - 1) / (y))
#define LOOP_SIZE 1024
#define L2_CACHE_LINE 1
namespace vllm {
    __global__ void l2_preload_kernel(char* ptr, int loop_size, int count) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int PER_THREAD_LOAD_BYTE = L2_CACHE_LINE * loop_size;
    int block_base_offset = blockIdx.x * blockDim.x * PER_THREAD_LOAD_BYTE + threadIdx.x * L2_CACHE_LINE;
    int idx = 0;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
        idx = block_base_offset + i * blockDim.x * L2_CACHE_LINE;
        
        if (idx > count) {
            return;
        }
        __asm__ __volatile__("prefetch.global.L2::evict_last [%0];":: "l"(&ptr[idx]) : "memory");
    }
    #endif
    }
}
void preload_to_l2cache(torch::Tensor input, int64_t offset, double ratio) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int THREAD_PER_BLOCK = 1024;
    const int BLOCK_PER_GRID = 40;
    size_t byte_size = static_cast<size_t>(input.element_size() * input.numel() * ratio);
    size_t max_preload_size = (std::min((size_t)64*1024*1024, byte_size));
    int block_one_loop_size = BLOCK_PER_GRID * THREAD_PER_BLOCK * L2_CACHE_LINE;
    int loop_size_per_thread = CEILDIV(max_preload_size, block_one_loop_size);
    dim3 grid(BLOCK_PER_GRID);
    dim3 block(THREAD_PER_BLOCK);
    char *ptr = static_cast<char*>(input.data_ptr());
    vllm::l2_preload_kernel<<<grid, block, 0, stream>>>(ptr, loop_size_per_thread, max_preload_size);
}

// #include <torch/all.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include "cuda_compat.h"
// #include "dispatch_utils.h"
// #define CEILDIV(x,y) (((x) + (y) - 1) / (y))
// #define LOOP_SIZE 1024
// #define L2_CACHE_LINE 1
// namespace vllm {
// __global__ void l2_preload_kernel(char* ptr, int count) {
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
//   constexpr int PER_THREAD_LOAD_BYTE = L2_CACHE_LINE * LOOP_SIZE;
//   int block_base_offset = blockIdx.x * blockDim.x * PER_THREAD_LOAD_BYTE + threadIdx.x * L2_CACHE_LINE;
//   int idx = 0;
// #pragma unroll
//   for (int i = 0; i < LOOP_SIZE; i++) {
//     idx = block_base_offset + i * blockDim.x * L2_CACHE_LINE;
    
//     if (idx > count) {
//       return;
//     }
//     __asm__ __volatile__("prefetch.global.L2::evict_last [%0];":: "l"(&ptr[idx]) : "memory");
//   }
// #endif
// }
// }
// void preload_to_l2cache(torch::Tensor input, int64_t offset, double ratio) {
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//     const int THREAD_PER_BLOCK = 1024;
//     const int BYTE_PER_BLOCK = LOOP_SIZE * THREAD_PER_BLOCK * L2_CACHE_LINE;
//     size_t byte_size = static_cast<size_t>(input.element_size() * input.numel() * ratio);
//     size_t max_preload_size = (std::min((size_t)64*1024*1024, byte_size));
//     dim3 grid((max_preload_size + BYTE_PER_BLOCK - 1) / BYTE_PER_BLOCK);
//     dim3 block(THREAD_PER_BLOCK);
//     char *ptr = static_cast<char*>(input.data_ptr());
//     vllm::l2_preload_kernel<<<grid, block, 0, stream>>>(ptr, max_preload_size);
// }