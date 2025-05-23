#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cuda_compat.h"
#include "dispatch_utils.h"

#define CEILDIV(x,y) (((x) + (y) - 1) / (y))
#define LOOP_SIZE 1024
#define L2_CACHE_LINE 1

namespace vllm {

    __global__ void l2_preload_weight_kernel(char* ptr, int loop_size, int count) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            int PER_THREAD_LOAD_BYTE = L2_CACHE_LINE * loop_size;
            int block_base_offset = blockIdx.x * blockDim.x * PER_THREAD_LOAD_BYTE + threadIdx.x * L2_CACHE_LINE;
            int idx = 0;
            #pragma unroll
            for (int i = 0; i < loop_size; i++) {
                idx = block_base_offset + i * blockDim.x * L2_CACHE_LINE;
                
                if (idx >= count) {
                    return;
                }
                __asm__ __volatile__("prefetch.global.L2::evict_last [%0];":: "l"(&ptr[idx]) : "memory");
            }
        #endif
    }

    __global__ void l2_preload_weight_kvcache_kernel(
        char* ptr, 
        int loop_size, 
        int count, 
        char* kv_ptr, 
        int* valid_blocks, 
        int k_loop_size, 
        int k_block_size, 
        int kv_stride,
        int max_kv_blocks
    ) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            // 预取权重部分
            int PER_THREAD_LOAD_BYTE = L2_CACHE_LINE * loop_size;
            int block_base_offset = blockIdx.x * blockDim.x * PER_THREAD_LOAD_BYTE + threadIdx.x * L2_CACHE_LINE;
            int idx = 0;
            #pragma unroll
            for (int i = 0; i < loop_size; i++) {
                idx = block_base_offset + i * blockDim.x * L2_CACHE_LINE;
                
                if (idx >= count) {
                    break;
                }
                __asm__ __volatile__("prefetch.global.L2::evict_last [%0];":: "l"(&ptr[idx]) : "memory");
            }

            // 预取KV cache部分 - 优化版本
            if (kv_ptr == nullptr || valid_blocks == nullptr || max_kv_blocks <= 0) {
                return;
            }

            // 计算每个线程负责的block范围
            int total_threads = gridDim.x * blockDim.x;
            int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            
            // 每个线程处理连续的blocks，提高缓存局部性
            int blocks_per_thread = CEILDIV(max_kv_blocks, total_threads);
            int start_block = thread_id * blocks_per_thread;
            int end_block = min(start_block + blocks_per_thread, max_kv_blocks);
            
            // 按block为单位预取，提高效率
            for (int block_idx = start_block; block_idx < end_block; block_idx++) {
                if (block_idx >= max_kv_blocks) break;
                
                int physical_block_id = valid_blocks[block_idx];
                
                // K cache地址计算 - 基于布局 [2, num_blocks, num_kv_heads, block_size, head_size]
                char* k_block_start = kv_ptr + physical_block_id * k_block_size;
                // V cache地址
                char* v_block_start = kv_ptr + kv_stride + physical_block_id * k_block_size;
                
                // 预取整个K block，按缓存行对齐
                for (int offset = 0; offset < k_block_size; offset += L2_CACHE_LINE * 128) {
                    if (offset < k_block_size) {
                        __asm__ __volatile__("prefetch.global.L2::evict_last [%0];":: "l"(k_block_start + offset) : "memory");
                    }
                }
                
                // 预取整个V block，按缓存行对齐
                for (int offset = 0; offset < k_block_size; offset += L2_CACHE_LINE * 128) {
                    if (offset < k_block_size) {
                        __asm__ __volatile__("prefetch.global.L2::evict_last [%0];":: "l"(v_block_start + offset) : "memory");
                    }
                }
            }
        #endif
    }
}


void preload_weight_to_l2cache(torch::Tensor weight, int64_t offset, double ratio) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int THREAD_PER_BLOCK = 1024;
    const int BLOCK_PER_GRID = 40;
    size_t byte_size = static_cast<size_t>(weight.element_size() * weight.numel() * ratio);
    size_t max_preload_size = (std::min((size_t)64*1024*1024, byte_size));
    int block_one_loop_size = BLOCK_PER_GRID * THREAD_PER_BLOCK * L2_CACHE_LINE;
    int loop_size_per_thread = CEILDIV(max_preload_size, block_one_loop_size);
    dim3 grid(BLOCK_PER_GRID);
    dim3 block(THREAD_PER_BLOCK);
    char *ptr = static_cast<char*>(weight.data_ptr());
    vllm::l2_preload_weight_kernel<<<grid, block, 0, stream>>>(ptr, loop_size_per_thread, max_preload_size);
}


void preload_weight_kvcache_to_l2cache(torch::Tensor weight, torch::Tensor kv_cache, torch::Tensor kv_cache_tables, int64_t offset, double ratio) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int THREAD_PER_BLOCK = 1024;
    const int BLOCK_PER_GRID = 40;
    
    // 总预取预算：64MB
    const size_t TOTAL_PRELOAD_BUDGET = 64 * 1024 * 1024;
    
    size_t weight_size = static_cast<size_t>(weight.element_size() * weight.numel() * ratio);
    size_t max_preload_weight_size = std::min(TOTAL_PRELOAD_BUDGET, weight_size);
    
    int block_one_loop_size = BLOCK_PER_GRID * THREAD_PER_BLOCK * L2_CACHE_LINE;
    int loop_size_per_thread = CEILDIV(max_preload_weight_size, block_one_loop_size);
    
    dim3 grid(BLOCK_PER_GRID);
    dim3 block(THREAD_PER_BLOCK);
    char *ptr = static_cast<char*>(weight.data_ptr());
    
    // KV cache预取
    if (kv_cache.numel() > 0 && kv_cache_tables.numel() > 0) {
        // 计算KV cache相关参数 - 基于布局 [2, num_blocks, num_kv_heads, block_size, head_size]
        int num_kv_heads = kv_cache.size(2);
        int block_size = kv_cache.size(3);
        int head_size = kv_cache.size(4);
        
        // 每个block的字节数 (单个K或V block的大小)
        size_t k_block_size = kv_cache.element_size() * num_kv_heads * block_size * head_size;
        
        // K和V之间的步长
        size_t kv_stride = kv_cache.element_size() * kv_cache.size(1) * num_kv_heads * block_size * head_size;
        
        // 计算剩余预算用于KV cache
        size_t remaining_budget = TOTAL_PRELOAD_BUDGET - max_preload_weight_size;
        
        // 计算可以预取的KV block数量 (K+V算作一对)
        size_t kv_pair_size = k_block_size * 2;  // K block + V block
        int max_preload_kv_blocks = static_cast<int>(remaining_budget / kv_pair_size);
        
        // 限制为实际有效的block数量
        int valid_blocks_count = kv_cache_tables.numel();
        max_preload_kv_blocks = std::min(max_preload_kv_blocks, valid_blocks_count);
        
        if (max_preload_kv_blocks > 0) {
            char *kv_ptr = static_cast<char*>(kv_cache.data_ptr());
            int *valid_blocks = static_cast<int*>(kv_cache_tables.data_ptr());
            
            // 计算KV预取的loop参数 (保持与原始代码兼容)
            size_t actual_kv_preload_size = max_preload_kv_blocks * k_block_size;  // 只算K的大小
            int k_loop_size_per_thread = CEILDIV(actual_kv_preload_size, block_one_loop_size);
            
            vllm::l2_preload_weight_kvcache_kernel<<<grid, block, 0, stream>>>(
                ptr, 
                loop_size_per_thread, 
                max_preload_weight_size, 
                kv_ptr, 
                valid_blocks, 
                k_loop_size_per_thread, 
                k_block_size, 
                kv_stride,
                max_preload_kv_blocks
            );
        } else {
            // 只预取权重
            vllm::l2_preload_weight_kernel<<<grid, block, 0, stream>>>(ptr, loop_size_per_thread, max_preload_weight_size);
        }
    } else {
        // 只预取权重
        vllm::l2_preload_weight_kernel<<<grid, block, 0, stream>>>(ptr, loop_size_per_thread, max_preload_weight_size);
    }
}
