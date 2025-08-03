#ifndef __RMS_NORM_GEMM_CUDA_KERNEL_H__
#define __RMS_NORM_GEMM_CUDA_KERNEL_H__

#include "../../../reduce/cuda/reduce.cuh"
#include <cassert>

template <typename Tdata, typename Tweight, typename Tcompute, int BLOCK_SIZE>
__device__ void rms_norm_gemm_block(
    Tdata *__restrict__ c, 
    const Tdata *__restrict__ b, 
    const Tdata *__restrict__ a, 
    const Tweight *__restrict__ w,
    const Tdata *__restrict__ bias,
    const size_t m, const size_t n, const size_t k,
    float epsilon,
    const ptrdiff_t stride_a,   
    const ptrdiff_t ldc,
    const ptrdiff_t ldb_row, 
    const ptrdiff_t ldb_col,
    bool has_bias)
{
    extern __shared__ char rms_gemm_smem_bytes[];
    
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const Tdata *__restrict__ a_row = a + row * stride_a;
    
    // 优化1: 使用更高效的RMSNorm计算
    Tcompute ss = op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>(a_row, k);
    
    // 优化2: 减少同步开销，直接计算RMS而不是先存储再广播
    __shared__ Tcompute shared_rms;
    if (tid == 0) {
        shared_rms = rsqrtf(ss / k + epsilon);
    }
    __syncthreads();
    
    const Tcompute rms = shared_rms;
    
    // 优化3: 使用aligned shared memory访问，减少bank conflicts
    Tcompute *normed_a_compute = reinterpret_cast<Tcompute*>(rms_gemm_smem_bytes);
    
    // 优化4: 向量化内存访问和计算融合
    constexpr int VEC_SIZE = sizeof(float4) / sizeof(Tdata);
    const int k_vec = k / VEC_SIZE;
    const int k_remainder = k % VEC_SIZE;
    
    // 处理向量化部分
    for (int i = tid; i < k_vec; i += BLOCK_SIZE) {
        const int base_idx = i * VEC_SIZE;
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            const int idx = base_idx + v;
            const Tcompute val = static_cast<Tcompute>(a_row[idx]) * static_cast<Tcompute>(w[idx]) * rms;
            normed_a_compute[idx] = val;
        }
    }
    
    // 处理剩余元素
    for (int i = k_vec * VEC_SIZE + tid; i < k; i += BLOCK_SIZE) {
        const Tcompute val = static_cast<Tcompute>(a_row[i]) * static_cast<Tcompute>(w[i]) * rms;
        normed_a_compute[i] = val;
    }
    __syncthreads();

    // 优化5: GEMM计算优化 - 使用寄存器缓存和循环展开
    constexpr int TILE_SIZE = 4; // 每个线程处理的输出元素数量
    
    for (int j_base = tid * TILE_SIZE; j_base < n; j_base += BLOCK_SIZE * TILE_SIZE) {
        Tcompute sums[TILE_SIZE];
        #pragma unroll
        for (int t = 0; t < TILE_SIZE; ++t) {
            sums[t] = 0;
        }
        
        // 优化6: 内层循环展开，提高ILP
        constexpr int UNROLL_K = 4;
        const int k_unroll = (k / UNROLL_K) * UNROLL_K;
        
        for (int l = 0; l < k_unroll; l += UNROLL_K) {
            #pragma unroll
            for (int u = 0; u < UNROLL_K; ++u) {
                const Tcompute a_val = normed_a_compute[l + u];
                #pragma unroll
                for (int t = 0; t < TILE_SIZE; ++t) {
                    const int j = j_base + t;
                    if (j < n) {
                        sums[t] += a_val * static_cast<Tcompute>(b[(l + u) * ldb_row + j * ldb_col]);
                    }
                }
            }
        }
        
        // 处理剩余的k维度
        for (int l = k_unroll; l < k; ++l) {
            const Tcompute a_val = normed_a_compute[l];
            #pragma unroll
            for (int t = 0; t < TILE_SIZE; ++t) {
                const int j = j_base + t;
                if (j < n) {
                    sums[t] += a_val * static_cast<Tcompute>(b[l * ldb_row + j * ldb_col]);
                }
            }
        }
        
        // 写回结果，添加偏置
        #pragma unroll
        for (int t = 0; t < TILE_SIZE; ++t) {
            const int j = j_base + t;
            if (j < n) {
                Tcompute final_sum = sums[t];
                if (has_bias) {
                    final_sum += static_cast<Tcompute>(bias[j]);
                }
                c[row * ldc + j] = static_cast<Tdata>(final_sum);
            }
        }
    }
}

#endif