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
    const size_t m, const size_t n, const size_t k,
    float epsilon,
    const ptrdiff_t stride_a,   
    const ptrdiff_t ldc,
    const ptrdiff_t ldb_row, 
    const ptrdiff_t ldb_col)
{
    extern __shared__ char rms_gemm_smem_bytes[];
    
    int row = blockIdx.x;
    auto a_row = a + row * stride_a;
    
    // RMSNorm
    Tcompute ss = op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>(a_row, k);
    __syncthreads();
    
    // Broadcast the result to all threads in the block
    __shared__ Tcompute shared_rms;
    if (threadIdx.x == 0) {
        shared_rms = rsqrtf(ss / k + epsilon);
    }
    __syncthreads();
    
    Tcompute rms = shared_rms;
    
    // Use the shared memory for normed_a after the reduction is complete
    Tdata *normed_a = reinterpret_cast<Tdata*>(rms_gemm_smem_bytes);
    
    for (int i = threadIdx.x; i < k; i += BLOCK_SIZE) {
        Tcompute val = static_cast<Tcompute>(a_row[i]) * static_cast<Tcompute>(w[i]) * rms;
        normed_a[i] = static_cast<Tdata>(val);
    }
    __syncthreads();

    // Gemm
    for (int j = threadIdx.x; j < n; j += BLOCK_SIZE) {
        Tcompute sum = 0;
        for (int l = 0; l < k; ++l) {
            sum += static_cast<Tcompute>(normed_a[l]) * static_cast<Tcompute>(b[l * ldb_row + j * ldb_col]);
        }
        c[row * ldc + j] = static_cast<Tdata>(sum);
    }
}

#endif