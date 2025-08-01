#ifndef __RMS_NORM_GEMM_CUDA_KERNEL_H__
#define __RMS_NORM_GEMM_CUDA_KERNEL_H__

#include "../../../reduce/cuda/reduce.cuh"

template <typename T, int BLOCK_SIZE>
__global__ void rms_norm_gemm_kernel(
    T *c, const T *a, const T *b, const T *w,
    int m, int n, int k,
    float epsilon,
    int stride_a,
    int ldc,
    int ldb_row, int ldb_col)
{
    int row = blockIdx.x;
    const T *a_row = a + row * stride_a;
    
    extern __shared__ T shared_mem[];
    T *normed_a = shared_mem;

    // RMSNorm
    T ss = op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, T, T>(a_row, k);
    __syncthreads();
    
    T rms = rsqrtf(ss / k + epsilon);
    
    for (int i = threadIdx.x; i < k; i += BLOCK_SIZE) {
        normed_a[i] = a_row[i] * w[i] * rms;
    }
    __syncthreads();

    // Gemm
    for (int j = threadIdx.x; j < n; j += BLOCK_SIZE) {
        T sum = 0;
        for (int l = 0; l < k; ++l) {
            sum += normed_a[l] * b[l * ldb_row + j * ldb_col];
        }
        c[row * ldc + j] = sum;
    }
}

#endif