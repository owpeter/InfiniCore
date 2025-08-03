#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "rms_norm_gemm_cuda.cuh"
#include "../cuda/kernel.cuh"

template <typename Tdata, typename Tweight, typename Tcompute, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL rms_norm_gemm_kernel(
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
    bool has_bias
) {
    rms_norm_gemm_block<Tdata, Tweight, Tcompute, BLOCK_SIZE> (
        c,b,a,w,bias,m,n,k,epsilon,stride_a,ldc,ldb_row,ldb_col,has_bias
    );
}

namespace op::rms_norm_gemm::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t bias_desc,
    float epsilon) {
    auto result = RMSNormGemmInfo::create(c_desc, a_desc, b_desc, w_desc, bias_desc, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
        void *c,                
        const void *b,          
        const void *a,
        const void *w,
        const void *bias,
        const size_t m, const size_t n, const size_t k,
        float epsilon,
        const ptrdiff_t stride_a,   
        const ptrdiff_t ldc,
        const ptrdiff_t ldb_row, 
        const ptrdiff_t ldb_col,
        infiniDtype_t atype,
        infiniDtype_t wtype,
        dim3 blocks,
        dim3 threads,
        void *stream,
        bool has_bias
) {
#define LAUNCH_KERNEL(Tdata, Tweight, Tcompute, BLOCK_SIZE)             \
    rms_norm_gemm_kernel<Tdata, Tweight, Tcompute, BLOCK_SIZE><<<blocks, threads, k * sizeof(Tcompute), (cudaStream_t)stream>>>(          \
        reinterpret_cast<Tdata *>(c),                 \
        reinterpret_cast<const Tdata *>(b),           \
        reinterpret_cast<const Tdata *>(a),           \
        reinterpret_cast<const Tweight *>(w),         \
        reinterpret_cast<const Tdata *>(bias),        \
        m, n, k,                 \
        epsilon,                                            \
        stride_a,                                       \
        ldc,                                            \
        ldb_row,                                        \
        ldb_col,                                        \
        has_bias                                        \
    )                                                                   \

    if (atype == INFINI_DTYPE_F32 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(float, float, float, BLOCK_SIZE);
    } else if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(half, half, float, BLOCK_SIZE);
    } else if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(half, float, float, BLOCK_SIZE);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *c, const void *a, const void *b, const void *w, const void *bias,
    void *stream) const {

    auto &gemm_info = _info.gemm_info;
    auto &rms_norm_info = _info.rms_norm_info;
    
    dim3 blocks(gemm_info.m);
                                         
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        dim3 threads(CUDA_BLOCK_SIZE_1024);
        launchKernel<CUDA_BLOCK_SIZE_1024>(c, b, a, w, bias,
        gemm_info.m, gemm_info.n, gemm_info.k,
        rms_norm_info.epsilon,
        rms_norm_info.x_strides[0], 
        gemm_info.c_matrix.row_stride,
        gemm_info.b_matrix.row_stride, gemm_info.b_matrix.col_stride,
        rms_norm_info.atype, rms_norm_info.wtype,
        blocks, threads, stream, _info.has_bias);
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        dim3 threads(CUDA_BLOCK_SIZE_512);
        launchKernel<CUDA_BLOCK_SIZE_512>(c, b, a, w, bias,
        gemm_info.m, gemm_info.n, gemm_info.k,
        rms_norm_info.epsilon,
        rms_norm_info.x_strides[0], 
        gemm_info.c_matrix.row_stride,
        gemm_info.b_matrix.row_stride, gemm_info.b_matrix.col_stride,
        rms_norm_info.atype, rms_norm_info.wtype,
        blocks, threads, stream, _info.has_bias);
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        dim3 threads(CUDA_BLOCK_SIZE_4096);
        launchKernel<CUDA_BLOCK_SIZE_4096>(c, b, a, w, bias,
        gemm_info.m, gemm_info.n, gemm_info.k,
        rms_norm_info.epsilon,
        rms_norm_info.x_strides[0], 
        gemm_info.c_matrix.row_stride,
        gemm_info.b_matrix.row_stride, gemm_info.b_matrix.col_stride,
        rms_norm_info.atype, rms_norm_info.wtype,
        blocks, threads, stream, _info.has_bias);
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}
}