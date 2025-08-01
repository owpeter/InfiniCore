#include "../../../devices/nvidia/nvidia_common.cuh"
#include "rms_norm_gemm_cuda.cuh"
#include "../cuda/kernel.cuh"

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
    float epsilon) {
    auto result = RMSNormGemmInfo::create(c_desc, a_desc, b_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *c, const void *a, const void *b, const void *w,
    void *stream) const {

    auto &gemm_info = _info.gemm_info;
    auto &rms_norm_info = _info.rms_norm_info;
    
    dim3 threads(256);
    dim3 blocks(gemm_info.m);

    if (rms_norm_info.atype == INFINI_DTYPE_F32) {
        rms_norm_gemm_kernel<float, 256><<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (float*)c, (const float*)a, (const float*)b, (const float*)w,
            gemm_info.m, gemm_info.n, gemm_info.k,
            rms_norm_info.epsilon,
            rms_norm_info.x_strides[0], 
            gemm_info.c_matrix.row_stride,
            gemm_info.b_matrix.row_stride, gemm_info.b_matrix.col_stride
        );
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
}