#include "rms_norm_gemm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::rms_norm_gemm::cpu {

Descriptor::~Descriptor() {}

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
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t rms_norm_gemm(const RMSNormGemmInfo *info, T *c, const T *a, const T *b, const T *w) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < ptrdiff_t(info->rms_norm_info.shape[0]); i++) {
        const T *a_row = a + i * info->rms_norm_info.x_strides[0];
        
        // RMSNorm
        T ss = op::common_cpu::reduce_op::sumSquared(a_row, info->rms_norm_info.shape[1], info->rms_norm_info.x_strides[1]);
        T rms = (T)1 / std::sqrt(ss / (T)(info->rms_norm_info.shape[1]) + (T)(info->rms_norm_info.epsilon));

        std::vector<T> normed_a(info->rms_norm_info.shape[1]);
        for (size_t j = 0; j < info->rms_norm_info.shape[1]; j++) {
            normed_a[j] = a_row[j * info->rms_norm_info.x_strides[1]] * w[j] * rms;
        }

        // Gemm
        for (size_t j = 0; j < info->gemm_info.n; ++j) {
            T sum = 0;
            for (size_t k = 0; k < info->gemm_info.k; ++k) {
                sum += normed_a[k] * b[k * info->gemm_info.b_matrix.row_stride + j * info->gemm_info.b_matrix.col_stride];
            }
            c[i * info->gemm_info.c_matrix.stride + j] = sum;
        }
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *c, const void *a, const void *b, const void *w,
    void *stream) const {
    if (_info.rms_norm_info.atype == INFINI_DTYPE_F32) {
        CHECK_STATUS(rms_norm_gemm(&_info, (float *)c, (const float *)a, (const float *)b, (const float *)w));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
}