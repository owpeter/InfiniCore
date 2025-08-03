#include "rms_norm_gemm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../../../../utils/custom_types.h"

namespace op::rms_norm_gemm::cpu {

Descriptor::~Descriptor() {}

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
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t rms_norm_gemm(const RMSNormGemmInfo *info, T *c, const T *a, const T *b, const T *w, const T *bias) {
    // Use the stride information from the info structures
    size_t m = info->rms_norm_info.shape[0];  // batch dimension
    size_t k = info->rms_norm_info.shape[1];  // feature dimension
    size_t n = info->gemm_info.n;             // output dimension

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < ptrdiff_t(m); i++) {
        // RMSNorm on row i using proper strides
        float ss = 0.f;
        for (size_t j = 0; j < k; ++j) {
            size_t a_idx = i * info->rms_norm_info.x_strides[0] + j * info->rms_norm_info.x_strides[1];
            float val = utils::cast<float>(a[a_idx]);
            ss += val * val;
        }
        float rms = 1.f / std::sqrt(ss / (float)k + info->rms_norm_info.epsilon);

        std::cout << "[debug in cpu.cc] rms: " << rms << std::endl;

        // Apply weights and compute GEMM using proper strides
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.f;
            for (size_t l = 0; l < k; ++l) {
                // Get a element using proper stride
                size_t a_idx = i * info->rms_norm_info.x_strides[0] + l * info->rms_norm_info.x_strides[1];
                float a_val = utils::cast<float>(a[a_idx]);
                
                // Get weight element (weights are contiguous)
                float w_val = utils::cast<float>(w[l]);
                
                // Get b element using GEMM info strides
                size_t b_idx = l * info->gemm_info.b_matrix.row_stride + j * info->gemm_info.b_matrix.col_stride;
                float b_val = utils::cast<float>(b[b_idx]);
                
                float normed_val = a_val * w_val * rms;
                sum += normed_val * b_val;
            }
            
            // 添加偏置项
            if (info->has_bias && bias != nullptr) {
                sum += utils::cast<float>(bias[j]);
            }
            
            // Store result using proper stride
            size_t c_idx = i * info->gemm_info.c_matrix.row_stride + j * info->gemm_info.c_matrix.col_stride;
            c[c_idx] = utils::cast<T>(sum);
        }
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *c, const void *a, const void *b, const void *w, const void *bias,
    void *stream) const {
    if (_info.rms_norm_info.atype == INFINI_DTYPE_F32) {
        CHECK_STATUS(rms_norm_gemm(&_info, (float *)c, (const float *)a, (const float *)b, (const float *)w, (const float *)bias));
    } else if (_info.rms_norm_info.atype == INFINI_DTYPE_F16) {
        CHECK_STATUS(rms_norm_gemm(&_info, (fp16_t *)c, (const fp16_t *)a, (const fp16_t *)b, (const fp16_t *)w, (const fp16_t *)bias));
    } else if (_info.rms_norm_info.atype == INFINI_DTYPE_BF16) {
        CHECK_STATUS(rms_norm_gemm(&_info, (bf16_t *)c, (const bf16_t *)a, (const bf16_t *)b, (const bf16_t *)w, (const bf16_t *)bias));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
}