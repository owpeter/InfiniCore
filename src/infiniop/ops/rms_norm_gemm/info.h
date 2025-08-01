#ifndef __RMS_NORM_GEMM_INFO_H__
#define __RMS_NORM_GEMM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "../gemm/info.h"
#include "../rms_norm/info.h"
#include <vector>

namespace op::rms_norm_gemm {

class RMSNormGemmInfo {
    RMSNormGemmInfo() = default;

public:
    op::rms_norm::RMSNormInfo rms_norm_info;
    op::gemm::MatmulInfo gemm_info;

    static utils::Result<RMSNormGemmInfo> create(
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t w_desc,
        float epsilon) {

        auto rms_norm_info_result = op::rms_norm::RMSNormInfo::create(a_desc, a_desc, w_desc, epsilon);
        CHECK_RESULT(rms_norm_info_result);

        auto gemm_info_result = op::gemm::MatmulInfo::create(c_desc, a_desc, b_desc, op::gemm::MatrixLayout::ROW_MAJOR);
        CHECK_RESULT(gemm_info_result);

        return utils::Result<RMSNormGemmInfo>(RMSNormGemmInfo{
            rms_norm_info_result.take(),
            gemm_info_result.take(),
        });
    }
};

} // namespace op::rms_norm_gemm

#endif // __RMS_NORM_GEMM_INFO_H__