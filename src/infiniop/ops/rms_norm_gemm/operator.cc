#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/rms_norm_gemm.h"

#ifdef ENABLE_CPU_API
#include "cpu/rms_norm_gemm_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/rms_norm_gemm_cuda.cuh"
#endif

__C infiniStatus_t infiniopCreateRMSNormGemmDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t bias_desc,
    float epsilon) {

#define CREATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                         \
        return op::rms_norm_gemm::NAMESPACE::Descriptor::create(                       \
            handle,                                                                    \
            reinterpret_cast<op::rms_norm_gemm::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                                    \
            a_desc,                                                                    \
            b_desc,                                                                    \
            w_desc,                                                                    \
            bias_desc,                                                                 \
            epsilon);

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, cuda);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetRMSNormGemmWorkspaceSize(infiniopRMSNormGemmDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                        \
    case CASE:                                                                                      \
        *size = reinterpret_cast<op::rms_norm_gemm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, cuda);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopRMSNormGemm(infiniopRMSNormGemmDescriptor_t desc, void *workspace, size_t workspace_size,
                                   void *c, const void *a, const void *b, const void *w, const void *bias, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                               \
        return reinterpret_cast<op::rms_norm_gemm::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, c, a, b, w, bias, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, cuda);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyRMSNormGemmDescriptor(infiniopRMSNormGemmDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                  \
    case CASE:                                                                    \
        delete reinterpret_cast<op::rms_norm_gemm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, cuda);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}