#ifndef INFINIOP_RMS_NORM_GEMM_H
#define INFINIOP_RMS_NORM_GEMM_H

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRMSNormGemmDescriptor_t;

__C __export infiniStatus_t infiniopCreateRMSNormGemmDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon);


__C __export infiniStatus_t infiniopGetRMSNormGemmWorkspaceSize(infiniopRMSNormGemmDescriptor_t desc, size_t *size);


__C __export infiniStatus_t infiniopRMSNormGemm(infiniopRMSNormGemmDescriptor_t desc,
                                   void *workspace,
                                   size_t workspace_size,
                                   void *c,
                                   const void *a,
                                   const void *b,
                                   const void *w,
                                   void *stream);

__C __export infiniStatus_t infiniopDestroyRMSNormGemmDescriptor(infiniopRMSNormGemmDescriptor_t desc);

#endif // INFINIOP_RMS_NORM_GEMM_H