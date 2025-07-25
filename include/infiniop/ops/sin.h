#ifndef __INFINIOP_SIN_API_H__
#define __INFINIOP_SIN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSinDescriptor_t;

__C __export infiniStatus_t infiniopCreateSinDescriptor(infiniopHandle_t handle,
                                                         infiniopSinDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t y,
                                                         infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetSinWorkspaceSize(infiniopSinDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSin(infiniopSinDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *y,
                                         const void *x,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroySinDescriptor(infiniopSinDescriptor_t desc);

#endif