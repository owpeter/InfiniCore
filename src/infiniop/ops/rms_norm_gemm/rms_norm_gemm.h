#ifndef RMS_NORM_GEMM_H
#define RMS_NORM_GEMM_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                             \
                                                                                                                          \
    namespace op::rms_norm_gemm::NAMESPACE {                                                                              \
    class Descriptor final : public InfiniopDescriptor {                                                                  \
        struct Opaque;                                                                                                    \
        Opaque *_opaque;                                                                                                  \
        RMSNormGemmInfo _info;                                                                                            \
        size_t _workspace_size;                                                                                           \
                                                                                                                          \
        Descriptor(                                                                                                       \
            Opaque *opaque,                                                                                               \
            RMSNormGemmInfo info,                                                                                         \
            size_t workspace_size,                                                                                        \
            infiniDevice_t device_type,                                                                                   \
            int device_id)                                                                                                \
            : InfiniopDescriptor{device_type, device_id},                                                                 \
              _opaque(opaque),                                                                                            \
              _info(info),                                                                                                \
              _workspace_size(workspace_size) {}                                                                          \
                                                                                                                          \
    public:                                                                                                               \
        ~Descriptor();                                                                                                    \
                                                                                                                          \
        size_t workspaceSize() const { return _workspace_size; }                                                          \
                                                                                                                          \
        static infiniStatus_t create(                                                                                     \
            infiniopHandle_t handle,                                                                                      \
            Descriptor **desc_ptr,                                                                                        \
            infiniopTensorDescriptor_t c_desc,                                                                            \
            infiniopTensorDescriptor_t a_desc,                                                                            \
            infiniopTensorDescriptor_t b_desc,                                                                            \
            infiniopTensorDescriptor_t w_desc,                                                                            \
            float epsilon);                                                                                               \
                                                                                                                          \
        infiniStatus_t calculate(                                                                                         \
            void *workspace, size_t workspace_size,                                                                       \
            void *c,                                                                                                      \
            const void *a,                                                                                                \
            const void *b,                                                                                                \
            const void *w,                                                                                                \
            void *stream) const;                                                                                          \
    };                                                                                                                    \
    }

#endif // RMS_NORM_GEMM_H