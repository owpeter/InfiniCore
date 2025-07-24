#ifndef __SIN_CPU_H__
#define __SIN_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(sin, cpu)

namespace op::sin::cpu {
typedef struct SinOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::sin(x);
    }
} SinOp;
} // namespace op::sin::cpu

#endif // __SIN_CPU_H__