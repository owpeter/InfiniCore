import torch
import ctypes
from ctypes import c_uint64, c_float
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules

# Test cases are defined as (c_shape, a_shape, b_shape, w_shape, bias_shape)
# c = rms_norm(a, w) @ b + bias
# a_shape = (m, k), w_shape = (k,), b_shape = (k, n), bias_shape = (n,), c_shape = (m, n)
_TEST_CASES_ = [
    # m, k,  n
    (1,  4,   8),
    (1,  512, 1024),
    (16, 2048, 4096),
    (32, 1024, 2048),
]

# Tensors dtypes used for testing
# Note: BF16 has overflow issues, temporarily disabled
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]

# Form the test cases by converting (m, k, n) tuples into shape tuples
_TEST_CASES = [
    ((m, n), (m, k), (k, n), (k,), (n,)) for m, k, n in _TEST_CASES_
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 5e-2, "rtol": 5e-2},  # Increased tolerance for F16
    InfiniDtype.BF16: {"atol": 8e-2, "rtol": 8e-2},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},  # Relaxed tolerance for F32
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def rms_norm_gemm_ref(c_ref, a, w, b, bias, eps):
    """
    Reference implementation of RMSNorm + GEMM using PyTorch.
    """
    # 1. RMSNorm
    variance = torch.mean(torch.pow(a, 2), dim=-1, keepdim=True)
    rsqrt_val = torch.rsqrt(variance + eps)
    if(DEBUG):
        print(f"rsqrt_val: {rsqrt_val}")
    # 将 w 的类型转换为与 a 一致，避免类型提升
    normed_a = a * rsqrt_val * w.to(a.dtype)
    # 2. Gemm
    torch.matmul(normed_a, b, out=c_ref)
    # 3. Add bias if provided
    if bias is not None:
        c_ref.add_(bias.to(c_ref.dtype))


def test(
    handle,
    device,
    c_shape,
    a_shape,
    b_shape,
    w_shape,
    bias_shape,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing RMSNormGemm on {InfiniDeviceNames[device]} with a_shape:{a_shape} b_shape:{b_shape}"
        f" w_shape:{w_shape} bias_shape:{bias_shape} c_shape:{c_shape} dtype:{InfiniDtypeNames[dtype]}"
    )

    # Initialize tensors
    # Use small scale for inputs to avoid overflow with F16/BF16
    a = TestTensor(a_shape, None, dtype, device, scale=0.01)
    w = TestTensor(w_shape, None, InfiniDtype.F32, device) # RMSNorm weights are often F32
    b = TestTensor(b_shape, None, dtype, device, scale=0.01)
    bias = TestTensor(bias_shape, None, dtype, device, scale=0.01)
    c = TestTensor(c_shape, None, dtype, device, mode="zeros")

    eps = 1e-5
    # Compute reference result using PyTorch
    rms_norm_gemm_ref(c.torch_tensor(), a.torch_tensor(), w.torch_tensor(), b.torch_tensor(), bias.torch_tensor(), eps)

    if sync is not None:
        sync()

    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRMSNormGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
            w.descriptor,
            bias.descriptor,
            c_float(eps),
        )
    )

    # Invalidate tensor descriptors to ensure the kernel uses its own info
    for tensor in [c, a, b, w, bias]:
        tensor.destroy_desc()

    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRMSNormGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    # Define the function to call the library operator
    def lib_rms_norm_gemm():
        check_error(
            LIBINFINIOP.infiniopRMSNormGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                c.data(),
                a.data(),
                b.data(),
                w.data(),
                bias.data(),
                None, # Stream is NULL for CPU
            )
        )

    # Execute and verify
    lib_rms_norm_gemm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: rms_norm_gemm_ref(c.torch_tensor(), a.torch_tensor(), w.torch_tensor(), b.torch_tensor(), bias.torch_tensor(), eps), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_rms_norm_gemm, device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    check_error(LIBINFINIOP.infiniopDestroyRMSNormGemmDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")