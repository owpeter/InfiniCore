import torch
import ctypes
from ctypes import c_uint64
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
from enum import Enum, auto

_TEST_CASES_ = [
    ((1, 3),),
    ((3, 3),),
    ((32, 20, 512),),
]

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()

_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def sin(x):
    return torch.sin(x).to(x.dtype)

def test(
    handle, device, shape, inplace=Inplace.OUT_OF_PLACE, dtype=torch.float16, sync=None
):
    x_torch_tensor = torch.rand(shape) * 2 - 1

    x = TestTensor(
        shape,
        x_torch_tensor.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=x_torch_tensor,
    )

    if inplace == Inplace.INPLACE_X:
        y = x
    else:
        y = TestTensor(shape, None, dtype, device)

    if y.is_broadcast():
        return

    print(
        f"Testing Sin on {InfiniDeviceNames[device]} with shape:{shape} dtype:{InfiniDtypeNames[dtype]} inplace: {inplace}"
    )

    ans = sin(x.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSinDescriptor(
            handle, ctypes.byref(descriptor), y.descriptor, x.descriptor
        )
    )

    for tensor in [x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSinWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_sin():
        LIBINFINIOP.infiniopSin(
            descriptor, workspace.data(), workspace.size(), y.data(), x.data(), None
        )

    lib_sin()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation("PyTorch", lambda: sin(x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_sin(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroySinDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")