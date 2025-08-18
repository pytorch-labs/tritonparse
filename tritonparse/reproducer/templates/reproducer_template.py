import json
import torch
import sys


# Add the path to the custom Tensor implementation
TRITON_KERNELS_PATH = (
    "/home/users/yhao24/.cache/huggingface/hub/models--kernels-community--"
    "triton_kernels/snapshots/22b535b359d6c144e0152060dc6fec78da07039e/"
    "build/torch-universal/"
)
if TRITON_KERNELS_PATH not in sys.path:
    sys.path.append(TRITON_KERNELS_PATH)

try:
    from triton_kernels.tensor import Tensor, Storage
    from triton_kernels.tensor_details.layout import StridedLayout
    # {{KERNEL_IMPORT_PLACEHOLDER}}
except ImportError:
    print(f"Warning: Could not import from {TRITON_KERNELS_PATH}")
    raise ImportError(
        f"Could not import triton_kernels after adding {TRITON_KERNELS_PATH}"
    )


def create_args_from_json(json_path):
    """
    Creates a list of arguments for a kernel launch from a JSON file.

    Args:
        json_path (str): The path to the JSON file containing the kernel
                         launch information.

    Returns:
        tuple: A tuple containing the grid and a dictionary of arguments.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    # Handle data format validation and extraction
    if isinstance(data, list):
        if len(data) != 1:
            print(
                f"Error: Expected single element list, got list with {len(data)} elements"
            )
            sys.exit(1)
        data = data[0]
    elif not isinstance(data, dict):
        print(f"Error: Expected list or dict, got {type(data)}")
        sys.exit(1)

    grid = data.get("grid", [])
    args_dict = {}
    extracted_args = data.get("extracted_args", {})

    for arg_name, arg_info in extracted_args.items():
        args_dict[arg_name] = _create_arg_from_info(arg_info)

    return grid, args_dict


def _create_arg_from_info(arg_info):
    """
    Recursively creates a kernel argument from its JSON info dictionary.
    """
    arg_type = arg_info.get("type")

    if arg_type in ["int", "bool"]:
        return arg_info.get("value")

    elif arg_type == "tensor":
        dtype_str = arg_info.get("dtype")
        try:
            torch_dtype = getattr(torch, dtype_str.split(".")[-1])
        except AttributeError:
            torch_dtype = torch.float32

        shape = arg_info.get("shape", [])
        device = arg_info.get("device", "cpu")

        # Use a dummy tensor to check properties of the dtype
        tensor_props = torch.empty(0, dtype=torch_dtype)

        # Case 1: Floating point, signed integers, uint8, and bool are supported by random_()
        if tensor_props.is_floating_point() or torch_dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
        ]:
            return torch.empty(shape, dtype=torch_dtype, device=device).random_()

        # Case 2: Complex numbers need special handling
        elif tensor_props.is_complex():
            float_dtype = (
                torch.float32 if torch_dtype == torch.complex64 else torch.float64
            )
            real_part = torch.rand(shape, dtype=float_dtype, device=device)
            imag_part = torch.rand(shape, dtype=float_dtype, device=device)
            return torch.complex(real_part, imag_part)

        # Case 3: Handle other unsigned integers (like uint32) which fail with random_()
        elif "uint" in str(torch_dtype):
            return torch.randint(0, 1000, shape, dtype=torch_dtype, device=device)

        # Case 4: If we don't know how to handle the type, raise an error
        else:
            raise NotImplementedError(
                f"Random data generation not implemented for dtype: {torch_dtype}"
            )

    elif arg_type == "triton_kernels.tensor.Tensor":
        storage = _create_arg_from_info(arg_info.get("storage"))
        dtype_str = arg_info.get("dtype")
        torch_dtype = getattr(torch, dtype_str.split(".")[-1])
        return Tensor(
            storage=storage,
            shape=arg_info.get("shape"),
            shape_max=arg_info.get("shape_max"),
            dtype=torch_dtype,
        )

    elif arg_type == "triton_kernels.tensor.Storage":
        data = _create_arg_from_info(arg_info.get("data"))
        layout = _create_arg_from_info(arg_info.get("layout"))
        return Storage(data=data, layout=layout)

    elif arg_type == "StridedLayout":
        return StridedLayout(shape=arg_info.get("initial_shape"))

    else:
        print(f"Warning: Unhandled argument type '{arg_type}'. Returning None.")
        return None


if __name__ == "__main__":
    json_file = "{{JSON_PATH_PLACEHOLDER}}"
    grid, args_dict = create_args_from_json(json_file)

    print("Generated kernel arguments dictionary:")
    for name, arg in args_dict.items():
        print(f"  {name}: {arg}")
    print(f"Grid: {grid}")

    # {{KERNEL_INVOCATION_PLACEHOLDER}}

    torch.cuda.synchronize()
    print("Kernel launch finished.")
