# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# These utility functions are borrowed from
# https://github.com/NVlabs/edm2/blob/4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/dnnlib/util.py

"""Miscellaneous utility classes and functions."""
import importlib
import inspect
import os
import re
import sys
import types
from typing import Any, Callable, Tuple, Union

import matplotlib.pyplot as plt
import torch
from lightning.fabric import seed_everything
from torchvision.utils import make_grid


def set_random_seed(*args):
    seed = hash(args) % (1 << 31)
    seed_everything(seed, workers=True)


# Util classes
# ------------------------------------------------------------------------------------------


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


# Functionality to import modules/objects by name, and call functions by name
# ------------------------------------------------------------------------------------------


def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [
        (".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)
    ]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            get_obj_from_module(module, local_obj_name)  # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name)  # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith(
                "No module named '" + module_name + "'"
            ):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            get_obj_from_module(module, local_obj_name)  # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == "":
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: Union[str, Callable], **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name) if isinstance(func_name, str) else func_name
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: Union[str, type], **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:
    """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    assert is_top_level_function(obj)
    module = obj.__module__
    if module == "__main__":
        fname = sys.modules[module].__file__
        assert fname is not None
        module = os.path.splitext(os.path.basename(fname))[0]
    return module + "." + obj.__name__


# ----------------------------------------------------------------------------
# Print summary table of module hierarchy.


@torch.no_grad()
def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {
            id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs
        }

    # Filter out redundant entries.
    if skip_redundant:
        entries = [
            e
            for e in entries
            if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)
        ]

    # Construct table.
    rows = [
        [type(module).__name__, "Parameters", "Buffers", "Output shape", "Datatype"]
    ]
    rows += [["-----"] * len(rows[0])]
    rows += [
        [f"Input {lbl}", "-", "-", str(list(t.shape)), str(t.dtype).split(".")[-1]]
        for lbl, t in zip(["data", "noise"], inputs[:2])
    ]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = "<top-level>" if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split(".")[-1] for t in e.outputs]
        rows += [
            [
                name + (":0" if len(e.outputs) >= 2 else ""),
                str(param_size) if param_size else "-",
                str(buffer_size) if buffer_size else "-",
                (output_shapes + ["-"])[0],
                (output_dtypes + ["-"])[0],
            ]
        ]
        for idx in range(1, len(e.outputs)):
            rows += [
                [name + f":{idx}", "-", "-", output_shapes[idx], output_dtypes[idx]]
            ]
        param_total += param_size
        buffer_total += buffer_size
    rows += [["---"] * len(rows[0])]
    rows += [["Total", str(param_total), str(buffer_total), "-", "-"]]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print(
            "  ".join(
                cell + " " * (width - len(cell)) for cell, width in zip(row, widths)
            )
        )
    print()


# Plotting


def trajectory_to_imgrid(LCHW_tensor):
    assert len(LCHW_tensor.shape) == 4
    L, C, H, W = LCHW_tensor.shape
    first_mid_end = LCHW_tensor[(0, L // 2, -1), ...]
    assert first_mid_end.shape == (3, C, H, W)
    im_tens = torch.flatten(first_mid_end, 0, 1).unsqueeze(1)
    assert im_tens.shape == (3 * C, 1, H, W)
    im_grid = make_grid(im_tens, nrow=C, normalize=True, scale_each=True)
    return im_grid


def value_histogram(LCHW_tensor, bins=255, **hist_kwargs):
    fig, ax = plt.subplots()
    flt_values = LCHW_tensor.flatten().numpy()
    ax.hist(flt_values, bins=bins, **hist_kwargs)
    return fig


def count_parameters_and_memory(model):
    # Initialize counters
    num_params = 0
    memory_size_bytes = 0

    # Iterate over all parameters
    for param in model.parameters():
        # Number of parameters
        num_params += param.numel()

        # Size in bytes (e.g., float32 is 4 bytes)
        memory_size_bytes += param.numel() * param.element_size()

    # Convert memory size to megabytes
    memory_size_mb = memory_size_bytes / (1024**2)

    return num_params, memory_size_mb
