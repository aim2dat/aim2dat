"""Module to handle internal functions to read and write structures."""

# Standard library imports
import os
import importlib
from inspect import getmembers, isfunction
from typing import Union
import re

# Internal library imports
from aim2dat.ext_interfaces import _return_ext_interface_modules


def _find_io_function(file_path: str, file_format: str):
    # Check if file_path is path to actual file or str:
    if file_format is None and not os.path.isfile(file_path):
        raise ValueError(
            "If `file_path` is not the path to a file, "
            + f"`file_format` needs to be set for '{file_path}'."
        )

    funcs = [
        f
        for f in getmembers(importlib.import_module("aim2dat.io", isfunction))
        if getattr(f[1], "_is_read_structure_method", False)
    ]
    if file_format is not None:
        file_format = file_format.replace("-", "_")
        for f_name, f in funcs:
            if file_format in f_name:
                return f
        raise ValueError(f"File format '{file_format}' is not supported.")
    else:
        for f_name, f in funcs:
            if re.search(f._pattern, file_path):
                return f
        raise ValueError(
            f"Could not find a suitable io function for '{file_path}' "
            + f"- `file_format`: {file_format}."
        )


def get_structures_from_file(
    file_path: str,
    backend: str,
    file_format: str,
    backend_kwargs: dict,
) -> Union[dict, list]:
    """Get function to read structure file."""
    backend_kwargs = {} if backend_kwargs is None else backend_kwargs
    if backend == "ase":
        backend_module = _return_ext_interface_modules("ase_atoms")
        if "format" not in backend_kwargs:
            backend_kwargs["format"] = file_format
        structure_dicts = backend_module._load_structure_from_file(file_path, backend_kwargs)
    elif backend == "internal":
        func = _find_io_function(file_path, file_format)
        backend_kwargs.update(func._preset_kwargs)
        structure_dicts = func(file_path, **backend_kwargs)
        if isinstance(structure_dicts, dict):
            if "structures" in structure_dicts:
                structure_dicts = structure_dicts["structures"]
            else:
                structure_dicts = [structure_dicts]
    else:
        raise ValueError(f"Backend '{backend}' is not supported.")

    return structure_dicts
