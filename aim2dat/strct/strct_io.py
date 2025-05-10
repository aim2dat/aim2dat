"""Module to handle internal functions to read and write structures."""

# Standard library imports
import os
import importlib
from inspect import getmembers, isfunction
from typing import Union
import re


def _find_io_function(file_path: str, file_format: str):
    # Check if file_path is path to actual file or str:
    if file_format is None and not os.path.isfile(file_path):
        raise ValueError(
            "If `file_path` is not the path to a file, `file_format` needs to be set."
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
        raise ValueError("Could not find a suitable io function.")


def get_structure_from_file(
    file_path: str, file_format: str, kwargs: dict = {}
) -> Union[dict, list]:
    """Get function to read structure file."""
    func = _find_io_function(file_path, file_format)
    kwargs.update(func._preset_kwargs)
    output = func(file_path, **kwargs)
    if isinstance(output, dict) and "structures" in output:
        return output["structures"]
    return output
