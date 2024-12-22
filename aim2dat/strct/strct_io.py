"""Module to handle internal functions to read and write structures."""

# Standard library imports
import os
import pkgutil
import importlib
from inspect import getmembers, isfunction
from typing import Union
import re

# internal library imports
import aim2dat.io as internal_io


def _find_io_function(file_path, file_format):
    file_type = ""

    if file_format is not None:
        ff_split = file_format.split("-")
        m_names = [ff_split[0]]
        if len(ff_split) > 1:
            file_type = ff_split[1]
    else:
        m_names = [x.name for x in pkgutil.iter_modules(internal_io.__path__)]

    for m_name in m_names:
        if m_name.startswith("base") or m_name == "utils":
            continue
        module = importlib.import_module("aim2dat.io." + m_name)
        for f_name, func in getmembers(module, isfunction):
            if not getattr(func, "_is_read_structure_method", False):
                continue
            if file_type not in f_name:
                continue
            if os.path.exists(file_path) and not re.search(func._pattern, file_path):
                continue

            return func

    # TODO check if we can give a better error message...
    raise ValueError("File format is not supported.")


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
