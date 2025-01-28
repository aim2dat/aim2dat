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
    # Check if file_path is path to actual file or str:
    if file_format is None and not os.path.isfile(file_path):
        raise ValueError(
            "If `file_path` is not the path to a file, `file_format` needs to be set."
        )

    file_type = ""
    if file_format is not None:
        ff_split = file_format.split("-")
        try:
            modules = [importlib.import_module("aim2dat.io." + ff_split[0])]
        except ModuleNotFoundError:
            raise ValueError(f"File format '{file_format}' is not supported.")
        if len(ff_split) > 1:
            file_type = ff_split[1]
    else:
        modules = [
            importlib.import_module("aim2dat.io." + x.name)
            for x in pkgutil.iter_modules(internal_io.__path__)
            if not x.name.startswith("base") and x.name != "utils"
        ]

    for module in modules:
        # module = importlib.import_module("aim2dat.io." + m_name)
        for f_name, func in getmembers(module, isfunction):
            if not getattr(func, "_is_read_structure_method", False):
                continue
            if file_type not in f_name:
                continue
            if not file_format and not re.search(func._pattern, file_path):
                continue

            return func

    # TODO check if we can give a better error message...
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
