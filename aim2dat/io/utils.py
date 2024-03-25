"""Utils for read and write functions."""

# Standard library imports
import os
import re
from functools import wraps


def read_multiple(pattern):
    """Add support for a list of multiple files or folder paths (decorator)."""
    # The following cases need to be covered:
    # Single file as file name/file content/file object
    # Multiple files as List of files or folder path

    def _check_file(file_like, file_dict, re_pattern, is_strict):
        if os.path.isfile(file_like):
            file_name = os.path.split(file_like)[1]
        elif hasattr(file_like, "filename"):
            # Support AiiDA single file:
            file_name = file_like.filename
        else:
            return None

        match = None if re_pattern is None else re_pattern.match(file_name)
        if match or not is_strict:
            file_dict["file"].append(file_like)
            file_dict["file_name"].append(file_name)
            for key, val in match.groupdict().items():
                file_dict[key].append(val)

    def read_func_decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get file(s) argument:
            files = None
            if len(args) > 0:
                files = args[0]
                args = args[1:]
            else:
                for file_arg in ["file", "files", "folder_path"]:  # TODO adapt?
                    if files in kwargs:
                        files = kwargs.pop(file_arg)
                        break

            # Find files to open:
            if not isinstance(files, (list, tuple)):
                files = [files]
            re_pattern = None
            file_dict = {"file": [], "file_name": []}
            if pattern:
                re_pattern = re.compile(pattern)
                for k0 in re_pattern.groupindex.keys():
                    file_dict[k0] = []
            for file_like in files:
                if os.path.isdir(file_like):
                    [
                        _check_file(os.path.join(file_like, f0), file_dict, re_pattern, True)
                        for f0 in os.listdir(file_like)
                    ]
                else:
                    _check_file(file_like, file_dict, re_pattern, False)
            # TODO add regex to error message:
            if len(file_dict["file"]) == 0:
                raise ValueError("No files with the correct naming scheme found.")
            # TODO add check for number of files and policy to handle multiple files.
            return func(file_dict, *args, **kwargs)

        return wrapper

    return read_func_decorator


def custom_open(file, mode="r", **kwargs):
    """
    Open files by distinguishing custom file classes (such as AiiDA's SingleFileData) with an
    open function.
    """
    if hasattr(file, "open"):
        return file.open(mode=mode)
    else:
        return open(file, mode=mode, **kwargs)
