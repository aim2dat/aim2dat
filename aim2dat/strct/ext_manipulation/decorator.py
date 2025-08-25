"""Decorator for manipulation methods."""

# Standard library imports
import inspect
from functools import wraps

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.manipulation.utils import _add_label_suffix


def external_manipulation_method(func):
    """Decorate external manipulation methods."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrap manipulation method and create output."""
        sig_pars = inspect.signature(func).parameters
        extracted_args = []
        for key in ["structure", "change_label"]:
            if key not in sig_pars:
                raise TypeError(f"`{key}` not in function arguments.")
            idx = list(sig_pars.keys()).index(key)
            if idx < len(args):
                extracted_args.append(args[idx])
            elif key in kwargs:
                extracted_args.append(kwargs[key])
            else:
                extracted_args.append(sig_pars[key].default)

        output = func(*args, **kwargs)
        if output is not None:
            new_strct, label_suffix = output
            if isinstance(new_strct, dict):
                new_strct = Structure(**new_strct)
            return _add_label_suffix(new_strct, label_suffix, extracted_args[1])
        return extracted_args[0]

    wrapper._manipulates_structure = True
    return wrapper
