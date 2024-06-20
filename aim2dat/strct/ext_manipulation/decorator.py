"""Decorator for manipulation methods."""

# Standard library imports
import inspect

# Internal library imports
from aim2dat.strct.strct import Structure
from aim2dat.strct.strct_manipulation import _add_label_suffix


def external_manipulation_method(func):
    """Decorate external manipulation methods."""

    def wrapper(*args, **kwargs):
        """Wrap manipulation method and create output."""
        sig_pars = inspect.signature(func).parameters
        extracted_args = []
        for key, pos in [("structure", 0), ("change_label", len(sig_pars) - 1)]:
            if key in kwargs:
                extracted_args.append(kwargs[key])
            elif len(args) > pos:
                extracted_args.append(args[pos])
            elif key in sig_pars:
                extracted_args.append(sig_pars[key].default)
            else:
                raise TypeError(f"'{key}' not in arguments.")

        output = func(*args, **kwargs)
        if output is not None:
            new_strct, label_suffix = output
            if isinstance(new_strct, dict):
                new_strct = Structure(**new_strct)
            return _add_label_suffix(new_strct, label_suffix, extracted_args[1])
        return extracted_args[0]

    wrapper._manipulates_structure = True
    return wrapper
