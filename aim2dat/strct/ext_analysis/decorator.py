"""
Decorator for analysis methods that are not hosted in the Structure and StructureOperations class.
"""

# Standard library imports
import inspect
from functools import wraps

# Internal library imports
from aim2dat.strct.strct import _check_calculated_properties


def external_analysis_method(func):
    """
    Decorate external analysis methods.
    """

    @wraps(func)
    def perform_strct_analysis(*args, **kwargs):
        func_args = {}
        for idx, (name, p0) in enumerate(inspect.signature(func).parameters.items()):
            if idx < len(args):
                func_args[name] = args[idx]
                if name in kwargs:
                    raise TypeError(f"{func.__name__}() got multiple values for argument '{name}'")
            elif name in kwargs:
                func_args[name] = kwargs[name]
        structure = func_args.pop("structure")
        return _check_calculated_properties(structure, func, func_args)

    perform_strct_analysis._is_analysis_method = True
    return perform_strct_analysis
