"""
Decorator for analysis methods that are not hosted in the Structure and StructureOperations class.
"""

# Standard library imports
import inspect
from functools import wraps

# Internal library imports
from aim2dat.strct.structure import _check_calculated_properties


def external_analysis_method(attr_mapping):
    """Decorate external analysis methods."""

    def decorator(func):
        func._is_analysis_method = True

        @wraps(func)
        def perform_strct_analysis(*args, **kwargs):
            sig_pars = inspect.signature(func).parameters
            func_args = {}
            for idx, arg in enumerate(args):
                func_args[list(sig_pars.keys())[idx]] = arg
            func_args.update(kwargs)
            for keyw, par in sig_pars.items():
                if keyw not in func_args and par.default is not par.empty:
                    func_args[keyw] = par.default
            structure = func_args.pop("structure")
            return _check_calculated_properties(structure, func, func_args, attr_mapping)

        return perform_strct_analysis

    return decorator
